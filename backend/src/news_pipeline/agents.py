import asyncio
from typing import Literal

from src.config import app_config
from src.news_pipeline.models import (
    AnalystBearish,
    AnalystBullish,
    AnalystNoise,
    AnalystResult,
    AnalystUncertain,
    AnalysisInput,
    AnalysisOutput,
    AnalysisState,
    analyst_result_adapter,
)
from src.shared.azure_ai import azure_ai_client
from src.shared.observability import langfuse_client, langfuse_tracing_enabled


ANALYST_SYSTEM_PROMPT = """You are a crypto news analyst filtering signal from noise. Your job: decide if a news article can realistically drive the price of the coin, and if yes — classify it.

## Classification

- "noise" — opinions, minor project updates, influencer drama, vague rumors, repetitive coverage of already-priced-in events, or anything unlikely to move the coin price. Return only:
{"label": "noise"}

- "uncertain" — potentially relevant news where the price direction cannot be determined from the article. Return only:
{"label": "uncertain"}

- "bullish" — news creates buying pressure or positive price expectation. Return:
{"label": "bullish", "news_summary": "...", "impact": 1-3}

- "bearish" — news creates selling pressure or negative price expectation. Return:
{"label": "bearish", "news_summary": "...", "impact": 1-3}

Use "uncertain" when the article may matter but its direction is ambiguous. Do not use "noise" as a substitute for uncertainty.

## Impact Scale (1-3)

### 1 — Notable (moves price for hours)
Examples:
- Coin gets listed on a major exchange (Binance, Coinbase, Kraken)
- Token burn announced or executed (meaningful % of supply)
- Large whale accumulation/dump detected (>$50M move)
- Successful protocol upgrade or hard fork completed
- Major partnership announced (real integration, not just MOU)
- Mining difficulty adjustment or hash rate spike

### 2 — High (moves price for days)
Examples:
- ETF approval or rejection for a major coin
- Major DeFi hack or exploit ($100M+ stolen)
- Large country bans or legalizes crypto trading
- Top institutional player enters or exits market (BlackRock, Fidelity buying/selling)
- Exchange insolvency or withdrawal freezes (major exchange)
- Major staking/unstaking unlock event (billions worth)

### 3 — Extreme (moves entire market for weeks)
Examples:
- Fed interest rate decision (cut or hike)
- Major stablecoin depeg event (USDT, USDC losing peg)
- Exchange collapse (FTX-level event)
- Coordinated international regulatory crackdown
- Global banking crisis affecting crypto custody/rails
- Bitcoin halving event

## Output Rules

- news_summary: 1-3 sentences, max 400 characters, factual
- Include news_summary and impact only for bullish or bearish labels
- Include no extra fields for noise or uncertain labels
- Respond ONLY with valid JSON, nothing else"""


def _build_user_prompt(input: AnalysisInput) -> str:
    parts = [
        f"**Title:** {input.raw_news.title}",
        f"**Source:** {input.raw_news.source}",
        f"**Published:** {input.raw_news.published_at.isoformat()}",
        f"**Content:** {input.raw_news.full_text}",
    ]

    if input.similar_context:
        parts.append("\n**Similar past news for context:**")
        for ctx in input.similar_context[:3]:
            summary = ctx.get("news_summary", "N/A")
            sentiment = ctx.get("sentiment", "N/A")
            delta_1h = ctx.get("realized_price_delta_pct_1h")
            delta_24h = ctx.get("realized_price_delta_pct_24h")
            delta_7d = ctx.get("realized_price_delta_pct_7d")
            delta_30d = ctx.get("realized_price_delta_pct_30d")
            parts.append(
                f"- {summary} (sentiment={sentiment}, "
                f"1h_delta={delta_1h}, 24h_delta={delta_24h}, "
                f"7d_delta={delta_7d}, 30d_delta={delta_30d})"
            )

    return "\n".join(parts)


async def _call_llm(
    deployment: str,
    user_prompt: str,
    *,
    stage: Literal["junior", "senior"],
) -> AnalystResult:
    """Call Azure AI via the Responses API and parse JSON response."""
    cfg = app_config().agents
    client = azure_ai_client(cfg.api_version, observe=True)
    tracing_options = {}
    if langfuse_tracing_enabled():
        tracing_options = {
            "name": "classify-news",
            "metadata": {"analyst_stage": stage},
        }

    response = await client.responses.create(
        model=deployment,
        input=[
            {"role": "system", "content": ANALYST_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        reasoning={"effort": cfg.reasoning_effort},
        text={"format": {"type": "json_object"}},
        timeout=120.0,
        **tracing_options,
    )

    if response.output_text:
        return analyst_result_adapter.validate_json(response.output_text)

    raise ValueError("No text output in Responses API response")


def analysis_trace_input(input: AnalysisInput) -> dict:
    context = [
        {
            "news_summary": item.get("news_summary"),
            "sentiment": item.get("sentiment"),
            "realized_price_delta_pct_1h": item.get("realized_price_delta_pct_1h"),
            "realized_price_delta_pct_24h": item.get("realized_price_delta_pct_24h"),
            "realized_price_delta_pct_7d": item.get("realized_price_delta_pct_7d"),
            "realized_price_delta_pct_30d": item.get("realized_price_delta_pct_30d"),
        }
        for item in input.similar_context[:3]
    ]
    return {
        "title": input.raw_news.title,
        "source": input.raw_news.source,
        "published_at": input.raw_news.published_at.isoformat(),
        "content": input.raw_news.full_text,
        "similar_context": context,
    }


async def analyze_single(input: AnalysisInput) -> AnalysisOutput | None:
    """
    Run the LangGraph analyst graph for a single news item.
    Junior analyst first, escalating uncertain results to Senior.
    Returns None if the final result is noise or uncertain.
    """
    from src.news_pipeline.graph import analysis_graph

    user_prompt = _build_user_prompt(input)
    initial_state: AnalysisState = {"user_prompt": user_prompt}
    with langfuse_client().start_as_current_observation(
        as_type="chain",
        name="analyze-news",
        input=analysis_trace_input(input),
        metadata={"component": "news-analyst"},
    ) as observation:
        final_state = await analysis_graph.ainvoke(initial_state)
        result = final_state.get("llm_result")
        final_label = result.label if result is not None else None
        kept = final_label in {"bullish", "bearish"}
        observation.update(
            output={
                "label": final_label,
                "disposition": "stored" if kept else "discarded",
                "predicted_by_model": final_state.get("predicted_by_model"),
            },
            metadata={
                "junior_label": final_state.get("junior_label"),
                "escalated": final_state.get("junior_label") == "uncertain",
                "final_label": final_label,
            },
        )

        match result:
            case AnalystBullish() | AnalystBearish():
                return AnalysisOutput(
                    news_summary=result.news_summary[:400],
                    sentiment=result.label,
                    impact=result.impact,
                    predicted_by_model=final_state["predicted_by_model"],
                )
            case AnalystNoise() | AnalystUncertain() | None:
                return None


async def analyze_batch(inputs: list[AnalysisInput]) -> list[AnalysisOutput | None]:
    """Run news analyses in parallel; None entries are discarded results."""
    return await asyncio.gather(*[analyze_single(inp) for inp in inputs])
