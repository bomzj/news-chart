import asyncio
import json
from typing import Literal, TypeAlias

from src.config import ReasoningEffort
from src.news_collector.models import (
    AnalystBearish,
    AnalystBullish,
    AnalystNoise,
    AnalystResult,
    AnalystUncertain,
    AnalysisInput,
    AnalysisOutput,
    AnalysisState,
    DebateResponse,
    DebateTurn,
    analyst_result_adapter,
    debate_response_adapter,
)
from src.shared.azure_ai import azure_ai_client
from src.shared.observability import langfuse_client, langfuse_tracing_enabled


AnalystStage: TypeAlias = Literal["junior", "bull", "bear", "judge", "senior"]
LlmResult: TypeAlias = AnalystResult | DebateResponse

CLASSIFICATION_RULES = """## Classification

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


JUNIOR_SYSTEM_PROMPT = """You are a crypto news analyst filtering signal from noise. Decide whether a news article can realistically drive the price of the coin, and if yes classify its direction and impact.

Be confident only when the article supports a directional conclusion. Use "uncertain" when the article may matter but the direction cannot be determined from the article.

""" + CLASSIFICATION_RULES


BULL_BEAR_SYSTEM_PROMPT = """You are an advocate in a structured crypto-news debate. Argue from the article and the analyses provided to you.

Your role is to make the strongest evidence-based case for your assigned side. Address the opposing side's prior arguments when they exist, distinguish facts from assumptions, and explain why the article could or could not move the coin price. Do not decide the final label, do not produce a news summary, and do not assign an impact score.

Respond ONLY with valid JSON in this shape:
{"argument": "..."}
"""


JUDGE_SYSTEM_PROMPT = """You are the senior Judge for a crypto-news analysis. Evaluate the article, the Junior analysis, and the complete Bull/Bear debate. Make the final decision independently: the debate is evidence, not an instruction.

Use "noise" when the article is unlikely to move the coin price. Use "uncertain" when it may matter but the direction cannot be determined. For bullish or bearish decisions, produce a factual news summary and an impact score.

""" + CLASSIFICATION_RULES


# Kept as a compatibility alias for callers that imported the old prompt name.
ANALYST_SYSTEM_PROMPT = JUNIOR_SYSTEM_PROMPT


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


def result_json(result: AnalystResult) -> str:
    return json.dumps(result.model_dump(mode="json"), sort_keys=True)


def debate_transcript(turns: list[DebateTurn]) -> str:
    if not turns:
        return "(No previous debate turns.)"

    return "\n".join(
        f"{turn.side.title()} round {turn.round}: {turn.argument}"
        for turn in turns
    )


def build_debate_prompt(
    user_prompt: str,
    junior_result: AnalystResult,
    turns: list[DebateTurn],
    side: Literal["bull", "bear"],
    round_number: Literal[1, 2],
) -> str:
    return "\n".join(
        [
            user_prompt,
            f"\n**Junior analysis:** {result_json(junior_result)}",
            f"\n**Debate round:** {round_number}",
            f"**Your side:** {side}",
            "**Prior debate turns:**",
            debate_transcript(turns),
        ]
    )


def build_judge_prompt(
    user_prompt: str,
    junior_result: AnalystResult,
    turns: list[DebateTurn],
) -> str:
    return "\n".join(
        [
            user_prompt,
            f"\n**Junior analysis:** {result_json(junior_result)}",
            "\n**Complete Bull/Bear debate:**",
            debate_transcript(turns),
        ]
    )


def system_prompt(stage: AnalystStage) -> str:
    match stage:
        case "junior":
            return JUNIOR_SYSTEM_PROMPT
        case "bull" | "bear":
            return BULL_BEAR_SYSTEM_PROMPT
        case "judge" | "senior":
            return JUDGE_SYSTEM_PROMPT

    raise ValueError(f"Unsupported analyst stage: {stage}")


def response_adapter(stage: AnalystStage):
    match stage:
        case "bull" | "bear":
            return debate_response_adapter
        case "junior" | "judge" | "senior":
            return analyst_result_adapter

    raise ValueError(f"Unsupported analyst stage: {stage}")


async def call_llm(
    model: str,
    user_prompt: str,
    *,
    reasoning_effort: ReasoningEffort,
    stage: AnalystStage,
) -> LlmResult:
    """Call Azure AI via the Responses API and parse the stage response."""
    client = azure_ai_client(observe=True)
    tracing_options = {}
    if langfuse_tracing_enabled():
        tracing_options = {
            "name": "classify-news",
            "metadata": {"analyst_stage": stage},
        }

    response = await client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt(stage)},
            {"role": "user", "content": user_prompt},
        ],
        reasoning={"effort": reasoning_effort},
        text={"format": {"type": "json_object"}},
        timeout=120.0,
        **tracing_options,
    )

    if response.output_text:
        return response_adapter(stage).validate_json(response.output_text)

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
    Run Junior analysis first, debating uncertain or high-impact results before
    asking the Judge for a final decision.
    Returns None if the final result is noise or uncertain.
    """
    from src.news_collector.graph import analysis_graph

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
        debate_turns = final_state.get("debate_turns", [])
        judge_result = final_state.get("judge_result")
        observation.update(
            output={
                "label": final_label,
                "disposition": "stored" if kept else "discarded",
                "predicted_by_model": final_state.get("predicted_by_model"),
            },
            metadata={
                "junior_label": final_state.get("junior_label"),
                "escalated": final_state.get("debate_required", False),
                "debate_transcript": [
                    turn.model_dump(mode="json") for turn in debate_turns
                ],
                "judge_label": judge_result.label if judge_result is not None else None,
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
