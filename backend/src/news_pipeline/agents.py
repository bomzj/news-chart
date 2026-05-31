import asyncio
import json

import httpx

from src.config import app_config, secrets
from src.news_pipeline.models import AnalysisInput, AnalysisOutput, AnalysisState


ANALYST_SYSTEM_PROMPT = """You are a crypto news analyst filtering signal from noise. Your job: decide if a news article can realistically drive the price of the coin, and if yes — classify it.

## Decision: Keep or Discard

If the news is noise (opinions, minor project updates, influencer drama, vague rumors, repetitive coverage of already-priced-in events) → return:
{"discard": true}

If the news has real potential to move price → return:
{"news_summary": "...", "sentiment": "bullish" or "bearish", "impact": 1-3, "confidence": 0.0-1.0}

## Sentiment (binary — pick one)

- "bullish" — news creates buying pressure or positive price expectation
- "bearish" — news creates selling pressure or negative price expectation

There is no neutral. If you can't determine direction, it's noise → discard.

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
- confidence: how certain you are about your sentiment call (0.0-1.0)
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
            parts.append(
                f"- {summary} (sentiment={sentiment}, "
                f"1h_delta={delta_1h}, 24h_delta={delta_24h})"
            )

    return "\n".join(parts)


async def _call_llm(deployment: str, user_prompt: str) -> dict:
    """Call Azure AI via the Responses API and parse JSON response."""
    sec = secrets()
    cfg = app_config().agents

    url = f"{sec.azure_ai_endpoint}/openai/responses?api-version={cfg.api_version}"

    async with httpx.AsyncClient() as client:
        response = await client.post(
            url,
            headers={"api-key": sec.azure_ai_api_key},
            json={
                "model": deployment,
                "input": [
                    {"role": "system", "content": ANALYST_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "reasoning": {"effort": cfg.reasoning_effort},
                "text": {"format": {"type": "json_object"}},
            },
            timeout=120.0,
        )
        response.raise_for_status()
        output = response.json()["output"]

    for item in output:
        if item["type"] == "message":
            for content in item["content"]:
                if content["type"] == "output_text":
                    return json.loads(content["text"])

    raise ValueError("No text output in Responses API response")


async def analyze_single(input: AnalysisInput) -> AnalysisOutput | None:
    """
    Run the LangGraph analyst graph for a single news item.
    Junior analyst first, conditional escalation to senior if low confidence.
    Returns None if the news is discarded as noise.
    """
    from src.news_pipeline.graph import analysis_graph

    user_prompt = _build_user_prompt(input)

    initial_state: AnalysisState = {"user_prompt": user_prompt}
    final_state = await analysis_graph.ainvoke(initial_state)

    result = final_state.get("llm_result")
    if result is None or result.get("discard"):
        return None

    return AnalysisOutput(
        news_summary=result["news_summary"][:400],
        sentiment=result["sentiment"],
        impact=result["impact"],
        confidence=result["confidence"],
        predicted_by_model=final_state["predicted_by_model"],
    )


async def analyze_batch(inputs: list[AnalysisInput]) -> list[AnalysisOutput | None]:
    """Run all news analyses in parallel via LangGraph. None entries = discarded noise."""
    return await asyncio.gather(*[analyze_single(inp) for inp in inputs])
