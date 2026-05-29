import asyncio
import json
import logging

import httpx

from src.config import app_config, secrets

logger = logging.getLogger(__name__)

CONDENSE_SYSTEM_PROMPT = """You are a news condensation assistant. Summarize the following news article text while preserving all key facts, entities, numbers, and quotes. Output ONLY the condensed text — no preamble, no labels."""


async def condense_texts(texts: list[str]) -> list[str]:
    """
    Condense texts exceeding max_full_text_chars via nano LLM.
    Short texts pass through unchanged.
    """
    limit = app_config().pipeline.max_full_text_chars

    async def _maybe_condense(text: str) -> str:
        if len(text) <= limit:
            return text
        return await _condense_single(text, limit)

    return await asyncio.gather(*[_maybe_condense(t) for t in texts])


async def _condense_single(text: str, limit: int) -> str:
    """Call nano LLM to summarize a single oversized article."""
    sec = secrets()
    cfg = app_config().agents

    url = f"{sec.azure_ai_endpoint}/openai/responses?api-version={cfg.api_version}"

    user_prompt = (
        f"Condense this article to under {limit} characters while keeping all important facts:\n\n{text}"
    )

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers={"api-key": sec.azure_ai_api_key},
                json={
                    "model": cfg.nano_deployment,
                    "input": [
                        {"role": "system", "content": CONDENSE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                },
                # Condensation is a simple task, 60s is generous
                timeout=60.0,
            )
            response.raise_for_status()
            output = response.json()["output"]

        for item in output:
            if item["type"] == "message":
                for content in item["content"]:
                    if content["type"] == "output_text":
                        condensed = content["text"]
                        # Hard cap as safety net
                        return condensed[:limit]

        logger.warning("No text output from condense LLM, truncating instead")
        return text[:limit]

    except (httpx.HTTPError, httpx.TimeoutException, json.JSONDecodeError) as exc:
        logger.warning("Condense LLM failed: %s, truncating instead", exc)
        return text[:limit]
