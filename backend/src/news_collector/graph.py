from typing import Literal

from langgraph.graph import END, StateGraph

from src.config import app_config
from src.news_collector.agents import (
    build_debate_prompt,
    build_judge_prompt,
    call_llm,
)
from src.news_collector.models import (
    AnalystBearish,
    AnalystBullish,
    AnalystNoise,
    AnalystResult,
    AnalystUncertain,
    AnalysisState,
    DebateResponse,
    DebateTurn,
)


DebateSide = Literal["bull", "bear"]
DebateRound = Literal[1, 2]


def analyst_result_from_state(state: AnalysisState) -> AnalystResult:
    result = state.get("junior_result")
    if result is None:
        result = state.get("llm_result")

    if isinstance(
        result,
        (AnalystNoise, AnalystUncertain, AnalystBullish, AnalystBearish),
    ):
        return result

    raise TypeError("Analysis state does not contain a valid analyst result")


def debate_required(result: AnalystResult) -> bool:
    match result:
        case AnalystNoise():
            return False
        case AnalystUncertain():
            return True
        case AnalystBullish(impact=impact) | AnalystBearish(impact=impact):
            return impact >= 2

    raise ValueError(f"Unsupported analyst result: {result!r}")


async def junior_analyst(state: AnalysisState) -> AnalysisState:
    """Evaluate the news item first as the Junior analyst."""
    cfg = app_config().llm
    result = await call_llm(
        cfg.name,
        state["user_prompt"],
        reasoning_effort=cfg.reasoning_effort.junior_analysis,
        stage="junior",
    )
    if not isinstance(
        result,
        (AnalystNoise, AnalystUncertain, AnalystBullish, AnalystBearish),
    ):
        raise TypeError("Junior analyst returned a non-classification response")

    return {
        "llm_result": result,
        "junior_result": result,
        "junior_label": result.label,
        "debate_required": debate_required(result),
        "predicted_by_model": cfg.name,
    }


def route_after_junior(state: AnalysisState) -> str:
    """Route only uncertain or high/extreme-impact Junior results to debate."""
    result = analyst_result_from_state(state)
    if debate_required(result):
        return "bull_round_1"

    return END


async def debate_turn(
    state: AnalysisState,
    side: DebateSide,
    round_number: DebateRound,
) -> AnalysisState:
    """Run one ordered Bull or Bear turn and append it to the transcript."""
    cfg = app_config().llm
    junior_result = analyst_result_from_state(state)
    turns = state.get("debate_turns", [])
    result = await call_llm(
        cfg.name,
        build_debate_prompt(
            state["user_prompt"],
            junior_result,
            turns,
            side,
            round_number,
        ),
        reasoning_effort=cfg.reasoning_effort.bull_bear,
        stage=side,
    )
    if not isinstance(result, DebateResponse):
        raise TypeError(f"{side} analyst returned a non-debate response")

    return {
        "debate_turns": [
            *turns,
            DebateTurn(side=side, round=round_number, argument=result.argument),
        ],
        "predicted_by_model": cfg.name,
    }


async def bull_round_1(state: AnalysisState) -> AnalysisState:
    return await debate_turn(state, "bull", 1)


async def bear_round_1(state: AnalysisState) -> AnalysisState:
    return await debate_turn(state, "bear", 1)


async def bull_round_2(state: AnalysisState) -> AnalysisState:
    return await debate_turn(state, "bull", 2)


async def bear_round_2(state: AnalysisState) -> AnalysisState:
    return await debate_turn(state, "bear", 2)


async def judge_analyst(state: AnalysisState) -> AnalysisState:
    """Make the final classification after the complete Bull/Bear debate."""
    cfg = app_config().llm
    junior_result = analyst_result_from_state(state)
    turns = state.get("debate_turns", [])
    result = await call_llm(
        cfg.name,
        build_judge_prompt(state["user_prompt"], junior_result, turns),
        reasoning_effort=cfg.reasoning_effort.judge,
        stage="judge",
    )
    if not isinstance(
        result,
        (AnalystNoise, AnalystUncertain, AnalystBullish, AnalystBearish),
    ):
        raise TypeError("Judge returned a non-classification response")

    return {
        "llm_result": result,
        "judge_result": result,
        "predicted_by_model": cfg.name,
    }


builder = StateGraph(AnalysisState)
builder.add_node("junior_analyst", junior_analyst)
builder.add_node("bull_round_1", bull_round_1)
builder.add_node("bear_round_1", bear_round_1)
builder.add_node("bull_round_2", bull_round_2)
builder.add_node("bear_round_2", bear_round_2)
builder.add_node("judge_analyst", judge_analyst)
builder.set_entry_point("junior_analyst")
builder.add_conditional_edges(
    "junior_analyst",
    route_after_junior,
    ["bull_round_1", END],
)
builder.add_edge("bull_round_1", "bear_round_1")
builder.add_edge("bear_round_1", "bull_round_2")
builder.add_edge("bull_round_2", "bear_round_2")
builder.add_edge("bear_round_2", "judge_analyst")
builder.add_edge("judge_analyst", END)

analysis_graph = builder.compile()
