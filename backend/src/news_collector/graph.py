from langgraph.graph import StateGraph, END

from src.config import app_config
from src.news_collector.agents import call_llm
from src.news_collector.models import AnalysisState


async def junior_analyst(state: AnalysisState) -> AnalysisState:
    """Evaluate the news item first as the Junior analyst."""
    cfg = app_config().llm
    result = await call_llm(
        cfg.name,
        state["user_prompt"],
        reasoning_effort=cfg.reasoning_effort.junior_analysis,
        stage="junior",
    )
    return {
        "llm_result": result,
        "junior_label": result.label,
        "predicted_by_model": cfg.name,
    }


def route_after_junior(state: AnalysisState) -> str:
    """Route uncertain Junior results to Senior; all other labels finish."""
    result = state.get("llm_result")

    if result is None:
        return END

    match result.label:
        case "uncertain":
            return "senior_analyst"
        case "noise" | "bullish" | "bearish":
            return END

    raise ValueError(f"Unsupported analyst label: {result.label}")


async def senior_analyst(state: AnalysisState) -> AnalysisState:
    """Re-evaluate uncertain Junior results as the Senior analyst."""
    cfg = app_config().llm
    result = await call_llm(
        cfg.name,
        state["user_prompt"],
        reasoning_effort=cfg.reasoning_effort.default,
        stage="senior",
    )
    return {"llm_result": result, "predicted_by_model": cfg.name}


# Build and compile the graph once at module level
_builder = StateGraph(AnalysisState)
_builder.add_node("junior_analyst", junior_analyst)
_builder.add_node("senior_analyst", senior_analyst)
_builder.set_entry_point("junior_analyst")
_builder.add_conditional_edges("junior_analyst", route_after_junior, ["senior_analyst", END])
_builder.add_edge("senior_analyst", END)

analysis_graph = _builder.compile()
