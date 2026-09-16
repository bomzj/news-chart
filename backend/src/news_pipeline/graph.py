from langgraph.graph import StateGraph, END

from src.config import app_config
from src.news_pipeline.agents import _call_llm
from src.news_pipeline.models import AnalysisState


async def junior_analyst(state: AnalysisState) -> AnalysisState:
    """Cheap model (Nano) evaluates the news item first."""
    cfg = app_config().agents
    result = await _call_llm(
        cfg.nano_deployment,
        state["user_prompt"],
        stage="junior",
    )
    return {
        "llm_result": result,
        "junior_label": result.label,
        "predicted_by_model": cfg.nano_deployment,
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
    """Expensive model (Mini) re-evaluates when junior is uncertain."""
    cfg = app_config().agents
    result = await _call_llm(
        cfg.mini_deployment,
        state["user_prompt"],
        stage="senior",
    )
    return {"llm_result": result, "predicted_by_model": cfg.mini_deployment}


# Build and compile the graph once at module level
_builder = StateGraph(AnalysisState)
_builder.add_node("junior_analyst", junior_analyst)
_builder.add_node("senior_analyst", senior_analyst)
_builder.set_entry_point("junior_analyst")
_builder.add_conditional_edges("junior_analyst", route_after_junior, ["senior_analyst", END])
_builder.add_edge("senior_analyst", END)

analysis_graph = _builder.compile()
