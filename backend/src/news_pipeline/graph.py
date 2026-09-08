from langgraph.graph import StateGraph, END

from src.config import app_config
from src.news_pipeline.agents import _call_llm
from src.news_pipeline.models import AnalysisState
from src.shared.observability import langfuse_client


async def junior_analyst(state: AnalysisState) -> AnalysisState:
    """Cheap model (Nano) evaluates the news item first."""
    cfg = app_config().agents
    with langfuse_client().start_as_current_observation(
        as_type="agent",
        name="run-junior-analyst",
        input={"role": "junior"},
        metadata={"deployment": cfg.nano_deployment},
    ) as observation:
        result = await _call_llm(
            cfg.nano_deployment,
            state["user_prompt"],
            analyst_role="junior",
        )
        observation.update(output=result)
        return {"llm_result": result, "predicted_by_model": cfg.nano_deployment}


def route_after_junior(state: AnalysisState) -> str:
    """Route based on junior's verdict: discard → end, low confidence → senior, else → end."""
    cfg = app_config().agents
    result = state.get("llm_result")

    if result is None or result.get("discard"):
        return END

    if result.get("confidence", 0) < cfg.confidence_threshold:
        return "senior_analyst"

    return END


async def senior_analyst(state: AnalysisState) -> AnalysisState:
    """Expensive model (Mini) re-evaluates when junior is uncertain."""
    cfg = app_config().agents
    with langfuse_client().start_as_current_observation(
        as_type="agent",
        name="run-senior-analyst",
        input={"role": "senior"},
        metadata={"deployment": cfg.mini_deployment},
    ) as observation:
        result = await _call_llm(
            cfg.mini_deployment,
            state["user_prompt"],
            analyst_role="senior",
        )
        observation.update(output=result)
        return {"llm_result": result, "predicted_by_model": cfg.mini_deployment}


# Build and compile the graph once at module level
_builder = StateGraph(AnalysisState)
_builder.add_node("junior_analyst", junior_analyst)
_builder.add_node("senior_analyst", senior_analyst)
_builder.set_entry_point("junior_analyst")
_builder.add_conditional_edges("junior_analyst", route_after_junior, ["senior_analyst", END])
_builder.add_edge("senior_analyst", END)

analysis_graph = _builder.compile()
