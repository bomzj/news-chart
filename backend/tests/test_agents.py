from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.config import app_config
from src.news_collector.agents import analyze_single, call_llm
from src.news_collector.models import (
    AnalystResult,
    AnalysisInput,
    AnalysisOutput,
    DebateResponse,
    analyst_result_adapter,
)
from src.shared.types import RawNews


class ObservationStub:
    def __init__(self, owner, kwargs):
        self.owner = owner
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def update(self, **kwargs):
        self.owner.updates.append((self.kwargs, kwargs))


class LangfuseStub:
    def __init__(self):
        self.observations = []
        self.updates = []

    def start_as_current_observation(self, **kwargs):
        self.observations.append(kwargs)
        return ObservationStub(self, kwargs)


def make_input(title: str = "Fed cuts rates") -> AnalysisInput:
    return AnalysisInput(
        raw_news=RawNews(
            title=title,
            description="Federal Reserve announces rate cut",
            full_text="The Federal Reserve announced a 25bp rate cut.",
            source="reuters",
            published_at=datetime.now(timezone.utc),
            url="https://example.com",
        ),
        similar_context=[],
    )


def analyst(payload: dict) -> AnalystResult:
    return analyst_result_adapter.validate_python(payload)


def debate(argument: str) -> DebateResponse:
    return DebateResponse(argument=argument)


def assert_reasoning_efforts(mock_llm, expected_stages: list[str], expected_efforts: list[str]):
    assert [call.kwargs["stage"] for call in mock_llm.call_args_list] == expected_stages
    assert [
        call.kwargs["reasoning_effort"] for call in mock_llm.call_args_list
    ] == expected_efforts


@pytest.mark.asyncio
class TestAgentRouting:
    @patch("src.news_collector.graph.call_llm")
    async def test_normal_impact_stays_junior(self, mock_llm):
        mock_llm.return_value = analyst(
            {
                "label": "bullish",
                "news_summary": "Fed cuts rates, bullish for crypto",
                "impact": 1,
            }
        )

        result = await analyze_single(make_input())

        assert result is not None
        assert result.sentiment == "bullish"
        assert result.impact == 1
        assert result.predicted_by_model == app_config().llm.name
        assert mock_llm.call_count == 1
        assert_reasoning_efforts(
            mock_llm,
            ["junior"],
            [app_config().llm.reasoning_effort.junior_analysis],
        )

    @patch("src.news_collector.graph.call_llm")
    async def test_uncertain_runs_alternating_debate_then_judge(self, mock_llm):
        mock_llm.side_effect = [
            analyst({"label": "uncertain"}),
            debate("Bull round 1"),
            debate("Bear rebuts Bull"),
            debate("Bull responds"),
            debate("Bear closes"),
            analyst(
                {
                    "label": "bearish",
                    "news_summary": "The final Judge found a bearish signal",
                    "impact": 2,
                }
            ),
        ]

        result = await analyze_single(make_input())

        assert result is not None
        assert result.sentiment == "bearish"
        assert result.news_summary == "The final Judge found a bearish signal"
        assert result.impact == 2
        assert mock_llm.call_count == 6
        assert_reasoning_efforts(
            mock_llm,
            ["junior", "bull", "bear", "bull", "bear", "judge"],
            [
                app_config().llm.reasoning_effort.junior_analysis,
                app_config().llm.reasoning_effort.bull_bear,
                app_config().llm.reasoning_effort.bull_bear,
                app_config().llm.reasoning_effort.bull_bear,
                app_config().llm.reasoning_effort.bull_bear,
                app_config().llm.reasoning_effort.judge,
            ],
        )

        debate_prompts = [
            call.args[1] for call in mock_llm.call_args_list[1:5]
        ]
        assert "No previous debate turns." in debate_prompts[0]
        assert "Bull round 1" in debate_prompts[1]
        assert "Bear rebuts Bull" in debate_prompts[2]
        assert "Bull responds" in debate_prompts[3]
        judge_prompt = mock_llm.call_args_list[5].args[1]
        assert "Junior analysis" in judge_prompt
        assert all(argument in judge_prompt for argument in [
            "Bull round 1",
            "Bear rebuts Bull",
            "Bull responds",
            "Bear closes",
        ])

    @pytest.mark.parametrize("impact", [2, 3])
    @patch("src.news_collector.graph.call_llm")
    async def test_high_or_extreme_impact_enters_debate(self, mock_llm, impact):
        mock_llm.side_effect = [
            analyst(
                {
                    "label": "bullish",
                    "news_summary": "Junior directional result",
                    "impact": impact,
                }
            ),
            debate("Bull round 1"),
            debate("Bear round 1"),
            debate("Bull round 2"),
            debate("Bear round 2"),
            analyst(
                {
                    "label": "bullish",
                    "news_summary": "Judge directional result",
                    "impact": 3,
                }
            ),
        ]

        result = await analyze_single(make_input())

        assert result is not None
        assert result.news_summary == "Judge directional result"
        assert result.impact == 3
        assert mock_llm.call_count == 6

    @patch("src.news_collector.graph.call_llm")
    async def test_noise_returns_none_without_debate(self, mock_llm):
        mock_llm.return_value = analyst({"label": "noise"})

        result = await analyze_single(make_input())

        assert result is None
        assert mock_llm.call_count == 1

    @pytest.mark.parametrize("label", ["noise", "uncertain"])
    @patch("src.news_collector.graph.call_llm")
    async def test_judge_noise_or_uncertain_returns_none(self, mock_llm, label):
        mock_llm.side_effect = [
            analyst({"label": "uncertain"}),
            debate("Bull round 1"),
            debate("Bear round 1"),
            debate("Bull round 2"),
            debate("Bear round 2"),
            analyst({"label": label}),
        ]

        result = await analyze_single(make_input())

        assert result is None
        assert mock_llm.call_count == 6

    @patch("src.news_collector.graph.call_llm")
    async def test_summary_truncated_to_400_chars(self, mock_llm):
        mock_llm.return_value = analyst(
            {
                "label": "bullish",
                "news_summary": "x" * 500,
                "impact": 1,
            }
        )

        result = await analyze_single(make_input())

        assert result is not None
        assert len(result.news_summary) <= 400

    @patch("src.news_collector.graph.call_llm")
    async def test_analysis_trace_records_direct_disposition(self, mock_llm, monkeypatch):
        langfuse = LangfuseStub()
        monkeypatch.setattr(
            "src.news_collector.agents.langfuse_client",
            lambda: langfuse,
        )
        mock_llm.return_value = analyst(
            {
                "label": "bearish",
                "news_summary": "Exchange withdrawals are frozen",
                "impact": 1,
            }
        )

        result = await analyze_single(make_input())

        assert result is not None
        assert len(langfuse.observations) == 1
        assert langfuse.observations[0]["name"] == "analyze-news"
        assert langfuse.observations[0]["as_type"] == "chain"
        assert langfuse.updates[0][1]["output"] == {
            "label": "bearish",
            "disposition": "stored",
            "predicted_by_model": app_config().llm.name,
        }
        assert langfuse.updates[0][1]["metadata"] == {
            "junior_label": "bearish",
            "escalated": False,
            "debate_transcript": [],
            "judge_label": None,
            "final_label": "bearish",
        }


@pytest.mark.asyncio
async def test_call_llm_uses_observed_client_and_validates_label(monkeypatch):
    create = AsyncMock(
        return_value=SimpleNamespace(
            output_text='{"label":"bullish","news_summary":"Positive update","impact":1}'
        )
    )
    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    client_factory = patch(
        "src.news_collector.agents.azure_ai_client",
        return_value=client,
    )

    with client_factory as factory:
        monkeypatch.setattr(
            "src.news_collector.agents.langfuse_tracing_enabled",
            lambda: True,
        )
        result = await call_llm(
            app_config().llm.name,
            "article text",
            reasoning_effort="high",
            stage="junior",
        )

    assert result.label == "bullish"
    factory.assert_called_once_with(observe=True)
    assert create.await_args.kwargs["model"] == app_config().llm.name
    assert create.await_args.kwargs["reasoning"] == {"effort": "high"}
    assert create.await_args.kwargs["name"] == "classify-news"
    assert create.await_args.kwargs["metadata"] == {"analyst_stage": "junior"}


@pytest.mark.asyncio
async def test_call_llm_validates_debate_response(monkeypatch):
    create = AsyncMock(
        return_value=SimpleNamespace(output_text='{"argument":"Evidence supports the bull case"}')
    )
    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    monkeypatch.setattr(
        "src.news_collector.agents.azure_ai_client",
        lambda *, observe=False: client,
    )
    monkeypatch.setattr(
        "src.news_collector.agents.langfuse_tracing_enabled",
        lambda: False,
    )

    result = await call_llm(
        app_config().llm.name,
        "article text",
        reasoning_effort="high",
        stage="bull",
    )

    assert result == DebateResponse(argument="Evidence supports the bull case")
    assert create.await_args.kwargs["reasoning"] == {"effort": "high"}


@pytest.mark.asyncio
async def test_call_llm_passes_explicit_max_reasoning_effort(monkeypatch):
    create = AsyncMock(return_value=SimpleNamespace(output_text='{"label":"noise"}'))
    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    monkeypatch.setattr(
        "src.news_collector.agents.azure_ai_client",
        lambda *, observe=False: client,
    )
    monkeypatch.setattr(
        "src.news_collector.agents.langfuse_tracing_enabled",
        lambda: False,
    )

    result = await call_llm(
        app_config().llm.name,
        "article text",
        reasoning_effort="max",
        stage="judge",
    )

    assert result.label == "noise"
    assert create.await_args.kwargs["model"] == app_config().llm.name
    assert create.await_args.kwargs["reasoning"] == {"effort": "max"}


@pytest.mark.asyncio
async def test_call_llm_omits_langfuse_options_when_tracing_is_disabled(monkeypatch):
    create = AsyncMock(return_value=SimpleNamespace(output_text='{"label":"noise"}'))
    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    client_factory = patch(
        "src.news_collector.agents.azure_ai_client",
        return_value=client,
    )

    with client_factory as factory:
        monkeypatch.setattr(
            "src.news_collector.agents.langfuse_tracing_enabled",
            lambda: False,
        )
        result = await call_llm(
            app_config().llm.name,
            "article text",
            reasoning_effort="high",
            stage="junior",
        )

    assert result.label == "noise"
    factory.assert_called_once_with(observe=True)
    assert "name" not in create.await_args.kwargs
    assert "metadata" not in create.await_args.kwargs


class TestGraphCompilation:
    def test_graph_compiles_and_has_expected_nodes(self):
        from src.news_collector.graph import analysis_graph

        graph_nodes = analysis_graph.get_graph().nodes
        node_ids = set(graph_nodes.keys())
        assert {
            "junior_analyst",
            "bull_round_1",
            "bear_round_1",
            "bull_round_2",
            "bear_round_2",
            "judge_analyst",
        } <= node_ids
