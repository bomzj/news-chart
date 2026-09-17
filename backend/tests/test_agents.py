import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from datetime import datetime, timezone

from src.shared.types import RawNews
from src.news_collector.models import AnalysisInput, AnalysisOutput, analyst_result_adapter
from src.news_collector.agents import _call_llm, analyze_single


class _Observation:
    def __init__(self, owner, kwargs):
        self.owner = owner
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def update(self, **kwargs):
        self.owner.updates.append((self.kwargs, kwargs))


class _LangfuseStub:
    def __init__(self):
        self.observations = []
        self.updates = []

    def start_as_current_observation(self, **kwargs):
        self.observations.append(kwargs)
        return _Observation(self, kwargs)


def _make_input(title: str = "Fed cuts rates") -> AnalysisInput:
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


def _result(payload: dict):
    return analyst_result_adapter.validate_python(payload)


@pytest.mark.asyncio
class TestAgentRouting:
    @patch("src.news_collector.graph._call_llm")
    async def test_signal_stays_junior(self, mock_llm):
        """A definitive Junior label is returned directly."""
        mock_llm.return_value = _result({
            "label": "bullish",
            "news_summary": "Fed cuts rates, bullish for crypto",
            "impact": 3,
        })

        result = await analyze_single(_make_input())

        assert result is not None
        assert result.predicted_by_model == "gpt-5.4-nano"
        assert result.sentiment == "bullish"
        assert mock_llm.call_count == 1

    @patch("src.news_collector.graph._call_llm")
    async def test_uncertain_escalates_to_senior(self, mock_llm):
        """Junior uncertainty escalates to Senior via LangGraph routing."""
        mock_llm.side_effect = [
            _result({
                "label": "uncertain",
            }),
            _result({
                "label": "bearish",
                "news_summary": "After deeper analysis, bearish signal confirmed",
                "impact": 2,
            }),
        ]

        result = await analyze_single(_make_input())

        assert result is not None
        assert result.predicted_by_model == "gpt-5.4-mini"
        assert result.sentiment == "bearish"
        assert mock_llm.call_count == 2

    @patch("src.news_collector.graph._call_llm")
    async def test_output_schema_valid(self, mock_llm):
        """Output conforms to AnalysisOutput schema."""
        mock_llm.return_value = _result({
            "label": "bullish",
            "news_summary": "Summary text here",
            "impact": 3,
        })

        result = await analyze_single(_make_input())

        assert isinstance(result, AnalysisOutput)
        assert result.sentiment in ("bullish", "bearish")
        assert 1 <= result.impact <= 3
        assert len(result.news_summary) <= 400

    @patch("src.news_collector.graph._call_llm")
    async def test_summary_truncated_to_400_chars(self, mock_llm):
        """Summary longer than 400 chars gets truncated."""
        mock_llm.return_value = _result({
            "label": "bullish",
            "news_summary": "x" * 500,
            "impact": 1,
        })

        result = await analyze_single(_make_input())
        assert result is not None
        assert len(result.news_summary) <= 400

    @patch("src.news_collector.graph._call_llm")
    async def test_noise_returns_none(self, mock_llm):
        """LLM returning noise results in None."""
        mock_llm.return_value = _result({"label": "noise"})

        result = await analyze_single(_make_input())

        assert result is None
        assert mock_llm.call_count == 1

    @patch("src.news_collector.graph._call_llm")
    async def test_senior_noise_returns_none(self, mock_llm):
        """If Junior is uncertain and Senior returns noise, the result is discarded."""
        mock_llm.side_effect = [
            _result({
                "label": "uncertain",
            }),
            _result({"label": "noise"}),
        ]

        result = await analyze_single(_make_input())

        assert result is None
        assert mock_llm.call_count == 2

    @patch("src.news_collector.graph._call_llm")
    async def test_senior_uncertain_returns_none(self, mock_llm):
        """If Senior remains uncertain, the result is discarded."""
        mock_llm.side_effect = [
            _result({"label": "uncertain"}),
            _result({"label": "uncertain"}),
        ]

        result = await analyze_single(_make_input())

        assert result is None
        assert mock_llm.call_count == 2

    @patch("src.news_collector.graph._call_llm")
    async def test_analysis_trace_records_disposition(self, mock_llm, monkeypatch):
        """The parent Langfuse observation records labels and final disposition."""
        langfuse = _LangfuseStub()
        monkeypatch.setattr("src.news_collector.agents.langfuse_client", lambda: langfuse)
        mock_llm.return_value = _result({
            "label": "bearish",
            "news_summary": "Exchange withdrawals are frozen",
            "impact": 2,
        })

        result = await analyze_single(_make_input())

        assert result is not None
        assert len(langfuse.observations) == 1
        assert langfuse.observations[0]["name"] == "analyze-news"
        assert langfuse.observations[0]["as_type"] == "chain"
        assert langfuse.updates[0][1]["output"] == {
            "label": "bearish",
            "disposition": "stored",
            "predicted_by_model": "gpt-5.4-nano",
        }
        assert langfuse.updates[0][1]["metadata"] == {
            "junior_label": "bearish",
            "escalated": False,
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
        result = await _call_llm("gpt-5.4-nano", "article text", stage="junior")

    assert result.label == "bullish"
    factory.assert_called_once_with(observe=True)
    assert create.await_args.kwargs["model"] == "gpt-5.4-nano"
    assert create.await_args.kwargs["reasoning"] == {"effort": "high"}
    assert create.await_args.kwargs["name"] == "classify-news"
    assert create.await_args.kwargs["metadata"] == {"analyst_stage": "junior"}


@pytest.mark.asyncio
async def test_call_llm_uses_smart_reasoning_effort(monkeypatch):
    create = AsyncMock(
        return_value=SimpleNamespace(output_text='{"label":"noise"}')
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

    result = await _call_llm("gpt-5.4-mini", "article text", stage="senior")

    assert result.label == "noise"
    assert create.await_args.kwargs["model"] == "gpt-5.4-mini"
    assert create.await_args.kwargs["reasoning"] == {"effort": "max"}


@pytest.mark.asyncio
async def test_call_llm_omits_langfuse_options_when_tracing_is_disabled(monkeypatch):
    create = AsyncMock(
        return_value=SimpleNamespace(
            output_text='{"label":"noise"}'
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
            lambda: False,
        )
        result = await _call_llm("gpt-5.4-nano", "article text", stage="junior")

    assert result.label == "noise"
    factory.assert_called_once_with(observe=True)
    assert "name" not in create.await_args.kwargs
    assert "metadata" not in create.await_args.kwargs


class TestGraphCompilation:
    def test_graph_compiles_and_has_expected_nodes(self):
        """The LangGraph analyst graph compiles with correct node structure."""
        from src.news_collector.graph import analysis_graph

        graph_nodes = analysis_graph.get_graph().nodes
        node_ids = set(graph_nodes.keys())
        assert "junior_analyst" in node_ids
        assert "senior_analyst" in node_ids
