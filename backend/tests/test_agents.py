import pytest
from unittest.mock import AsyncMock, patch
from datetime import datetime, timezone

from src.shared.types import RawNews
from src.news_pipeline.models import AnalysisInput, AnalysisOutput
from src.news_pipeline.agents import analyze_single


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


@pytest.mark.asyncio
class TestAgentRouting:
    @patch("src.news_pipeline.graph._call_llm")
    async def test_high_confidence_stays_junior(self, mock_llm):
        """Junior result with confidence ≥ 0.75 is returned directly (no senior invocation)."""
        mock_llm.return_value = {
            "news_summary": "Fed cuts rates, bullish for crypto",
            "sentiment": "bullish",
            "impact": 3,
            "confidence": 0.85,
        }

        result = await analyze_single(_make_input())

        assert result is not None
        assert result.predicted_by_model == "gpt-5.4-nano"
        assert result.confidence == 0.85
        assert mock_llm.call_count == 1

    @patch("src.news_pipeline.graph._call_llm")
    async def test_low_confidence_escalates_to_senior(self, mock_llm):
        """Junior result with confidence < 0.75 escalates to senior via LangGraph routing."""
        mock_llm.side_effect = [
            # Junior: low confidence
            {
                "news_summary": "Unclear news impact",
                "sentiment": "bearish",
                "impact": 2,
                "confidence": 0.55,
            },
            # Senior: higher confidence
            {
                "news_summary": "After deeper analysis, bearish signal confirmed",
                "sentiment": "bearish",
                "impact": 2,
                "confidence": 0.88,
            },
        ]

        result = await analyze_single(_make_input())

        assert result is not None
        assert result.predicted_by_model == "gpt-5.4-mini"
        assert result.confidence == 0.88
        assert result.sentiment == "bearish"
        assert mock_llm.call_count == 2

    @patch("src.news_pipeline.graph._call_llm")
    async def test_output_schema_valid(self, mock_llm):
        """Output conforms to AnalysisOutput schema."""
        mock_llm.return_value = {
            "news_summary": "Summary text here",
            "sentiment": "bullish",
            "impact": 3,
            "confidence": 0.95,
        }

        result = await analyze_single(_make_input())

        assert isinstance(result, AnalysisOutput)
        assert result.sentiment in ("bullish", "bearish")
        assert 1 <= result.impact <= 3
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.news_summary) <= 400

    @patch("src.news_pipeline.graph._call_llm")
    async def test_summary_truncated_to_400_chars(self, mock_llm):
        """Summary longer than 400 chars gets truncated."""
        mock_llm.return_value = {
            "news_summary": "x" * 500,
            "sentiment": "bullish",
            "impact": 1,
            "confidence": 0.80,
        }

        result = await analyze_single(_make_input())
        assert result is not None
        assert len(result.news_summary) <= 400

    @patch("src.news_pipeline.graph._call_llm")
    async def test_discard_returns_none(self, mock_llm):
        """LLM returning discard=true results in None (noise filtered)."""
        mock_llm.return_value = {"discard": True}

        result = await analyze_single(_make_input())

        assert result is None
        assert mock_llm.call_count == 1

    @patch("src.news_pipeline.graph._call_llm")
    async def test_senior_discard_returns_none(self, mock_llm):
        """If junior has low confidence and senior discards, result is None."""
        mock_llm.side_effect = [
            {
                "news_summary": "Maybe noise",
                "sentiment": "bullish",
                "impact": 1,
                "confidence": 0.40,
            },
            {"discard": True},
        ]

        result = await analyze_single(_make_input())

        assert result is None
        assert mock_llm.call_count == 2


class TestGraphCompilation:
    def test_graph_compiles_and_has_expected_nodes(self):
        """The LangGraph analyst graph compiles with correct node structure."""
        from src.news_pipeline.graph import analysis_graph

        graph_nodes = analysis_graph.get_graph().nodes
        node_ids = set(graph_nodes.keys())
        assert "junior_analyst" in node_ids
        assert "senior_analyst" in node_ids
