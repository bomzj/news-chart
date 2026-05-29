from typing import Literal

from pydantic import BaseModel

from src.shared.types import Sentiment, Impact, RawNews, AnalysisKept, AnalysisDiscarded, AnalysisResult


class AnalysisInput(BaseModel):
    raw_news: RawNews
    similar_context: list[dict]


class AnalysisOutput(BaseModel):
    news_summary: str
    sentiment: Sentiment
    impact: Impact
    confidence: float
    predicted_by_model: str
