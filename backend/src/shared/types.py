from datetime import datetime
from typing import Literal

from pydantic import BaseModel


Sentiment = Literal["bullish", "bearish"]

# 1: Notable (hours) — exchange listings, token burns, whale moves
# 2: High (days) — ETF decisions, major hacks, country-level regulation
# 3: Extreme (weeks) — Fed rates, stablecoin depegs, exchange collapses
Impact = Literal[1, 2, 3]


class RawNews(BaseModel):
    title: str
    description: str
    full_text: str
    source: str
    published_at: datetime
    url: str


class NewsRecord(BaseModel):
    ticker: str
    source: str
    published_at: datetime
    news_summary: str
    news_full_text: str
    sentiment: Sentiment
    impact: Impact
    confidence: float
    predicted_by_model: str
    price_at_ingestion: float
    realized_price_delta_pct_1h: float | None = None
    realized_price_delta_pct_24h: float | None = None
    realized_price_delta_pct_7d: float | None = None
    realized_price_delta_pct_30d: float | None = None


class AnalysisInput(BaseModel):
    raw_news: RawNews
    similar_context: list[dict]  # past news payloads with realized price info


class AnalysisKept(BaseModel):
    type: Literal["kept"] = "kept"
    news_summary: str
    sentiment: Sentiment
    impact: Impact
    confidence: float


class AnalysisDiscarded(BaseModel):
    type: Literal["discarded"] = "discarded"


AnalysisResult = AnalysisKept | AnalysisDiscarded
