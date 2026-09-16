from typing import Annotated, Literal, TypeAlias, TypedDict

from pydantic import BaseModel, Field, TypeAdapter

from src.shared.types import AnalysisLabel, Impact, RawNews, Sentiment


class AnalysisInput(BaseModel):
    raw_news: RawNews
    similar_context: list[dict]


class AnalystNoise(BaseModel):
    label: Literal["noise"]


class AnalystUncertain(BaseModel):
    label: Literal["uncertain"]


class AnalystBullish(BaseModel):
    label: Literal["bullish"]
    news_summary: str
    impact: Impact


class AnalystBearish(BaseModel):
    label: Literal["bearish"]
    news_summary: str
    impact: Impact


AnalystResult: TypeAlias = Annotated[
    AnalystNoise | AnalystUncertain | AnalystBullish | AnalystBearish,
    Field(discriminator="label"),
]
analyst_result_adapter = TypeAdapter(AnalystResult)


class AnalysisOutput(BaseModel):
    news_summary: str
    sentiment: Sentiment
    impact: Impact
    predicted_by_model: str


class AnalysisState(TypedDict, total=False):
    """LangGraph state flowing through the analyst graph."""
    user_prompt: str
    llm_result: AnalystResult | None
    junior_label: AnalysisLabel | None
    predicted_by_model: str
