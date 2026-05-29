from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class DedupConfig(BaseModel):
    cosine_threshold: float = 0.90
    lookback_hours: int = 24


class AgentsConfig(BaseModel):
    confidence_threshold: float = 0.75
    nano_deployment: str = "gpt-5.4-nano"
    mini_deployment: str = "gpt-5.4-mini"
    api_version: str = "2025-04-01-preview"
    reasoning_effort: str = "high"


class EmbeddingsConfig(BaseModel):
    deployment: str = "text-embedding-3-large"
    dimensions: int = 256
    api_version: str = "2023-05-15"


class PipelineConfig(BaseModel):
    fetch_news_interval_minutes: int = 10
    max_full_text_chars: int = 2000


DeltaWindow = Literal["1h", "24h", "7d", "30d"]


class PriceUpdaterConfig(BaseModel):
    deltas: list[DeltaWindow] = ["1h", "24h", "7d", "30d"]


class AppConfig(BaseModel):
    tickers: list[str] = ["BTC"]
    dedup: DedupConfig = DedupConfig()
    agents: AgentsConfig = AgentsConfig()
    embeddings: EmbeddingsConfig = EmbeddingsConfig()
    pipeline: PipelineConfig = PipelineConfig()
    price_updater: PriceUpdaterConfig = PriceUpdaterConfig()


class Secrets(BaseSettings):
    marketaux_api_key: str = ""
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    azure_ai_endpoint: str = ""
    azure_ai_api_key: str = ""

    model_config = {"env_file": ".env", "extra": "ignore"}


def load_app_config(path: Path = Path("config.yaml")) -> AppConfig:
    if path.exists():
        raw = yaml.safe_load(path.read_text())
        return AppConfig.model_validate(raw or {})
    return AppConfig()


def load_secrets() -> Secrets:
    return Secrets()


_app_config: AppConfig | None = None
_secrets: Secrets | None = None


def app_config() -> AppConfig:
    global _app_config
    if _app_config is None:
        _app_config = load_app_config()
    return _app_config


def secrets() -> Secrets:
    global _secrets
    if _secrets is None:
        _secrets = load_secrets()
    return _secrets
