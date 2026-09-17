from src.config import CollectorConfig, load_app_config


def test_collector_default_limit_is_5000():
    assert CollectorConfig().max_full_text_chars == 5000


def test_load_app_config_uses_collector_and_llm_settings(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
llm:
  name: custom-model
  reasoning_effort:
    default: max
    condense: high
    junior_analysis: high
collector:
  max_full_text_chars: 1234
""".strip()
    )

    config = load_app_config(config_path)

    assert config.llm.name == "custom-model"
    assert config.llm.reasoning_effort.default == "max"
    assert config.llm.reasoning_effort.condense == "high"
    assert config.llm.reasoning_effort.junior_analysis == "high"
    assert config.collector.max_full_text_chars == 1234


def test_load_app_config_accepts_extended_reasoning_efforts(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
llm:
  reasoning_effort:
    default: xhigh
    condense: none
    junior_analysis: max
""".strip()
    )

    config = load_app_config(config_path)

    assert config.llm.reasoning_effort.default == "xhigh"
    assert config.llm.reasoning_effort.condense == "none"
    assert config.llm.reasoning_effort.junior_analysis == "max"
