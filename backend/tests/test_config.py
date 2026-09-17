from src.config import load_app_config


def test_load_app_config_uses_collector_and_model_aliases(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
agents:
  lite_model: custom-lite
  smart_model: custom-smart
collector:
  max_full_text_chars: 1234
""".strip()
    )

    config = load_app_config(config_path)

    assert config.agents.lite_model == "custom-lite"
    assert config.agents.smart_model == "custom-smart"
    assert config.collector.max_full_text_chars == 1234
