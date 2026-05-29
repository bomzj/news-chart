import pytest
from datetime import datetime, timezone, timedelta

from src.price_updater.updater import price_delta_pct, DELTA_MAP


class TestPriceDeltaPct:
    def test_positive_delta(self):
        """Price increase → positive fraction."""
        assert price_delta_pct(100.0, 105.0) == pytest.approx(0.05)

    def test_negative_delta(self):
        """Price decrease → negative fraction."""
        assert price_delta_pct(100.0, 95.0) == pytest.approx(-0.05)

    def test_no_change(self):
        """No price change → zero."""
        assert price_delta_pct(68250.0, 68250.0) == 0.0

    def test_large_drop(self):
        """50% drop returns -0.5."""
        assert price_delta_pct(100.0, 50.0) == pytest.approx(-0.50)

    def test_double(self):
        """Price doubles → 1.0 (100%)."""
        assert price_delta_pct(50000.0, 100000.0) == pytest.approx(1.0)

    def test_real_world_small_move(self):
        """Typical small move: $68250 → $68590 ≈ 0.498%."""
        result = price_delta_pct(68250.0, 68590.0)
        assert result == pytest.approx(0.004981, rel=1e-3)


class TestDeltaMap:
    def test_all_windows_defined(self):
        """All expected delta windows exist in map."""
        assert "1h" in DELTA_MAP
        assert "24h" in DELTA_MAP
        assert "7d" in DELTA_MAP
        assert "30d" in DELTA_MAP

    def test_durations_correct(self):
        """Timedelta values are correct."""
        assert DELTA_MAP["1h"][0] == timedelta(hours=1)
        assert DELTA_MAP["24h"][0] == timedelta(hours=24)
        assert DELTA_MAP["7d"][0] == timedelta(days=7)
        assert DELTA_MAP["30d"][0] == timedelta(days=30)

    def test_field_names_correct(self):
        """Payload field names match expected pattern."""
        assert DELTA_MAP["1h"][1] == "realized_price_delta_pct_1h"
        assert DELTA_MAP["24h"][1] == "realized_price_delta_pct_24h"
        assert DELTA_MAP["7d"][1] == "realized_price_delta_pct_7d"
        assert DELTA_MAP["30d"][1] == "realized_price_delta_pct_30d"


class TestTimeWindowEligibility:
    def test_1h_window_eligible(self):
        """News published > 1h ago is eligible for 1h delta."""
        published = datetime.now(timezone.utc) - timedelta(hours=2)
        delta_duration = DELTA_MAP["1h"][0]
        assert datetime.now(timezone.utc) >= published + delta_duration

    def test_1h_window_not_eligible(self):
        """News published < 1h ago is NOT eligible for 1h delta."""
        published = datetime.now(timezone.utc) - timedelta(minutes=30)
        delta_duration = DELTA_MAP["1h"][0]
        assert datetime.now(timezone.utc) < published + delta_duration

    def test_30d_window_eligible(self):
        """News published > 30d ago is eligible for 30d delta."""
        published = datetime.now(timezone.utc) - timedelta(days=31)
        delta_duration = DELTA_MAP["30d"][0]
        assert datetime.now(timezone.utc) >= published + delta_duration
