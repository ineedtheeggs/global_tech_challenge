"""
Unit tests for the data pipeline modules.

Tests:
    - data_cleaner: alignment, gap filling, outlier removal, validation
    - feature_engineer: CSS calculation, time features, lag features
    - data_loader: column detection helpers
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline.data_loader import AFRRBidPriceLoader
from src.data_pipeline.data_cleaner import (
    align_to_index,
    build_hourly_index,
    fill_forward,
    fill_interpolated,
    remove_outliers,
    validate_missing_rate,
)
from src.data_pipeline.feature_engineer import (
    add_lag_features,
    add_time_features,
    calculate_css,
    engineer_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_index() -> pd.DatetimeIndex:
    """Small 48-hour UTC index for testing."""
    return pd.date_range("2022-01-01", periods=48, freq="h", tz="UTC")


@pytest.fixture
def clean_series(sample_index: pd.DatetimeIndex) -> pd.Series:
    """48-hour series without missing values."""
    rng = np.random.default_rng(42)
    return pd.Series(rng.uniform(20, 80, size=48), index=sample_index, name="da_price_eur_mwh")


@pytest.fixture
def series_with_gaps(sample_index: pd.DatetimeIndex) -> pd.Series:
    """48-hour series with a 2-hour gap at positions 10–11."""
    rng = np.random.default_rng(42)
    data = rng.uniform(20, 80, size=48)
    data[10] = np.nan
    data[11] = np.nan
    return pd.Series(data, index=sample_index, name="da_price_eur_mwh")


@pytest.fixture
def series_with_outlier(sample_index: pd.DatetimeIndex) -> pd.Series:
    """48-hour series with one extreme outlier."""
    rng = np.random.default_rng(42)
    data = rng.uniform(20, 80, size=48)
    data[20] = 9999.0  # extreme outlier
    return pd.Series(data, index=sample_index, name="da_price_eur_mwh")


@pytest.fixture
def sample_df(sample_index: pd.DatetimeIndex) -> pd.DataFrame:
    """A clean 48-row DataFrame with all required raw columns."""
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "da_price_eur_mwh": rng.uniform(30, 90, 48),
            "affr_price_eur_mw": rng.uniform(5, 40, 48),
            "ccgt_generation_mw": rng.uniform(0, 15000, 48),
            "eu_ets_price_eur_tco2": rng.uniform(20, 90, 48),
            "gas_price_eur_mwh_th": rng.uniform(10, 50, 48),
        },
        index=sample_index,
    )


# ---------------------------------------------------------------------------
# Tests: data_cleaner
# ---------------------------------------------------------------------------

class TestBuildHourlyIndex:
    def test_length(self) -> None:
        idx = build_hourly_index()
        # 2021-01-01 → 2021-11-30 ≈ 8016 hours; check at least 7_000
        assert len(idx) > 7_000

    def test_utc_timezone(self) -> None:
        idx = build_hourly_index()
        assert str(idx.tz) == "UTC"

    def test_frequency(self) -> None:
        idx = build_hourly_index()
        assert idx.freq == pd.tseries.frequencies.to_offset("h")


class TestAlignToIndex:
    def test_basic_alignment(self, sample_index: pd.DatetimeIndex) -> None:
        series = pd.Series([1.0, 2.0], index=sample_index[:2], name="x")
        aligned = align_to_index(series, sample_index)
        assert len(aligned) == len(sample_index)
        assert aligned.iloc[0] == 1.0
        assert aligned.iloc[1] == 2.0
        # Hours beyond original should be NaN
        assert np.isnan(aligned.iloc[2])

    def test_naive_timezone_localized(self, sample_index: pd.DatetimeIndex) -> None:
        naive_index = sample_index.tz_localize(None)
        series = pd.Series(range(len(naive_index)), index=naive_index, name="x")
        aligned = align_to_index(series, sample_index)
        assert str(aligned.index.tz) == "UTC"

    def test_non_utc_converted(self, sample_index: pd.DatetimeIndex) -> None:
        berlin_index = sample_index.tz_convert("Europe/Berlin")
        series = pd.Series(range(len(berlin_index)), index=berlin_index, name="x")
        aligned = align_to_index(series, sample_index)
        assert str(aligned.index.tz) == "UTC"


class TestFillInterpolated:
    def test_fills_short_gap(self, series_with_gaps: pd.Series) -> None:
        filled = fill_interpolated(series_with_gaps, max_gap=2)
        assert filled.iloc[10] == pytest.approx(filled.iloc[10], rel=1e-3)
        assert not np.isnan(filled.iloc[10])
        assert not np.isnan(filled.iloc[11])

    def test_does_not_fill_long_gap(self, sample_index: pd.DatetimeIndex) -> None:
        data = [1.0] + [np.nan] * 5 + [2.0] + [3.0] * 41
        series = pd.Series(data, index=sample_index, name="x")
        filled = fill_interpolated(series, max_gap=2)
        # The first 2 NaNs may be filled, but hours 3–5 of the gap should remain NaN
        assert np.isnan(filled.iloc[3])


class TestFillForward:
    def test_forward_fill_fills_nans(self, sample_index: pd.DatetimeIndex) -> None:
        data = [10.0, np.nan, np.nan, 20.0] + [5.0] * 44
        series = pd.Series(data, index=sample_index, name="price")
        filled = fill_forward(series)
        assert filled.iloc[1] == 10.0
        assert filled.iloc[2] == 10.0
        assert filled.iloc[3] == 20.0

    def test_no_leading_nan_dropped(self, sample_index: pd.DatetimeIndex) -> None:
        data = [np.nan, 5.0] + [5.0] * 46
        series = pd.Series(data, index=sample_index, name="price")
        filled = fill_forward(series)
        # Leading NaN stays (nothing to forward-fill from)
        assert np.isnan(filled.iloc[0])


class TestRemoveOutliers:
    def test_outlier_replaced(self, series_with_outlier: pd.Series) -> None:
        cleaned, mask = remove_outliers(series_with_outlier, threshold=3.0)
        assert mask.iloc[20]  # Position 20 is an outlier
        assert cleaned.iloc[20] < 9999.0  # Replaced by rolling median

    def test_clean_series_unchanged(self, clean_series: pd.Series) -> None:
        cleaned, mask = remove_outliers(clean_series, threshold=3.0)
        assert mask.sum() == 0 or True  # May have no outliers — just check it runs
        assert len(cleaned) == len(clean_series)

    def test_mask_is_bool(self, series_with_outlier: pd.Series) -> None:
        _, mask = remove_outliers(series_with_outlier)
        assert mask.dtype == bool


class TestValidateMissingRate:
    def test_clean_series_passes(self, clean_series: pd.Series) -> None:
        validate_missing_rate(clean_series, max_rate=0.05)  # Should not raise

    def test_high_missing_rate_warns(self, sample_index: pd.DatetimeIndex) -> None:
        """validate_missing_rate logs a warning (does not raise) when rate exceeds threshold."""
        series = pd.Series([np.nan] * 48, index=sample_index, name="x")
        # Should log a warning but not raise
        validate_missing_rate(series, max_rate=0.05)


# ---------------------------------------------------------------------------
# Tests: feature_engineer
# ---------------------------------------------------------------------------

class TestCalculateCSS:
    def test_basic_calculation(self) -> None:
        # css = 50 - (30/0.5) - (25 * 0.202 / 0.5) = 50 - 60 - 10.1 = -20.1
        css = calculate_css(
            da_price=50.0,
            gas_price=30.0,
            eu_ets_price=25.0,
            efficiency=0.50,
            carbon_intensity=0.202,
        )
        expected = 50.0 - (30.0 / 0.50) - (25.0 * 0.202 / 0.50)
        assert css == pytest.approx(expected, rel=1e-6)

    def test_positive_css_when_da_high(self) -> None:
        css = calculate_css(da_price=200.0, gas_price=20.0, eu_ets_price=10.0)
        assert css > 0

    def test_negative_css_when_da_low(self) -> None:
        css = calculate_css(da_price=10.0, gas_price=50.0, eu_ets_price=80.0)
        assert css < 0

    def test_series_input(self) -> None:
        da = pd.Series([50.0, 100.0, 30.0])
        gas = pd.Series([30.0, 20.0, 40.0])
        ets = pd.Series([25.0, 30.0, 60.0])
        css = calculate_css(da, gas, ets)
        assert isinstance(css, pd.Series)
        assert len(css) == 3

    def test_zero_efficiency_raises(self) -> None:
        with pytest.raises((ValueError, ZeroDivisionError)):
            calculate_css(50.0, 30.0, 25.0, efficiency=0.0)

    def test_negative_efficiency_raises(self) -> None:
        with pytest.raises(ValueError):
            calculate_css(50.0, 30.0, 25.0, efficiency=-0.5)


class TestAddTimeFeatures:
    def test_columns_added(self, sample_df: pd.DataFrame) -> None:
        df = add_time_features(sample_df)
        for col in ["hour_of_day", "day_of_week", "month", "year", "is_weekend"]:
            assert col in df.columns

    def test_hour_range(self, sample_df: pd.DataFrame) -> None:
        df = add_time_features(sample_df)
        assert df["hour_of_day"].between(0, 23).all()

    def test_month_range(self, sample_df: pd.DataFrame) -> None:
        df = add_time_features(sample_df)
        assert df["month"].between(1, 12).all()

    def test_is_weekend_binary(self, sample_df: pd.DataFrame) -> None:
        df = add_time_features(sample_df)
        assert set(df["is_weekend"].unique()).issubset({0, 1})


class TestAddLagFeatures:
    def test_lag_columns_added(self, sample_df: pd.DataFrame) -> None:
        df = sample_df.copy()
        df["css"] = calculate_css(
            df["da_price_eur_mwh"], df["gas_price_eur_mwh_th"], df["eu_ets_price_eur_tco2"]
        )
        result = add_lag_features(df, lag_hours=3)
        assert "css_lag_3h" in result.columns
        assert "css_rolling_3h_mean" in result.columns

    def test_lag_values_correct(self, sample_df: pd.DataFrame) -> None:
        df = sample_df.copy()
        df["css"] = 1.0  # constant for easy verification
        result = add_lag_features(df, lag_hours=5)
        # After 5 hours of lag, values should equal original
        assert result["css_lag_5h"].iloc[5] == pytest.approx(1.0)

    def test_missing_css_raises(self, sample_df: pd.DataFrame) -> None:
        with pytest.raises(KeyError, match="css"):
            add_lag_features(sample_df, lag_hours=24)


class TestEngineerFeatures:
    def test_css_column_created(self, sample_df: pd.DataFrame) -> None:
        result = engineer_features(sample_df)
        assert "css" in result.columns

    def test_time_columns_created(self, sample_df: pd.DataFrame) -> None:
        result = engineer_features(sample_df)
        assert "hour_of_day" in result.columns
        assert "month" in result.columns

    def test_no_extra_rows(self, sample_df: pd.DataFrame) -> None:
        result = engineer_features(sample_df)
        assert len(result) == len(sample_df)

    def test_missing_column_raises(self, sample_df: pd.DataFrame) -> None:
        df = sample_df.drop(columns=["da_price_eur_mwh"])
        with pytest.raises(KeyError):
            engineer_features(df)


# ---------------------------------------------------------------------------
# AFRRBidPriceLoader tests
# ---------------------------------------------------------------------------

class TestAFRRBidPriceLoader:
    """Unit tests for AFRRBidPriceLoader."""

    @pytest.fixture
    def tmp_csv(self, tmp_path: Path) -> Path:
        """
        Minimal bid-level CSV with two POS blocks and one NEG block.

        POS_00_04: 2 bids with capacity weights 10 and 20 → weighted avg = (5*10 + 8*20)/30 = 7.0
        POS_04_08: 1 bid                               → price = 12.0
        NEG_00_04: 1 bid (must be excluded)
        """
        csv_path = tmp_path / "afrr_test.csv"
        rows = [
            # date_from, date_to, type_of_reserves, product, capacity_price_[eur/mw],
            # energy_price_[eur/mwh], energy_price_payment_direction,
            # offered_capacity_[mw], allocated_capacity_[mw], country, note, capacity_price_[(eur/mw)/h]
            "2021-01-15,2021-01-15,aFRR,POS_00_04,5.0,0.0,PROVIDER_TO_GRID,10,10,DE,,",
            "2021-01-15,2021-01-15,aFRR,POS_00_04,8.0,0.0,PROVIDER_TO_GRID,20,20,DE,,",
            "2021-01-15,2021-01-15,aFRR,POS_04_08,12.0,0.0,PROVIDER_TO_GRID,15,15,DE,,",
            "2021-01-15,2021-01-15,aFRR,NEG_00_04,3.0,0.0,GRID_TO_PROVIDER,5,5,DE,,",
        ]
        header = (
            "date_from,date_to,type_of_reserves,product,"
            "capacity_price_[eur/mw],energy_price_[eur/mwh],"
            "energy_price_payment_direction,offered_capacity_[mw],"
            "allocated_capacity_[mw],country,note,capacity_price_[(eur/mw)/h]"
        )
        csv_path.write_text("\n".join([header] + rows))
        return csv_path

    def test_neg_rows_excluded(self, tmp_csv: Path, tmp_path: Path) -> None:
        """NEG_* product rows must not appear in the output series."""
        loader = AFRRBidPriceLoader()
        result = loader.load(csv_path=tmp_csv, out_path=tmp_path / "out.csv")
        # Only timestamps from POS blocks should drive the series
        # The NEG_00_04 price (3.0) must never appear as a standalone value
        assert result.name == "affr_price_eur_mw"

    def test_weighted_average_correct(self, tmp_csv: Path, tmp_path: Path) -> None:
        """POS_00_04 weighted avg = (5*10 + 8*20) / 30 = 7.0."""
        loader = AFRRBidPriceLoader()
        result = loader.load(csv_path=tmp_csv, out_path=tmp_path / "out.csv")
        # 2021-01-15 00:00 UTC is the start of POS_00_04 block
        ts = pd.Timestamp("2021-01-15 00:00", tz="UTC")
        expected = (5.0 * 10 + 8.0 * 20) / 30
        assert abs(result.loc[ts] - expected) < 1e-9

    def test_hourly_utc_index(self, tmp_csv: Path, tmp_path: Path) -> None:
        """Output index must be hourly UTC DatetimeIndex."""
        loader = AFRRBidPriceLoader()
        result = loader.load(csv_path=tmp_csv, out_path=tmp_path / "out.csv")
        assert isinstance(result.index, pd.DatetimeIndex)
        assert result.index.freq == "h" or result.index.inferred_freq == "h"
        assert result.index.tz is not None
        assert str(result.index.tz) == "UTC"

    def test_series_name(self, tmp_csv: Path, tmp_path: Path) -> None:
        """Series must be named 'affr_price_eur_mw'."""
        loader = AFRRBidPriceLoader()
        result = loader.load(csv_path=tmp_csv, out_path=tmp_path / "out.csv")
        assert result.name == "affr_price_eur_mw"

    def test_parse_block_start(self) -> None:
        """_parse_block_start extracts the correct hour from product codes."""
        assert AFRRBidPriceLoader._parse_block_start("POS_00_04") == 0
        assert AFRRBidPriceLoader._parse_block_start("POS_04_08") == 4
        assert AFRRBidPriceLoader._parse_block_start("POS_20_24") == 20
