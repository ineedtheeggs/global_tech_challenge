"""
Data cleaning module: align timestamps, handle missing values, and remove outliers.

Steps:
    1. Align all 5 series to a common hourly UTC DatetimeIndex (2020-01-01 to 2023-12-31)
    2. Handle missing values (interpolation for DA/CCGT; forward-fill for prices)
    3. Flag and replace outliers (Z-score > threshold → rolling median)
    4. Validate final data quality (assert < MAX_MISSING_RATE)
"""

from __future__ import annotations

import pandas as pd

from src.utils.config import (
    ANALYSIS_END,
    ANALYSIS_START,
    MAX_INTERPOLATION_GAP,
    MAX_MISSING_RATE,
    OUTLIER_THRESHOLD,
)
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def build_hourly_index() -> pd.DatetimeIndex:
    """
    Return the canonical hourly UTC DatetimeIndex for the analysis period.

    Returns:
        pd.DatetimeIndex from ANALYSIS_START to ANALYSIS_END (inclusive) at 1h frequency.
    """
    return pd.date_range(start=ANALYSIS_START, end=ANALYSIS_END, freq="h", tz="UTC")


def align_to_index(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """
    Reindex a series to the canonical hourly index.

    Args:
        series: Input time series with a DatetimeIndex.
        index:  Target hourly UTC DatetimeIndex.

    Returns:
        Reindexed series (may contain NaNs for missing hours).
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        return pd.Series(index=index, name=series.name, dtype=series.dtype)

    if series.index.tz is None:
        series = series.copy()
        series.index = series.index.tz_localize("UTC")
    elif str(series.index.tz) != "UTC":
        series = series.copy()
        series.index = series.index.tz_convert("UTC")

    return series.reindex(index)


def fill_interpolated(
    series: pd.Series,
    max_gap: int = MAX_INTERPOLATION_GAP,
) -> pd.Series:
    """
    Fill gaps using linear interpolation up to `max_gap` consecutive hours.

    Gaps longer than `max_gap` are left as NaN (flagged for review).

    Args:
        series:  Time series that may contain NaN values.
        max_gap: Maximum consecutive NaN count to interpolate (hours).

    Returns:
        Series with short gaps filled by interpolation.
    """
    return series.interpolate(method="linear", limit=max_gap, limit_direction="forward")


def fill_forward(series: pd.Series) -> pd.Series:
    """
    Forward-fill the series (appropriate for daily price data resampled to hourly).

    Args:
        series: Time series with potential NaN values.

    Returns:
        Forward-filled series.
    """
    return series.ffill()


def remove_outliers(
    series: pd.Series,
    threshold: float = OUTLIER_THRESHOLD,
    window: int = 24,
) -> tuple[pd.Series, pd.Series]:
    """
    Replace outliers (|Z-score| > threshold) with the rolling median.

    Args:
        series:    Input time series.
        threshold: Z-score threshold beyond which a value is an outlier.
        window:    Rolling window size in hours for the median replacement.

    Returns:
        Tuple of (cleaned_series, outlier_mask) where outlier_mask is bool Series.
    """
    z_scores = (series - series.mean()) / series.std(ddof=1)
    outlier_mask = z_scores.abs() > threshold

    n_outliers = outlier_mask.sum()
    if n_outliers > 0:
        logger.info(
            "  Outliers detected in '%s': %d (%.2f%%) — replacing with rolling median",
            series.name,
            n_outliers,
            100 * n_outliers / len(series),
        )
        rolling_median = series.rolling(window=window, center=True, min_periods=1).median()
        cleaned = series.copy()
        cleaned[outlier_mask] = rolling_median[outlier_mask]
    else:
        cleaned = series.copy()

    return cleaned, outlier_mask


def validate_missing_rate(series: pd.Series, max_rate: float = MAX_MISSING_RATE) -> None:
    """
    Warn if the fraction of missing values exceeds the threshold.

    Args:
        series:   Cleaned time series.
        max_rate: Maximum allowed fraction of NaN values.
    """
    missing_rate = series.isna().mean()
    if missing_rate > max_rate:
        logger.warning(
            "  Column '%s' has %.2f%% missing values (threshold %.2f%%) — "
            "pipeline continues but downstream models may be affected",
            series.name,
            100 * missing_rate,
            100 * max_rate,
        )
    else:
        logger.info(
            "  Validation OK: '%s' — missing rate %.2f%%", series.name, 100 * missing_rate
        )


def clean_dataset(
    da_prices: pd.Series,
    affr_prices: pd.Series,
    ccgt_generation: pd.Series,
    eu_ets_prices: pd.Series,
    gas_prices: pd.Series,
) -> pd.DataFrame:
    """
    Full cleaning pipeline: align, fill, de-outlier, and validate all 5 series.

    Args:
        da_prices:       Hourly DA electricity prices (€/MWh).
        affr_prices:     Hourly aFRR capacity prices (€/MW/h).
        ccgt_generation: Hourly CCGT generation (MW).
        eu_ets_prices:   Hourly EU-ETS carbon prices (€/tCO₂).
        gas_prices:      Hourly gas prices (€/MWh_th).

    Returns:
        Cleaned pd.DataFrame with all 5 columns and the canonical hourly UTC index.
    """
    canonical_index = build_hourly_index()
    logger.info(
        "Cleaning dataset: canonical index %s → %s (%d hours)",
        canonical_index[0].date(),
        canonical_index[-1].date(),
        len(canonical_index),
    )

    # ------------------------------------------------------------------
    # Step 1: Align to canonical index
    # ------------------------------------------------------------------
    logger.info("Step 1: Aligning all series to hourly UTC index …")
    da = align_to_index(da_prices, canonical_index)
    affr = align_to_index(affr_prices, canonical_index)
    ccgt = align_to_index(ccgt_generation, canonical_index)
    ets = align_to_index(eu_ets_prices, canonical_index)
    gas = align_to_index(gas_prices, canonical_index)

    # ------------------------------------------------------------------
    # Step 2: Fill missing values
    # ------------------------------------------------------------------
    logger.info("Step 2: Filling missing values …")

    # DA prices and CCGT generation: short gaps can be linearly interpolated
    da = fill_interpolated(da, max_gap=MAX_INTERPOLATION_GAP)
    ccgt = fill_interpolated(ccgt, max_gap=MAX_INTERPOLATION_GAP)

    # Price series: forward-fill (price valid until next quote)
    affr = fill_forward(affr)
    ets = fill_forward(ets)
    gas = fill_forward(gas)

    # ------------------------------------------------------------------
    # Step 3: Remove outliers
    # ------------------------------------------------------------------
    logger.info("Step 3: Detecting and replacing outliers (Z > %.1f) …", OUTLIER_THRESHOLD)
    da, _ = remove_outliers(da)
    affr, _ = remove_outliers(affr)
    ccgt, _ = remove_outliers(ccgt)
    # Gas and ETS prices are daily → don't apply hourly outlier detection

    # ------------------------------------------------------------------
    # Step 4: Assemble and validate
    # ------------------------------------------------------------------
    df = pd.DataFrame(
        {
            "da_price_eur_mwh": da,
            "affr_price_eur_mw": affr,
            "ccgt_generation_mw": ccgt,
            "eu_ets_price_eur_tco2": ets,
            "gas_price_eur_mwh_th": gas,
        },
        index=canonical_index,
    )

    logger.info("Step 4: Validating missing rates …")
    for col in df.columns:
        if df[col].isna().all():
            logger.warning(
                "  Column '%s' is entirely NaN — no data available (skipping validation)", col
            )
            continue
        validate_missing_rate(df[col])

    # Log summary
    total_rows = len(df)
    missing_summary = df.isna().sum()
    logger.info("Cleaning complete: %d rows, %d columns", total_rows, len(df.columns))
    logger.info("Missing value counts:\n%s", missing_summary.to_string())

    return df
