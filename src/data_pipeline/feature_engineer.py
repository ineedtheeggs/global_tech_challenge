"""
Feature engineering module: compute Clean Spark Spread (CSS) and helper columns.

CSS formula (from CLAUDE.md):
    carbon_cost = eu_ets_price * CARBON_INTENSITY / EFFICIENCY
    css = da_price - (gas_price / EFFICIENCY) - carbon_cost

Additional features:
    hour_of_day, day_of_week, month, year  (for regime analysis context)
    css_lag_24h, css_rolling_24h_mean      (autocorrelation features)
"""

from __future__ import annotations

import pandas as pd

from src.utils.config import CARBON_INTENSITY, EFFICIENCY
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def calculate_css(
    da_price: float | pd.Series,
    gas_price: float | pd.Series,
    eu_ets_price: float | pd.Series,
    efficiency: float = EFFICIENCY,
    carbon_intensity: float = CARBON_INTENSITY,
) -> float | pd.Series:
    """
    Calculate Clean Spark Spread (CSS) for a CCGT peaker plant.

    The CSS represents the per-MWh gross margin from running the plant:
    it is the electricity revenue minus the fuel cost and carbon cost,
    normalised by plant efficiency.

    Args:
        da_price:         Day-ahead electricity price (€/MWh_e).
        gas_price:        Gas day-ahead price (€/MWh_th).
        eu_ets_price:     EU-ETS carbon allowance spot price (€/tCO₂).
        efficiency:       Plant thermal efficiency (default 0.50 = 50%).
        carbon_intensity: CO₂ emissions per MWh of gas input (tCO₂/MWh_th).
                          Default 0.202 (natural gas).

    Returns:
        CSS value(s) in €/MWh_e. Positive → profitable to generate.

    Raises:
        ValueError: If efficiency is zero or negative.
    """
    if isinstance(efficiency, (int, float)) and efficiency <= 0:
        raise ValueError(f"efficiency must be positive, got {efficiency}")

    fuel_cost = gas_price / efficiency
    carbon_cost = eu_ets_price * carbon_intensity / efficiency
    css = da_price - fuel_cost - carbon_cost
    return css


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar-based columns for regime analysis context.

    Args:
        df: DataFrame with a UTC DatetimeIndex.

    Returns:
        DataFrame with additional columns:
            hour_of_day, day_of_week, month, year, is_weekend
    """
    df = df.copy()
    df["hour_of_day"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek   # 0=Monday, 6=Sunday
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def add_lag_features(df: pd.DataFrame, lag_hours: int = 24) -> pd.DataFrame:
    """
    Add lagged and rolling CSS features to capture autocorrelation.

    Args:
        df:        DataFrame containing a 'css' column.
        lag_hours: Number of hours for the lag (default 24).

    Returns:
        DataFrame with additional columns:
            css_lag_{lag_hours}h, css_rolling_{lag_hours}h_mean
    """
    if "css" not in df.columns:
        raise KeyError("DataFrame must contain a 'css' column before adding lag features.")

    df = df.copy()
    df[f"css_lag_{lag_hours}h"] = df["css"].shift(lag_hours)
    df[f"css_rolling_{lag_hours}h_mean"] = (
        df["css"].rolling(window=lag_hours, min_periods=1).mean()
    )
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline applied to the cleaned combined dataset.

    Args:
        df: Cleaned DataFrame with columns:
            da_price_eur_mwh, gas_price_eur_mwh_th, eu_ets_price_eur_tco2,
            affr_price_eur_mw, ccgt_generation_mw

    Returns:
        DataFrame with added columns: css, time features, lag features.
    """
    required = [
        "da_price_eur_mwh",
        "gas_price_eur_mwh_th",
        "eu_ets_price_eur_tco2",
    ]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")

    logger.info("Engineering features …")
    df = df.copy()

    # CSS
    df["css"] = calculate_css(
        da_price=df["da_price_eur_mwh"],
        gas_price=df["gas_price_eur_mwh_th"],
        eu_ets_price=df["eu_ets_price_eur_tco2"],
    )
    logger.info(
        "  CSS: mean=%.2f, std=%.2f, min=%.2f, max=%.2f",
        df["css"].mean(),
        df["css"].std(),
        df["css"].min(),
        df["css"].max(),
    )

    # Time features
    df = add_time_features(df)

    # Lag features
    df = add_lag_features(df, lag_hours=24)

    logger.info("Feature engineering complete: %d columns", len(df.columns))
    return df
