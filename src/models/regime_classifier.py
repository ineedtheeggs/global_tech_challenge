"""
Market regime classifier for aFRR Opportunity Cost Calculator.

Classifies each PTU (hourly observation) into one of three market regimes
based on CCGT generation percentiles:
  - high:   CCGT generation > 75th percentile
  - medium: CCGT generation between 25th and 75th percentile
  - low:    CCGT generation < 25th percentile

Usage:
    python src/models/regime_classifier.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Allow running as a script from the project root
sys.path.insert(0, str(Path(__file__).parents[2]))

from src.utils.config import (
    ANALYSIS_END,
    ANALYSIS_START,
    DATA_PROCESSED,
    REGIME_THRESHOLDS,
)
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = ["ccgt_generation_mw", "affr_price_eur_mw", "da_price_eur_mwh", "css"]
OUTPUT_PATH = DATA_PROCESSED / "market_regimes.csv"


def load_and_filter(path: Path) -> pd.DataFrame:
    """Load combined dataset and filter to the analysis window.

    Args:
        path: Path to combined_dataset.csv.

    Returns:
        Filtered DataFrame with datetime index.
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    logger.info("Loaded combined dataset: %d rows", len(df))

    df = df.loc[ANALYSIS_START:ANALYSIS_END]
    logger.info("After date filter (%s → %s): %d rows", ANALYSIS_START, ANALYSIS_END, len(df))

    before = len(df)
    df = df.dropna(subset=REQUIRED_COLUMNS)
    logger.info("After dropping NaN rows: %d rows (dropped %d)", len(df), before - len(df))

    return df


def assign_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """Assign regime labels based on CCGT generation percentiles.

    Thresholds are computed on the filtered dataset so they represent
    the actual distribution within the analysis window.

    Args:
        df: Filtered DataFrame with 'ccgt_generation_mw' column.

    Returns:
        DataFrame with added 'regime' column.
    """
    low_threshold = df["ccgt_generation_mw"].quantile(REGIME_THRESHOLDS["medium"])
    high_threshold = df["ccgt_generation_mw"].quantile(REGIME_THRESHOLDS["high"])

    logger.info(
        "CCGT generation thresholds — low/medium boundary: %.1f MW, medium/high boundary: %.1f MW",
        low_threshold,
        high_threshold,
    )

    def _label(val: float) -> str:
        if val > high_threshold:
            return "high"
        if val >= low_threshold:
            return "medium"
        return "low"

    df = df.copy()
    df["regime"] = df["ccgt_generation_mw"].map(_label)
    return df


def log_regime_distribution(df: pd.DataFrame) -> None:
    """Log count and percentage breakdown by regime."""
    counts = df["regime"].value_counts().sort_index()
    total = len(df)
    for regime, count in counts.items():
        logger.info("Regime %-8s: %5d rows (%5.1f%%)", regime, count, 100 * count / total)


def run() -> pd.DataFrame:
    """Execute the full regime classification pipeline.

    Returns:
        DataFrame with regime labels, also saved to disk.
    """
    combined_path = DATA_PROCESSED / "combined_dataset.csv"
    if not combined_path.exists():
        raise FileNotFoundError(
            f"Combined dataset not found at {combined_path}. "
            "Run the data pipeline first: python src/data_pipeline/main.py"
        )

    df = load_and_filter(combined_path)
    df = assign_regimes(df)
    log_regime_distribution(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH)
    logger.info("Market regimes saved to %s", OUTPUT_PATH)

    return df


if __name__ == "__main__":
    run()
