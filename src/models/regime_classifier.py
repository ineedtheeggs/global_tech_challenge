"""
Market regime classifier for aFRR Opportunity Cost Calculator.

Classifies each PTU (hourly observation) into one of three market regimes
based on fixed MW thresholds derived from K-Means centroids (default) or
capacity-based thirds.  The boundary between two regimes is the midpoint
between the two adjacent centroids.

Toggle the classification method via REGIME_CLASSIFICATION_METHOD in config.py:
  - 'kmeans'   → boundaries derived from K-Means cluster centroids
  - 'capacity' → boundaries derived from capacity-based thirds

Usage:
    python src/models/regime_classifier.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# Allow running as a script from the project root
sys.path.insert(0, str(Path(__file__).parents[2]))

from src.utils.config import (
    ANALYSIS_END,
    ANALYSIS_START,
    DATA_PROCESSED,
    REGIME_CAPACITY_JSON,
    REGIME_CLASSIFICATION_METHOD,
    REGIME_KMEANS_JSON,
)
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = ["ccgt_generation_mw", "affr_price_eur_mw", "da_price_eur_mwh", "css"]
OUTPUT_PATH = DATA_PROCESSED / "market_regimes.csv"


def load_centroids(method: str) -> dict[str, float]:
    """Load regime centroid MW values from the appropriate JSON reference file.

    Args:
        method: Classification method — 'kmeans' or 'capacity'.

    Returns:
        Dict mapping regime name ('low', 'medium', 'high') to centroid MW.

    Raises:
        ValueError: If method is not 'kmeans' or 'capacity'.
        FileNotFoundError: If the JSON reference file does not exist.
    """
    if method == "kmeans":
        path = REGIME_KMEANS_JSON
    elif method == "capacity":
        path = REGIME_CAPACITY_JSON
    else:
        raise ValueError(
            f"Unknown classification method '{method}'. "
            "Must be 'kmeans' or 'capacity'."
        )

    if not path.exists():
        raise FileNotFoundError(
            f"Regime reference file not found: {path}. "
            "Ensure data/references/ contains the JSON file."
        )

    with open(path) as fh:
        data = json.load(fh)

    return {regime: float(info["ccgt_mean_mw"]) for regime, info in data.items()}


def compute_thresholds(centroids: dict[str, float]) -> tuple[float, float]:
    """Compute regime boundary thresholds as midpoints between adjacent centroids.

    Args:
        centroids: Dict with keys 'low', 'medium', 'high' and MW values.

    Returns:
        Tuple of (low_medium_boundary_mw, medium_high_boundary_mw).
    """
    low_medium = (centroids["low"] + centroids["medium"]) / 2.0
    medium_high = (centroids["medium"] + centroids["high"]) / 2.0
    return low_medium, medium_high


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


def assign_regimes(
    df: pd.DataFrame,
    method: str = REGIME_CLASSIFICATION_METHOD,
) -> pd.DataFrame:
    """Assign regime labels using centroid-derived MW boundaries.

    Loads centroids from the appropriate JSON reference file, computes
    midpoint thresholds, then labels each row accordingly.

    Args:
        df: Filtered DataFrame with 'ccgt_generation_mw' column.
        method: Classification method — 'kmeans' or 'capacity'.

    Returns:
        DataFrame with added 'regime' column.
    """
    centroids = load_centroids(method)
    low_medium, medium_high = compute_thresholds(centroids)

    logger.info(
        "Regime boundaries (%s) — low/medium: %.1f MW, medium/high: %.1f MW",
        method,
        low_medium,
        medium_high,
    )

    df = df.copy()
    df["regime"] = df["ccgt_generation_mw"].map(
        lambda v: "high" if v >= medium_high else ("medium" if v >= low_medium else "low")
    )
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
