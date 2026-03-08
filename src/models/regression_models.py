"""
OLS regression models for aFRR opportunity cost estimation.

Fits one model per market regime:
    aFRR_Price = β₀ + β₁·CSS + ε

CSS is the sole predictor because:
- DA price is already embedded in CSS (no double-counting)
- Market regimes already isolate CCGT generation effects

Models are persisted as pickle files; per-regime statistics are stored in
a JSON sidecar used by the report and dashboard modules.

Usage:
    python src/models/regression_models.py
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson

# Allow running as a script from the project root
sys.path.insert(0, str(Path(__file__).parents[2]))

from src.utils.config import DATA_PROCESSED, OLS_FIT_METHOD, OUTPUTS_MODELS
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

REGIMES = ["high", "medium", "low"]
FEATURES = ["css"]
TARGET = "affr_price_eur_mw"
METADATA_PATH = OUTPUTS_MODELS / "regime_metadata.json"


def fit_regime_model(
    df: pd.DataFrame,
    regime: str,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fit OLS model for a single regime.

    Args:
        df: Full labelled dataset with 'regime' column.
        regime: One of 'high', 'medium', 'low'.

    Returns:
        Fitted statsmodels OLS results object.

    Raises:
        ValueError: If the regime subset has fewer than 10 observations.
    """
    subset = df[df["regime"] == regime].copy()
    logger.info("Regime '%s': %d observations", regime, len(subset))

    if len(subset) < 10:
        raise ValueError(
            f"Regime '{regime}' has only {len(subset)} rows — insufficient for regression."
        )

    X = subset[FEATURES]
    y = subset[TARGET]

    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit(method=OLS_FIT_METHOD)

    dw_stat = durbin_watson(model.resid)
    css_coef = model.params.get("css", float("nan"))
    css_pval = model.pvalues.get("css", float("nan"))

    logger.info(
        "Regime '%s' | R²=%.4f | Adj-R²=%.4f | DW=%.3f | β(CSS)=%+.4f (p=%.4f)",
        regime,
        model.rsquared,
        model.rsquared_adj,
        dw_stat,
        css_coef,
        css_pval,
    )

    return model


def save_model(model: sm.regression.linear_model.RegressionResultsWrapper, regime: str) -> Path:
    """Pickle the fitted model to disk.

    Args:
        model: Fitted OLS results.
        regime: Regime label used to construct the filename.

    Returns:
        Path where the model was saved.
    """
    OUTPUTS_MODELS.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_MODELS / f"regime_{regime}_model.pkl"
    with open(path, "wb") as fh:
        pickle.dump(model, fh)
    logger.info("Model saved: %s", path)
    return path


def build_metadata(df: pd.DataFrame) -> dict[str, dict]:
    """Compute per-regime summary statistics used by the predictions module.

    Args:
        df: Labelled dataset with 'regime' column.

    Returns:
        Dict keyed by regime with mean CCGT generation and other stats.
    """
    metadata: dict[str, dict] = {}
    for regime in REGIMES:
        subset = df[df["regime"] == regime]
        metadata[regime] = {
            "ccgt_mean_mw": float(subset["ccgt_generation_mw"].mean()),
            "ccgt_std_mw": float(subset["ccgt_generation_mw"].std()),
            "n_observations": int(len(subset)),
        }
    return metadata


def run() -> dict[str, sm.regression.linear_model.RegressionResultsWrapper]:
    """Execute the full model-fitting pipeline.

    Returns:
        Dict mapping regime name to fitted OLS results.
    """
    regimes_path = DATA_PROCESSED / "market_regimes.csv"
    if not regimes_path.exists():
        raise FileNotFoundError(
            f"Regime-labelled dataset not found at {regimes_path}. "
            "Run regime classifier first: python src/models/regime_classifier.py"
        )

    df = pd.read_csv(regimes_path, index_col=0, parse_dates=True)
    logger.info("Loaded regime dataset: %d rows", len(df))

    # Drop any residual NaN rows in the required columns
    required = FEATURES + [TARGET, "regime"]
    df = df.dropna(subset=required)
    logger.info("After NaN drop: %d rows", len(df))

    models: dict[str, sm.regression.linear_model.RegressionResultsWrapper] = {}

    for regime in REGIMES:
        logger.info("=" * 60)
        logger.info("Fitting model for regime: %s", regime.upper())
        logger.info("=" * 60)
        model = fit_regime_model(df, regime)
        save_model(model, regime)
        models[regime] = model

    # Save metadata sidecar
    metadata = build_metadata(df)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_PATH, "w") as fh:
        json.dump(metadata, fh, indent=2)
    logger.info("Regime metadata saved: %s", METADATA_PATH)

    return models


if __name__ == "__main__":
    run()
