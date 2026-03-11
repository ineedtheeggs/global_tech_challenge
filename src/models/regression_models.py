"""
OLS regression models for aFRR opportunity cost estimation.

Fits one model per year × regime combination (up to 18 models over 6 years):
    aFRR_Price = β₀ + β₁·CSS + ε

CSS is the sole predictor because:
- DA price is already embedded in CSS (no double-counting)
- Market regimes already isolate CCGT generation effects

Models are persisted as pickle files named regime_{regime}_{year}_model.pkl.
The metadata JSON is nested by year and also contains flat regime keys for the
most recent complete year, preserving backward compatibility with report and
dashboard modules.

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
MIN_OBSERVATIONS = 20


def fit_regime_model(
    subset: pd.DataFrame,
    regime: str,
    year: int | None = None,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fit OLS model on a pre-filtered regime subset.

    Args:
        subset: DataFrame already filtered to the desired regime (and year).
        regime: One of 'high', 'medium', 'low' (used for logging only).
        year: Optional year for logging context.

    Returns:
        Fitted statsmodels OLS results object.

    Raises:
        ValueError: If the subset has fewer than MIN_OBSERVATIONS observations.
    """
    year_tag = f" [{year}]" if year is not None else ""
    logger.info("Regime '%s'%s: %d observations", regime, year_tag, len(subset))

    if len(subset) < MIN_OBSERVATIONS:
        raise ValueError(
            f"Regime '{regime}'{year_tag} has only {len(subset)} rows "
            f"— minimum required is {MIN_OBSERVATIONS}."
        )

    X = subset[FEATURES]
    y = subset[TARGET]

    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit(method=OLS_FIT_METHOD)

    dw_stat = durbin_watson(model.resid)
    css_coef = model.params.get("css", float("nan"))
    css_pval = model.pvalues.get("css", float("nan"))

    logger.info(
        "Regime '%s'%s | R²=%.4f | Adj-R²=%.4f | DW=%.3f | β(CSS)=%+.4f (p=%.4f)",
        regime,
        year_tag,
        model.rsquared,
        model.rsquared_adj,
        dw_stat,
        css_coef,
        css_pval,
    )

    return model


def build_year_regime_metadata(
    subset: pd.DataFrame,
    model: sm.regression.linear_model.RegressionResultsWrapper,
    regime: str,
    year: int,
) -> dict:
    """Build a metadata dict for a single (year, regime) model.

    Args:
        subset: Filtered DataFrame for this year/regime.
        model: Fitted OLS results.
        regime: Regime label.
        year: Year of the model.

    Returns:
        Dict with model statistics and data summary.
    """
    dw_stat = durbin_watson(model.resid)
    return {
        "beta_0": float(model.params.get("const", float("nan"))),
        "beta_1_css": float(model.params.get("css", float("nan"))),
        "rsquared": float(model.rsquared),
        "rsquared_adj": float(model.rsquared_adj),
        "durbin_watson": float(dw_stat),
        "n_observations": int(len(subset)),
        "css_pvalue": float(model.pvalues.get("css", float("nan"))),
        "ccgt_mean_mw": float(subset["ccgt_generation_mw"].mean()),
        "ccgt_std_mw": float(subset["ccgt_generation_mw"].std()),
    }


def save_model(
    model: sm.regression.linear_model.RegressionResultsWrapper,
    regime: str,
    year: int,
) -> Path:
    """Pickle the fitted model to disk.

    Args:
        model: Fitted OLS results.
        regime: Regime label.
        year: Year associated with this model.

    Returns:
        Path where the model was saved.
    """
    OUTPUTS_MODELS.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_MODELS / f"regime_{regime}_{year}_model.pkl"
    with open(path, "wb") as fh:
        pickle.dump(model, fh)
    logger.info("Model saved: %s", path)
    return path


def run() -> dict[str, sm.regression.linear_model.RegressionResultsWrapper]:
    """Execute the per-year, per-regime model-fitting pipeline.

    Iterates over each year present in the data and fits one OLS model per
    regime. Saves up to 18 pickle files and writes a nested metadata JSON
    that also contains flat compat keys for the most recent complete year.

    Returns:
        Dict mapping "{year}_{regime}" to fitted OLS results.
    """
    regimes_path = DATA_PROCESSED / "market_regimes.csv"
    if not regimes_path.exists():
        raise FileNotFoundError(
            f"Regime-labelled dataset not found at {regimes_path}. "
            "Run regime classifier first: python src/models/regime_classifier.py"
        )

    df = pd.read_csv(regimes_path, index_col=0, parse_dates=True)
    logger.info("Loaded regime dataset: %d rows", len(df))

    required = FEATURES + [TARGET, "regime", "ccgt_generation_mw"]
    df = df.dropna(subset=required)
    logger.info("After NaN drop: %d rows", len(df))

    years = sorted(df.index.year.unique())
    logger.info("Years in dataset: %s", years)

    all_metadata: dict = {}
    models: dict[str, sm.regression.linear_model.RegressionResultsWrapper] = {}

    for year in years:
        df_year = df[df.index.year == year]
        all_metadata[str(year)] = {}

        for regime in REGIMES:
            logger.info("=" * 60)
            logger.info("Fitting model: year=%d  regime=%s", year, regime.upper())
            logger.info("=" * 60)

            subset = df_year[df_year["regime"] == regime].copy()

            if len(subset) < MIN_OBSERVATIONS:
                logger.warning(
                    "Skipping (%d, %s): only %d observations (minimum %d)",
                    year,
                    regime,
                    len(subset),
                    MIN_OBSERVATIONS,
                )
                continue

            model = fit_regime_model(subset, regime, year=year)
            save_model(model, regime, year)
            all_metadata[str(year)][regime] = build_year_regime_metadata(
                subset, model, regime, year
            )
            models[f"{year}_{regime}"] = model

    # Backward-compat flat keys: use the most recent year that has all 3 regimes
    complete_years = [y for y in years if len(all_metadata.get(str(y), {})) == 3]
    if complete_years:
        summary_year = max(complete_years)
        logger.info(
            "Adding flat compat keys from year %d (most recent complete year)", summary_year
        )
        for regime in REGIMES:
            all_metadata[regime] = all_metadata[str(summary_year)][regime]
    else:
        logger.warning("No year with all 3 complete regime models found — no flat compat keys written")

    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_PATH, "w") as fh:
        json.dump(all_metadata, fh, indent=2)
    logger.info("Regime metadata saved: %s", METADATA_PATH)

    return models


if __name__ == "__main__":
    run()
