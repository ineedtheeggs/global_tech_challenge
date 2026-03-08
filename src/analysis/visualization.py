"""
Static visualization module for aFRR opportunity cost analysis.

Generates 5 PNG plots:
1. CSS vs aFRR scatter (coloured by regime)
2. Regime distribution (bar + time series)
3. Model diagnostic grid (residuals, Q-Q, scale-location)
4. Coefficient comparison across regimes
5. Forecast curves (aFRR price vs DA price by regime)

Usage:
    python src/analysis/visualization.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — must be before pyplot import

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats
from statsmodels.nonparametric.smoothers_lowess import lowess

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.models.predictions import _load_metadata, estimate_afrr_price
from src.utils.config import (
    DATA_PROCESSED,
    FORECAST_DA_MAX,
    FORECAST_DA_MIN,
    FORECAST_DA_STEPS,
    FORECAST_ETS_PRICE,
    FORECAST_GAS_PRICE,
    OUTPUTS_MODELS,
    OUTPUTS_PLOTS,
)
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

REGIME_COLORS: dict[str, str] = {
    "high": "#2196F3",
    "medium": "#FF9800",
    "low": "#F44336",
}
REGIME_ORDER: list[str] = ["high", "medium", "low"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_model(regime: str, outputs_models: Path):
    """Load pickled OLS model for a regime."""
    path = outputs_models / f"regime_{regime}_model.pkl"
    with open(path, "rb") as fh:
        return pickle.load(fh)


# ---------------------------------------------------------------------------
# Plot 1: CSS vs aFRR prices
# ---------------------------------------------------------------------------

def plot_css_vs_affr(
    df: pd.DataFrame,
    outputs_plots: Path = OUTPUTS_PLOTS,
    outputs_models: Path = OUTPUTS_MODELS,
) -> Path:
    """Scatter plot of CSS vs aFRR prices, coloured and trend-lined per regime.

    Args:
        df: DataFrame with 'css', 'affr_price_eur_mw', 'regime' columns.
        outputs_plots: Directory to save PNG into.
        outputs_models: Not used; included for consistent API.

    Returns:
        Path to saved PNG file.
    """
    outputs_plots.mkdir(parents=True, exist_ok=True)
    out_path = outputs_plots / "css_vs_affr_prices.png"

    y_col = "affr_price_eur_mw"
    y_clip = df[y_col].quantile(0.99)
    clipped_count = (df[y_col] > y_clip).sum()

    fig, ax = plt.subplots(figsize=(10, 6))

    for regime in REGIME_ORDER:
        sub = df[df["regime"] == regime]
        ax.scatter(
            sub["css"],
            sub[y_col].clip(upper=y_clip),
            color=REGIME_COLORS[regime],
            alpha=0.4,
            s=10,
            label=f"{regime.capitalize()} regime (n={len(sub):,})",
            rasterized=True,
        )
        # OLS trend line
        if len(sub) > 1:
            coeffs = np.polyfit(sub["css"], sub[y_col].clip(upper=y_clip), 1)
            x_range = np.linspace(sub["css"].min(), sub["css"].max(), 200)
            ax.plot(x_range, np.polyval(coeffs, x_range), color=REGIME_COLORS[regime], linewidth=2)

    if clipped_count:
        ax.annotate(
            f"{clipped_count} pts clipped at 99th pctile ({y_clip:.0f} €/MW)",
            xy=(0.01, 0.97),
            xycoords="axes fraction",
            va="top",
            fontsize=8,
            color="gray",
        )

    ax.set_xlabel("Clean Spark Spread (€/MWh)", fontsize=11)
    ax.set_ylabel("aFRR-up Price (€/MW)", fontsize=11)
    ax.set_title("CSS vs aFRR-up Prices by Market Regime", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Plot 2: Regime distribution
# ---------------------------------------------------------------------------

def plot_regime_distribution(
    df: pd.DataFrame,
    outputs_plots: Path = OUTPUTS_PLOTS,
    outputs_models: Path = OUTPUTS_MODELS,
) -> Path:
    """Bar chart of regime counts and scatter of regime over time.

    Args:
        df: DataFrame with 'regime' column and DatetimeIndex.
        outputs_plots: Directory to save PNG into.
        outputs_models: Not used; included for consistent API.

    Returns:
        Path to saved PNG file.
    """
    outputs_plots.mkdir(parents=True, exist_ok=True)
    out_path = outputs_plots / "regime_distribution.png"

    fig, (ax_bar, ax_ts) = plt.subplots(
        1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1, 3]}
    )

    # Left: bar chart
    counts = df["regime"].value_counts().reindex(REGIME_ORDER)
    total = counts.sum()
    bars = ax_bar.bar(
        REGIME_ORDER,
        counts.values,
        color=[REGIME_COLORS[r] for r in REGIME_ORDER],
        edgecolor="white",
        width=0.6,
    )
    for bar, count in zip(bars, counts.values):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.005,
            f"{count:,}\n({100 * count / total:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax_bar.set_xlabel("Market Regime", fontsize=11)
    ax_bar.set_ylabel("PTU Count", fontsize=11)
    ax_bar.set_title("Regime Distribution", fontsize=12)
    ax_bar.grid(True, axis="y", alpha=0.3)

    # Right: regime over time
    regime_num = df["regime"].map({"low": 0, "medium": 1, "high": 2})
    idx = df.index
    if hasattr(idx, "tz") and idx.tz is not None:
        idx = idx.tz_localize(None)

    color_array = [REGIME_COLORS[r] for r in df["regime"]]
    ax_ts.scatter(idx, regime_num.values, c=color_array, s=3, alpha=0.5, rasterized=True)
    ax_ts.set_yticks([0, 1, 2])
    ax_ts.set_yticklabels(["Low", "Medium", "High"])
    ax_ts.set_xlabel("Date", fontsize=11)
    ax_ts.set_title("Market Regime Over Time", fontsize=12)
    ax_ts.grid(True, alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=REGIME_COLORS[r], label=r.capitalize()) for r in REGIME_ORDER]
    ax_ts.legend(handles=legend_elements, loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Plot 3: Model diagnostics
# ---------------------------------------------------------------------------

def plot_model_diagnostics(
    df: pd.DataFrame,
    outputs_plots: Path = OUTPUTS_PLOTS,
    outputs_models: Path = OUTPUTS_MODELS,
) -> Path:
    """3×3 diagnostic grid: residuals vs fitted, Q-Q, scale-location per regime.

    Args:
        df: DataFrame (not directly used; models loaded from disk).
        outputs_plots: Directory to save PNG into.
        outputs_models: Directory containing regime pkl model files.

    Returns:
        Path to saved PNG file.
    """
    outputs_plots.mkdir(parents=True, exist_ok=True)
    out_path = outputs_plots / "model_diagnostics.png"

    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    row_titles = ["Residuals vs Fitted", "Normal Q-Q", "Scale-Location"]

    for col_idx, regime in enumerate(REGIME_ORDER):
        model = _load_model(regime, outputs_models)
        resid = model.resid.values
        fitted = model.fittedvalues.values
        std_resid = (resid - resid.mean()) / resid.std()

        color = REGIME_COLORS[regime]

        # Row 0: Residuals vs Fitted
        ax = axes[0, col_idx]
        ax.scatter(fitted, resid, color=color, alpha=0.3, s=8, rasterized=True)
        ax.axhline(0, color="black", linewidth=1, linestyle="--")
        ax.set_xlabel("Fitted values")
        ax.set_ylabel("Residuals" if col_idx == 0 else "")
        ax.set_title(f"{regime.capitalize()} — {row_titles[0]}", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Row 1: Q-Q plot
        ax = axes[1, col_idx]
        (osm, osr), (slope, intercept, _r) = scipy.stats.probplot(resid, fit=True)
        ax.scatter(osm, osr, color=color, alpha=0.4, s=8, rasterized=True)
        x_ref = np.array([osm[0], osm[-1]])
        ax.plot(x_ref, slope * x_ref + intercept, color="black", linewidth=1.5)
        ax.set_xlabel("Theoretical quantiles")
        ax.set_ylabel("Sample quantiles" if col_idx == 0 else "")
        ax.set_title(f"{regime.capitalize()} — {row_titles[1]}", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Row 2: Scale-Location
        ax = axes[2, col_idx]
        sqrt_abs_std = np.sqrt(np.abs(std_resid))
        ax.scatter(fitted, sqrt_abs_std, color=color, alpha=0.3, s=8, rasterized=True)
        # LOWESS smoother
        sort_idx = np.argsort(fitted)
        smoothed = lowess(sqrt_abs_std[sort_idx], fitted[sort_idx], frac=0.3)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color="red", linewidth=1.5)
        ax.set_xlabel("Fitted values")
        ax.set_ylabel("√|Std. Residuals|" if col_idx == 0 else "")
        ax.set_title(f"{regime.capitalize()} — {row_titles[2]}", fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle("OLS Model Diagnostics by Market Regime", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Plot 4: Coefficient comparison
# ---------------------------------------------------------------------------

def plot_coefficient_comparison(
    df: pd.DataFrame,
    outputs_plots: Path = OUTPUTS_PLOTS,
    outputs_models: Path = OUTPUTS_MODELS,
) -> Path:
    """Grouped bar chart of OLS coefficients with 95% confidence intervals.

    Note: ccgt_generation_mw coefficient is scaled ×1000 for display.

    Args:
        df: DataFrame (not directly used; models loaded from disk).
        outputs_plots: Directory to save PNG into.
        outputs_models: Directory containing regime pkl model files.

    Returns:
        Path to saved PNG file.
    """
    outputs_plots.mkdir(parents=True, exist_ok=True)
    out_path = outputs_plots / "coefficient_comparison.png"

    predictors = ["css", "da_price_eur_mwh", "ccgt_generation_mw"]
    predictor_labels = ["CSS", "DA Price", "CCGT Gen (×1000 MW)"]

    params: dict[str, dict] = {}
    ci_low: dict[str, dict] = {}
    ci_high: dict[str, dict] = {}

    for regime in REGIME_ORDER:
        model = _load_model(regime, outputs_models)
        p = model.params
        ci = model.conf_int()
        params[regime] = {}
        ci_low[regime] = {}
        ci_high[regime] = {}
        for pred in predictors:
            scale = 1000.0 if pred == "ccgt_generation_mw" else 1.0
            params[regime][pred] = p[pred] * scale
            ci_low[regime][pred] = ci.loc[pred, 0] * scale
            ci_high[regime][pred] = ci.loc[pred, 1] * scale

    n_groups = len(predictors)
    n_regimes = len(REGIME_ORDER)
    bar_width = 0.25
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(11, 6))

    for r_idx, regime in enumerate(REGIME_ORDER):
        offsets = x + (r_idx - 1) * bar_width
        values = [params[regime][p] for p in predictors]
        lows = [params[regime][p] - ci_low[regime][p] for p in predictors]
        highs = [ci_high[regime][p] - params[regime][p] for p in predictors]
        ax.bar(
            offsets,
            values,
            width=bar_width,
            color=REGIME_COLORS[regime],
            label=f"{regime.capitalize()} regime",
            edgecolor="white",
            yerr=[lows, highs],
            capsize=4,
        )

    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(predictor_labels, fontsize=11)
    ax.set_ylabel("Coefficient Value", fontsize=11)
    ax.set_title("OLS Coefficient Comparison by Market Regime (with 95% CI)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Plot 5: Forecast curves
# ---------------------------------------------------------------------------

def plot_forecast_curves(
    df: pd.DataFrame,
    outputs_plots: Path = OUTPUTS_PLOTS,
    outputs_models: Path = OUTPUTS_MODELS,
) -> Path:
    """aFRR price vs DA price forecast curves per regime.

    Uses FORECAST_GAS_PRICE and FORECAST_ETS_PRICE from config.
    CCGT generation set to each regime's training mean.

    Note: medium/low slopes may be negative — this is a real model finding
    from 2021 data where DA price and CSS moved together in those regimes.

    Args:
        df: DataFrame (not directly used; CCGT means loaded from metadata).
        outputs_plots: Directory to save PNG into.
        outputs_models: Directory containing regime pkl model files and metadata.

    Returns:
        Path to saved PNG file.
    """
    outputs_plots.mkdir(parents=True, exist_ok=True)
    out_path = outputs_plots / "forecast_curves.png"

    metadata = _load_metadata()
    da_range = np.linspace(FORECAST_DA_MIN, FORECAST_DA_MAX, FORECAST_DA_STEPS)

    fig, ax = plt.subplots(figsize=(10, 6))

    for regime in REGIME_ORDER:
        ccgt_mean = metadata[regime]["ccgt_mean_mw"]
        prices = [
            estimate_afrr_price(
                da_price=float(da),
                gas_price=FORECAST_GAS_PRICE,
                eu_ets_price=FORECAST_ETS_PRICE,
                regime=regime,
                ccgt_generation=ccgt_mean,
            )
            for da in da_range
        ]
        ax.plot(
            da_range,
            prices,
            color=REGIME_COLORS[regime],
            linewidth=2.5,
            label=f"{regime.capitalize()} regime",
        )

    # Scenario annotation box
    scenario_text = (
        f"Gas: {FORECAST_GAS_PRICE:.0f} €/MWh\n"
        f"ETS: {FORECAST_ETS_PRICE:.0f} €/tCO₂\n"
        f"CCGT: regime mean"
    )
    ax.text(
        0.98, 0.97,
        scenario_text,
        transform=ax.transAxes,
        va="top", ha="right",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "lightyellow", "edgecolor": "gray"},
    )

    ax.set_xlabel("DA Price Forecast (€/MWh)", fontsize=11)
    ax.set_ylabel("Estimated aFRR-up Price (€/MW)", fontsize=11)
    ax.set_title("Forecast: aFRR-up Opportunity Cost vs DA Price", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    csv_path = DATA_PROCESSED / "market_regimes.csv"
    logger.info("Loading %s", csv_path)
    df_main = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if df_main.index.tz is None:
        df_main.index = df_main.index.tz_localize("UTC")

    OUTPUTS_PLOTS.mkdir(parents=True, exist_ok=True)

    plot_css_vs_affr(df_main)
    plot_regime_distribution(df_main)
    plot_model_diagnostics(df_main)
    plot_coefficient_comparison(df_main)
    plot_forecast_curves(df_main)

    logger.info("All 5 plots saved to %s", OUTPUTS_PLOTS)
