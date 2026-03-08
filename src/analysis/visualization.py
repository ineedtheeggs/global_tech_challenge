"""
Static visualization module for aFRR opportunity cost analysis.

Generates 3 PNG plots:
1. Historical aFRR-up capacity prices over time (coloured by regime)
2. CSS vs aFRR scatter by regime (model evidence)
3. Forecast curves — aFRR price vs gas forward price, parametrised by DA scenario

Usage:
    python src/analysis/visualization.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — must be before pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.models.predictions import estimate_afrr_price
from src.utils.config import (
    DATA_PROCESSED,
    FORECAST_DA_SCENARIOS,
    FORECAST_ETS_PRICE,
    FORECAST_GAS_MAX,
    FORECAST_GAS_MIN,
    FORECAST_GAS_STEPS,
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
DA_SCENARIO_COLORS: list[str] = ["#1E88E5", "#43A047", "#FB8C00", "#E53935"]


# ---------------------------------------------------------------------------
# Plot 1: Historical aFRR prices
# ---------------------------------------------------------------------------

def plot_historical_afrr(
    df: pd.DataFrame,
    outputs_plots: Path = OUTPUTS_PLOTS,
    outputs_models: Path = OUTPUTS_MODELS,
) -> Path:
    """Time-series of historical aFRR-up capacity prices, coloured by market regime.

    Args:
        df: DataFrame with 'affr_price_eur_mw', 'regime' columns and DatetimeIndex.
        outputs_plots: Directory to save PNG into.
        outputs_models: Not used; included for consistent API.

    Returns:
        Path to saved PNG file.
    """
    outputs_plots.mkdir(parents=True, exist_ok=True)
    out_path = outputs_plots / "historical_afrr_prices.png"

    idx = df.index
    if hasattr(idx, "tz") and idx.tz is not None:
        idx = idx.tz_localize(None)

    fig, ax = plt.subplots(figsize=(14, 5))

    for regime in REGIME_ORDER:
        mask = df["regime"] == regime
        ax.scatter(
            idx[mask],
            df.loc[mask, "affr_price_eur_mw"],
            color=REGIME_COLORS[regime],
            s=4,
            alpha=0.6,
            label=f"{regime.capitalize()} regime",
            rasterized=True,
        )

    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("aFRR-up Capacity Price (€/MW)", fontsize=11)
    ax.set_title("Historical aFRR-up Capacity Prices by Market Regime", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Plot 2: CSS vs aFRR prices
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
# Plot 3: Forecast curves
# ---------------------------------------------------------------------------

def plot_forecast_curves(
    df: pd.DataFrame,
    outputs_plots: Path = OUTPUTS_PLOTS,
    outputs_models: Path = OUTPUTS_MODELS,
) -> Path:
    """aFRR opportunity cost vs gas forward price, parametrised by DA price scenario.

    Three subplots (one per regime). X-axis = gas forward price; each line
    represents a different DA price scenario. ETS price is fixed.

    Args:
        df: Not directly used; included for consistent API.
        outputs_plots: Directory to save PNG into.
        outputs_models: Not used; estimate_afrr_price uses OUTPUTS_MODELS from config.

    Returns:
        Path to saved PNG file.
    """
    outputs_plots.mkdir(parents=True, exist_ok=True)
    out_path = outputs_plots / "forecast_curves.png"

    gas_range = np.linspace(FORECAST_GAS_MIN, FORECAST_GAS_MAX, FORECAST_GAS_STEPS)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, regime in zip(axes, REGIME_ORDER):
        for i, da_price in enumerate(FORECAST_DA_SCENARIOS):
            prices = [
                estimate_afrr_price(
                    da_price=da_price,
                    gas_price=float(gas),
                    eu_ets_price=FORECAST_ETS_PRICE,
                    regime=regime,
                )
                for gas in gas_range
            ]
            ax.plot(
                gas_range,
                prices,
                color=DA_SCENARIO_COLORS[i % len(DA_SCENARIO_COLORS)],
                linewidth=2,
                label=f"DA = {da_price:.0f} €/MWh",
            )

        ax.set_title(f"{regime.capitalize()} Regime", fontsize=11)
        ax.set_xlabel("Gas Forward Price (€/MWh_th)", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Estimated aFRR-up Price (€/MW)", fontsize=10)
    fig.suptitle(
        f"aFRR-up Opportunity Cost vs Gas Forward Price  (ETS = {FORECAST_ETS_PRICE:.0f} €/tCO₂)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    csv_path = DATA_PROCESSED / "market_regimes.csv"
    logger.info("Loading %s", csv_path)
    df_main = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if df_main.index.tz is None:
        df_main.index = df_main.index.tz_localize("UTC")

    OUTPUTS_PLOTS.mkdir(parents=True, exist_ok=True)

    plot_historical_afrr(df_main)
    plot_css_vs_affr(df_main)
    plot_forecast_curves(df_main)

    logger.info("All 3 plots saved to %s", OUTPUTS_PLOTS)
