"""
Static visualization module for aFRR opportunity cost analysis.

Generates plots:
1. Historical aFRR-up capacity prices over time (coloured by regime)
2. CSS vs aFRR scatter by year (2×3 subplot grid)
3. Nine weekly forecast vs actual plots — January 2026 (W1–W5) and February 2026 (W1–W4)
4. Regime distribution over time (stacked bar by year)

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

from src.utils.config import (
    DATA_PROCESSED,
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
# Plot 2: CSS vs aFRR prices — per-year subplots
# ---------------------------------------------------------------------------

def plot_css_vs_affr(
    df: pd.DataFrame,
    outputs_plots: Path = OUTPUTS_PLOTS,
    outputs_models: Path = OUTPUTS_MODELS,
) -> list[Path]:
    """Scatter plots of CSS vs aFRR prices — one PNG per regime, subplotted by year.

    Produces 3 files (one per regime), each containing one subplot per year.
    Files are named ``css_vs_affr_{regime}.png`` (e.g. ``css_vs_affr_high.png``).

    Args:
        df: DataFrame with 'css', 'affr_price_eur_mw', 'regime' columns and DatetimeIndex.
        outputs_plots: Directory to save PNGs into.
        outputs_models: Not used; included for consistent API.

    Returns:
        List of 3 Paths to saved PNG files, ordered high → medium → low.
    """
    outputs_plots.mkdir(parents=True, exist_ok=True)

    years = sorted(df.index.year.unique())
    n_years = len(years)
    ncols = min(3, n_years)
    nrows = (n_years + ncols - 1) // ncols

    y_col = "affr_price_eur_mw"
    y_clip = df[y_col].quantile(0.99)
    paths: list[Path] = []

    for regime in REGIME_ORDER:
        regime_df = df[df["regime"] == regime]
        out_path = outputs_plots / f"css_vs_affr_{regime}.png"

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

        for i, year in enumerate(years):
            row_idx, col_idx = divmod(i, ncols)
            ax = axes[row_idx][col_idx]
            sub = regime_df[regime_df.index.year == year]

            ax.scatter(
                sub["css"],
                sub[y_col].clip(upper=y_clip),
                color=REGIME_COLORS[regime],
                alpha=0.5,
                s=8,
                label=f"n={len(sub):,}",
                rasterized=True,
            )
            if len(sub) > 1:
                coeffs = np.polyfit(sub["css"], sub[y_col].clip(upper=y_clip), 1)
                x_range = np.linspace(sub["css"].min(), sub["css"].max(), 200)
                ax.plot(
                    x_range,
                    np.polyval(coeffs, x_range),
                    color=REGIME_COLORS[regime],
                    linewidth=1.5,
                    linestyle="--",
                )

            ax.set_title(f"{year} (n={len(sub):,})", fontsize=10)
            ax.set_xlabel("CSS (€/MWh)", fontsize=9)
            ax.grid(True, alpha=0.3)

        # Hide unused subplot cells
        for j in range(n_years, nrows * ncols):
            row_idx, col_idx = divmod(j, ncols)
            axes[row_idx][col_idx].set_visible(False)

        axes[0][0].set_ylabel("aFRR-up Price (€/MW)", fontsize=9)
        fig.suptitle(
            f"CSS vs aFRR-up Prices — {regime.capitalize()} Regime by Year",
            fontsize=13,
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.info("Saved: %s", out_path)
        paths.append(out_path)

    return paths


# ---------------------------------------------------------------------------
# Plot 3: January & February 2026 forecast vs actual — 9 weekly plots
# ---------------------------------------------------------------------------

# Week day ranges per month. January has a 5th partial week (29–31).
_MONTH_WEEK_RANGES: dict[int, list[tuple[int, int]]] = {
    1: [(1, 7), (8, 14), (15, 21), (22, 28), (29, 31)],
    2: [(1, 7), (8, 14), (15, 21), (22, 28)],
}
_MONTH_ABBR: dict[int, str] = {1: "jan", 2: "feb"}
_MONTH_NAME: dict[int, str] = {1: "January", 2: "February"}


def _plot_forecast_week(
    df_week: pd.DataFrame,
    month: int,
    week_num: int,
    outputs_plots: Path,
    available_years: list[int],
    year_colors,
    _predict_row,
) -> Path:
    """Plot actual vs model-forecast aFRR for one week of a 2026 month.

    Args:
        df_week: Subset of 2026 data for this month/week.
        month: Month number (1=January, 2=February).
        week_num: Week number within the month (1-based).
        outputs_plots: Directory to save PNG into.
        available_years: Sorted list of years with pkl models.
        year_colors: Color array (one colour per year).
        _predict_row: Callable(da, gas, ets, regime, year) -> float.

    Returns:
        Path to saved PNG file.
    """
    d_start, d_end = _MONTH_WEEK_RANGES[month][week_num - 1]
    month_abbr = _MONTH_ABBR[month]
    month_name = _MONTH_NAME[month]
    out_path = outputs_plots / f"forecast_curves_{month_abbr}_w{week_num}.png"

    if df_week.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.text(0.5, 0.5, f"No {month_name} 2026 Week {week_num} data available",
                ha="center", va="center", transform=ax.transAxes)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.warning("No data for %s 2026 Week %d — empty plot saved.", month_name, week_num)
        return out_path

    # Build forecast columns
    df_w = df_week.copy()
    for year in available_years:
        preds = []
        for _, row in df_w.iterrows():
            try:
                p = _predict_row(
                    row["da_price_eur_mwh"],
                    row["gas_price_eur_mwh_th"],
                    row["eu_ets_price_eur_tco2"],
                    row["regime"],
                    year,
                )
            except Exception:
                p = np.nan
            preds.append(p)
        df_w[f"forecast_{year}"] = preds

    idx = df_w.index
    if hasattr(idx, "tz") and idx.tz is not None:
        idx = idx.tz_localize(None)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(idx, df_w["affr_price_eur_mw"], color="black", linewidth=2,
            label="Actual aFRR", zorder=10)

    for i, year in enumerate(available_years):
        col = f"forecast_{year}"
        if col in df_w.columns and df_w[col].notna().any():
            ax.plot(idx, df_w[col], color=year_colors[i], linewidth=1.5,
                    alpha=0.8, label=f"Model {year}")

    title = (
        f"{month_name} 2026 aFRR Forecast vs Actual — "
        f"Week {week_num} ({month_name[:3]} {d_start}–{d_end})"
    )
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("aFRR-up Price (€/MW)", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, ncol=4)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", out_path)
    return out_path


def plot_2026_forecast(
    df: pd.DataFrame,
    outputs_plots: Path = OUTPUTS_PLOTS,
    outputs_models: Path = OUTPUTS_MODELS,
    data_processed: Path = DATA_PROCESSED,
) -> list[Path]:
    """Actual vs model-forecast aFRR prices for Jan & Feb 2026 — 9 weekly plots.

    Loads combined_dataset.csv, filters to 2026-01 and 2026-02, assigns regimes,
    then applies each available year's OLS model to generate forecasts.
    Produces one PNG per week: January W1–W5 and February W1–W4.

    Args:
        df: Accepted for API consistency; not used (data loaded from CSV).
        outputs_plots: Directory to save PNGs into.
        outputs_models: Directory containing per-year pkl files (injectable for tests).
        data_processed: Directory containing combined_dataset.csv (injectable for tests).

    Returns:
        List of 9 Paths to saved PNG files (Jan W1–W5, Feb W1–W4).
    """
    import pickle
    from src.models.regime_classifier import assign_regimes
    from src.utils.config import CARBON_INTENSITY, EFFICIENCY

    outputs_plots.mkdir(parents=True, exist_ok=True)

    # Load January + February 2026 data
    combined_csv = data_processed / "combined_dataset.csv"
    df_all = pd.read_csv(combined_csv, index_col=0, parse_dates=True)
    if df_all.index.tz is None:
        df_all.index = df_all.index.tz_localize("UTC")

    df_janfeb = df_all[
        (df_all.index.year == 2026) & (df_all.index.month.isin([1, 2]))
    ].copy()

    required = ["da_price_eur_mwh", "gas_price_eur_mwh_th", "eu_ets_price_eur_tco2",
                "affr_price_eur_mw", "css"]
    if not df_janfeb.empty:
        df_janfeb = assign_regimes(df_janfeb)
        df_janfeb = df_janfeb.dropna(subset=required)

    # Discover available years from pkl files (exclude 2026 — no look-ahead)
    available_years: list[int] = []
    for p in outputs_models.glob("regime_*_*_model.pkl"):
        parts = p.stem.split("_")
        if len(parts) == 4 and parts[3] == "model" and parts[1] in ("high", "medium", "low"):
            try:
                yr = int(parts[2])
                if yr != 2026:
                    available_years.append(yr)
            except ValueError:
                pass
    available_years = sorted(set(available_years))

    year_colors = plt.cm.tab10(np.linspace(0, 0.9, max(len(available_years), 1)))

    def _predict_row(da_price, gas_price, ets_price, regime, year):
        carbon_cost = ets_price * CARBON_INTENSITY / EFFICIENCY
        css = da_price - (gas_price / EFFICIENCY) - carbon_cost
        model_path = outputs_models / f"regime_{regime}_{year}_model.pkl"
        with open(model_path, "rb") as fh:
            model = pickle.load(fh)
        fv = pd.DataFrame([[1.0, css]], columns=["const", "css"])
        return float(model.predict(fv)[0])

    paths: list[Path] = []
    for month, week_ranges in _MONTH_WEEK_RANGES.items():
        df_month = df_janfeb[df_janfeb.index.month == month]
        for wk, (d_start, d_end) in enumerate(week_ranges, start=1):
            mask = (df_month.index.day >= d_start) & (df_month.index.day <= d_end)
            p = _plot_forecast_week(
                df_month[mask], month, wk, outputs_plots, available_years, year_colors, _predict_row
            )
            paths.append(p)

    logger.info("Saved %d weekly forecast plots (Jan+Feb 2026) to %s", len(paths), outputs_plots)
    return paths


# ---------------------------------------------------------------------------
# Plot 4: Regime distribution over time
# ---------------------------------------------------------------------------

def plot_regime_distribution(
    df: pd.DataFrame,
    outputs_plots: Path = OUTPUTS_PLOTS,
    outputs_models: Path = OUTPUTS_MODELS,
) -> Path:
    """Stacked bar chart showing regime distribution (High/Medium/Low) per year.

    Args:
        df: DataFrame with 'regime' column and DatetimeIndex.
        outputs_plots: Directory to save PNG into.
        outputs_models: Not used; included for consistent API.

    Returns:
        Path to saved PNG file.
    """
    outputs_plots.mkdir(parents=True, exist_ok=True)
    out_path = outputs_plots / "regime_distribution.png"

    df = df.copy()
    df["year"] = df.index.year
    pivot = df.groupby(["year", "regime"]).size().unstack(fill_value=0)

    # Ensure all regimes present as columns
    for r in REGIME_ORDER:
        if r not in pivot.columns:
            pivot[r] = 0
    pivot = pivot[REGIME_ORDER]

    totals = pivot.sum(axis=1)
    pivot_pct = pivot.div(totals, axis=0) * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(pivot))
    years = pivot.index.tolist()

    for regime in REGIME_ORDER:
        heights = pivot_pct[regime].values
        bars = ax.bar(
            years,
            heights,
            bottom=bottom,
            color=REGIME_COLORS[regime],
            label=f"{regime.capitalize()} regime",
        )
        for bar, h, b in zip(bars, heights, bottom):
            if h >= 5:  # Only label if slice is ≥5%
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    b + h / 2,
                    f"{h:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    fontweight="bold",
                )
        bottom += heights

    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Share of PTUs (%)", fontsize=11)
    ax.set_title("Market Regime Distribution by Year", fontsize=13)
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years])
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
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
    css_paths = plot_css_vs_affr(df_main)
    logger.info("CSS vs aFRR: %d plots saved.", len(css_paths))
    weekly_paths = plot_2026_forecast(df_main)
    logger.info("February 2026 weekly forecast plots: %s", weekly_paths)
    plot_regime_distribution(df_main)

    logger.info("All plots saved to %s", OUTPUTS_PLOTS)
