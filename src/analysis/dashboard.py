"""
Interactive Plotly dashboard for aFRR opportunity cost analysis.

Produces a self-contained HTML file with 12 interactive charts:
1.  Historical aFRR-up capacity prices over time (coloured by regime)
2.  CSS vs aFRR scatter by year (per-year subplots)
3.  January 2026 Week 1 forecast vs actual aFRR prices
4.  January 2026 Week 2 forecast vs actual aFRR prices
5.  January 2026 Week 3 forecast vs actual aFRR prices
6.  January 2026 Week 4 forecast vs actual aFRR prices
7.  January 2026 Week 5 forecast vs actual aFRR prices
8.  February 2026 Week 1 forecast vs actual aFRR prices
9.  February 2026 Week 2 forecast vs actual aFRR prices
10. February 2026 Week 3 forecast vs actual aFRR prices
11. February 2026 Week 4 forecast vs actual aFRR prices
12. Regime distribution by year (stacked bar)

Usage:
    python src/analysis/dashboard.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.utils.config import (
    CARBON_INTENSITY,
    DATA_PROCESSED,
    EFFICIENCY,
    OUTPUTS_MODELS,
    OUTPUTS_REPORTS,
)
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

REGIME_ORDER: list[str] = ["high", "medium", "low"]
REGIME_COLORS: dict[str, str] = {
    "high": "#2196F3",
    "medium": "#FF9800",
    "low": "#F44336",
}
PLOTLY_CDN = "https://cdn.plot.ly/plotly-latest.min.js"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fig_to_div(fig: go.Figure, div_id: str) -> str:
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, div_id=div_id)


# ---------------------------------------------------------------------------
# Figure 1: Historical aFRR prices
# ---------------------------------------------------------------------------

def _fig_historical_afrr(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    idx = df.index
    if hasattr(idx, "tz") and idx.tz is not None:
        idx = idx.tz_localize(None)

    for regime in REGIME_ORDER:
        mask = df["regime"] == regime
        fig.add_trace(go.Scatter(
            x=idx[mask],
            y=df.loc[mask, "affr_price_eur_mw"],
            mode="markers",
            name=f"{regime.capitalize()} regime",
            marker=dict(color=REGIME_COLORS[regime], size=3, opacity=0.6),
        ))

    fig.update_layout(
        title="Historical aFRR-up Capacity Prices by Market Regime",
        xaxis_title="Date",
        yaxis_title="aFRR-up Price (€/MW)",
        legend=dict(orientation="h", y=-0.15),
        height=450,
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 2: CSS vs aFRR scatter — per-year subplots
# ---------------------------------------------------------------------------

def _fig_css_vs_affr(df: pd.DataFrame) -> go.Figure:
    years = sorted(df.index.year.unique())
    n_years = len(years)
    ncols = min(3, n_years)
    nrows = (n_years + ncols - 1) // ncols

    y_col = "affr_price_eur_mw"
    y_clip = df[y_col].quantile(0.99)

    subplot_titles = [f"Year {y}" for y in years]
    # Pad to fill grid
    while len(subplot_titles) < nrows * ncols:
        subplot_titles.append("")

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=subplot_titles,
        shared_yaxes=False,
    )

    shown_regimes: set[str] = set()

    for i, year in enumerate(years):
        row_idx = i // ncols + 1
        col_idx = i % ncols + 1
        year_df = df[df.index.year == year]

        for regime in REGIME_ORDER:
            sub = year_df[year_df["regime"] == regime]
            y_clipped = sub[y_col].clip(upper=y_clip)
            show_legend = regime not in shown_regimes
            fig.add_trace(go.Scatter(
                x=sub["css"],
                y=y_clipped,
                mode="markers",
                name=f"{regime.capitalize()}",
                legendgroup=regime,
                marker=dict(color=REGIME_COLORS[regime], size=4, opacity=0.4),
                showlegend=show_legend,
            ), row=row_idx, col=col_idx)
            if show_legend:
                shown_regimes.add(regime)

            if len(sub) > 1:
                coeffs = np.polyfit(sub["css"], y_clipped, 1)
                x_range = np.linspace(sub["css"].min(), sub["css"].max(), 200)
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=np.polyval(coeffs, x_range),
                    mode="lines",
                    name=f"{regime.capitalize()} trend",
                    legendgroup=regime,
                    line=dict(color=REGIME_COLORS[regime], width=1.5),
                    showlegend=False,
                ), row=row_idx, col=col_idx)

    fig.update_layout(
        title="CSS vs aFRR-up Prices by Year and Market Regime",
        height=400 * nrows,
        legend=dict(orientation="h", y=-0.08),
    )
    return fig


# ---------------------------------------------------------------------------
# Figures 3–11: January & February 2026 forecast vs actual — one per week
# ---------------------------------------------------------------------------

_MONTH_WEEK_RANGES: dict[int, list[tuple[int, int]]] = {
    1: [(1, 7), (8, 14), (15, 21), (22, 28), (29, 31)],
    2: [(1, 7), (8, 14), (15, 21), (22, 28)],
}
_MONTH_NAME: dict[int, str] = {1: "January", 2: "February"}


def _fig_forecast_week(
    month: int,
    week: int,
    data_processed: Path = DATA_PROCESSED,
    outputs_models: Path = OUTPUTS_MODELS,
) -> go.Figure:
    """Plotly figure for one week of a 2026 month — forecast vs actual.

    Args:
        month: Month number (1=January, 2=February).
        week: Week number within the month (1-based).
        data_processed: Directory containing combined_dataset.csv.
        outputs_models: Directory with per-year pkl files.

    Returns:
        go.Figure with actual + per-year model forecast traces.
    """
    from src.models.regime_classifier import assign_regimes

    d_start, d_end = _MONTH_WEEK_RANGES[month][week - 1]
    month_name = _MONTH_NAME[month]
    title = (
        f"{month_name} 2026 Week {week} — aFRR Forecast vs Actual "
        f"({month_name[:3]} {d_start}–{d_end})"
    )

    combined_csv = data_processed / "combined_dataset.csv"
    df_all = pd.read_csv(combined_csv, index_col=0, parse_dates=True)
    if df_all.index.tz is None:
        df_all.index = df_all.index.tz_localize("UTC")

    df_month = df_all[
        (df_all.index.year == 2026) & (df_all.index.month == month)
    ].copy()

    fig = go.Figure()

    if not df_month.empty:
        df_month = assign_regimes(df_month)
        required = ["da_price_eur_mwh", "gas_price_eur_mwh_th", "eu_ets_price_eur_tco2",
                    "affr_price_eur_mw", "css"]
        df_month = df_month.dropna(subset=required)

    mask = (df_month.index.day >= d_start) & (df_month.index.day <= d_end)
    df_week = df_month[mask]

    if df_week.empty:
        fig.add_annotation(text=f"No {month_name} 2026 Wk{week} data available",
                           x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        fig.update_layout(title=title, height=450)
        return fig

    # Discover available years (exclude 2026 — no look-ahead)
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

    def _predict_row(da_price, gas_price, ets_price, regime, year):
        carbon_cost = ets_price * CARBON_INTENSITY / EFFICIENCY
        css = da_price - (gas_price / EFFICIENCY) - carbon_cost
        model_path = outputs_models / f"regime_{regime}_{year}_model.pkl"
        with open(model_path, "rb") as fh:
            model = pickle.load(fh)
        fv = pd.DataFrame([[1.0, css]], columns=["const", "css"])
        return float(model.predict(fv)[0])

    idx = df_week.index
    if hasattr(idx, "tz") and idx.tz is not None:
        idx = idx.tz_localize(None)

    fig.add_trace(go.Scatter(
        x=idx,
        y=df_week["affr_price_eur_mw"],
        mode="lines",
        name="Actual aFRR",
        line=dict(color="black", width=2),
    ))

    palette = [
        "#E53935", "#8E24AA", "#1E88E5", "#00ACC1",
        "#43A047", "#FB8C00", "#6D4C41", "#546E7A",
    ]

    for i, year in enumerate(available_years):
        preds = []
        for _, row in df_week.iterrows():
            try:
                p = _predict_row(
                    row["da_price_eur_mwh"],
                    row["gas_price_eur_mwh_th"],
                    row["eu_ets_price_eur_tco2"],
                    row["regime"],
                    year,
                )
            except Exception:
                p = None
            preds.append(p)

        fig.add_trace(go.Scatter(
            x=idx,
            y=preds,
            mode="lines",
            name=f"Model {year}",
            line=dict(color=palette[i % len(palette)], width=1.5),
            opacity=0.8,
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="aFRR-up Price (€/MW)",
        legend=dict(orientation="h", y=-0.15),
        height=450,
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 4: Regime distribution
# ---------------------------------------------------------------------------

def _fig_regime_distribution(df: pd.DataFrame) -> go.Figure:
    df = df.copy()
    df["year"] = df.index.year
    pivot = df.groupby(["year", "regime"]).size().unstack(fill_value=0)

    for r in REGIME_ORDER:
        if r not in pivot.columns:
            pivot[r] = 0
    pivot = pivot[REGIME_ORDER]

    totals = pivot.sum(axis=1)
    pivot_pct = pivot.div(totals, axis=0) * 100
    years = [str(y) for y in pivot.index.tolist()]

    fig = go.Figure()
    for regime in REGIME_ORDER:
        heights = pivot_pct[regime].values
        fig.add_trace(go.Bar(
            name=f"{regime.capitalize()} regime",
            x=years,
            y=heights,
            marker_color=REGIME_COLORS[regime],
            text=[f"{h:.0f}%" if h >= 5 else "" for h in heights],
            textposition="inside",
        ))

    fig.update_layout(
        barmode="stack",
        title="Market Regime Distribution by Year",
        xaxis_title="Year",
        yaxis_title="Share of PTUs (%)",
        legend=dict(orientation="h", y=-0.15),
        height=450,
        yaxis_range=[0, 105],
    )
    return fig


# ---------------------------------------------------------------------------
# HTML wrapper
# ---------------------------------------------------------------------------

def _html_wrapper(divs: list[tuple[str, str]]) -> str:
    cards = ""
    for title, div in divs:
        cards += f"""
        <div class="card">
          <h2>{title}</h2>
          {div}
        </div>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>aFRR Opportunity Cost Dashboard</title>
  <script src="{PLOTLY_CDN}"></script>
  <style>
    body {{
      font-family: Arial, sans-serif;
      max-width: 1400px;
      margin: 0 auto;
      padding: 20px;
      background: #f0f2f5;
      color: #333;
    }}
    h1 {{ color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 10px; }}
    h2 {{ color: #283593; margin-top: 0; }}
    .card {{
      background: white;
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
      padding: 24px;
      margin-bottom: 28px;
    }}
    .subtitle {{ color: #666; font-size: 14px; margin-top: -8px; margin-bottom: 24px; }}
    footer {{
      margin-top: 40px; font-size: 12px; color: #9e9e9e;
      text-align: center; border-top: 1px solid #ddd; padding-top: 16px;
    }}
  </style>
</head>
<body>
  <h1>aFRR-up Opportunity Cost Dashboard</h1>
  <p class="subtitle">Gas peaker plant (CCGT) — DE-LU 2020–2026 | Model: aFRR = β₀ + β₁·CSS per regime × year</p>

  {cards}

  <footer>
    Generated by <code>src/analysis/dashboard.py</code> &mdash;
    aFRR Opportunity Cost Calculator &mdash; DE-LU 2020–2026
  </footer>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def generate_dashboard(
    df: pd.DataFrame,
    outputs_reports: Path = OUTPUTS_REPORTS,
    outputs_models: Path = OUTPUTS_MODELS,
    data_processed: Path = DATA_PROCESSED,
) -> Path:
    """Generate a self-contained interactive Plotly HTML dashboard.

    Args:
        df: DataFrame with all market_regimes.csv columns.
        outputs_reports: Directory to save HTML dashboard into.
        outputs_models: Directory with pkl files; used for forecast figure.
        data_processed: Directory with combined_dataset.csv; used for forecast figure.

    Returns:
        Path to saved HTML file.
    """
    outputs_reports.mkdir(parents=True, exist_ok=True)
    out_path = outputs_reports / "dashboard.html"

    forecast_charts = []
    chart_num = 3
    for month, week_ranges in _MONTH_WEEK_RANGES.items():
        month_name = _MONTH_NAME[month]
        for wk in range(1, len(week_ranges) + 1):
            d_start, d_end = week_ranges[wk - 1]
            label = (
                f"{chart_num}. {month_name[:3]} 2026 Forecast vs Actual — "
                f"Week {wk} ({month_name[:3]} {d_start}–{d_end})"
            )
            forecast_charts.append(
                (label, _fig_forecast_week(month, wk, data_processed, outputs_models))
            )
            chart_num += 1

    charts = [
        ("1. Historical aFRR-up Capacity Prices", _fig_historical_afrr(df)),
        ("2. CSS vs aFRR-up Prices by Year", _fig_css_vs_affr(df)),
        *forecast_charts,
        (f"{chart_num}. Market Regime Distribution by Year", _fig_regime_distribution(df)),
    ]

    divs = [(title, _fig_to_div(fig, f"chart-{i+1}")) for i, (title, fig) in enumerate(charts)]
    html = _html_wrapper(divs)
    out_path.write_text(html, encoding="utf-8")
    logger.info("Saved dashboard: %s", out_path)
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

    path = generate_dashboard(df_main)
    print(f"Dashboard saved to: {path}")
