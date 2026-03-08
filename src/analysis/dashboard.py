"""
Interactive Plotly dashboard for aFRR opportunity cost analysis.

Produces a self-contained HTML file with 3 interactive charts:
1. Historical aFRR-up capacity prices over time (coloured by regime)
2. CSS vs aFRR scatter by regime (model evidence)
3. Forecast: aFRR price vs gas forward price, parametrised by DA scenario

Usage:
    python src/analysis/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

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
DA_SCENARIO_COLORS: list[str] = ["#1E88E5", "#43A047", "#FB8C00", "#E53935"]
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
# Figure 2: CSS vs aFRR scatter
# ---------------------------------------------------------------------------

def _fig_css_vs_affr(df: pd.DataFrame) -> go.Figure:
    y_col = "affr_price_eur_mw"
    y_clip = df[y_col].quantile(0.99)

    fig = go.Figure()
    for regime in REGIME_ORDER:
        sub = df[df["regime"] == regime]
        y_clipped = sub[y_col].clip(upper=y_clip)
        fig.add_trace(go.Scatter(
            x=sub["css"],
            y=y_clipped,
            mode="markers",
            name=f"{regime.capitalize()} (n={len(sub):,})",
            marker=dict(color=REGIME_COLORS[regime], size=4, opacity=0.5),
        ))
        if len(sub) > 1:
            coeffs = np.polyfit(sub["css"], y_clipped, 1)
            x_range = np.linspace(sub["css"].min(), sub["css"].max(), 200)
            fig.add_trace(go.Scatter(
                x=x_range,
                y=np.polyval(coeffs, x_range),
                mode="lines",
                name=f"{regime.capitalize()} trend",
                line=dict(color=REGIME_COLORS[regime], width=2),
                showlegend=False,
            ))

    fig.update_layout(
        title="CSS vs aFRR-up Prices by Market Regime",
        xaxis_title="Clean Spark Spread (€/MWh)",
        yaxis_title=f"aFRR-up Price (€/MW, clipped at 99th pctile: {y_clip:.0f})",
        yaxis_range=[0, y_clip * 1.02],
        legend=dict(orientation="h", y=-0.15),
        height=500,
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 3: Forecast curves
# ---------------------------------------------------------------------------

def _fig_forecast_curves() -> go.Figure:
    gas_range = np.linspace(FORECAST_GAS_MIN, FORECAST_GAS_MAX, FORECAST_GAS_STEPS)

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[f"{r.capitalize()} Regime" for r in REGIME_ORDER],
        shared_yaxes=True,
    )

    for col_idx, regime in enumerate(REGIME_ORDER, start=1):
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
            fig.add_trace(go.Scatter(
                x=gas_range,
                y=prices,
                mode="lines",
                name=f"DA = {da_price:.0f} €/MWh",
                line=dict(color=DA_SCENARIO_COLORS[i % len(DA_SCENARIO_COLORS)], width=2),
                showlegend=(col_idx == 1),
            ), row=1, col=col_idx)

        fig.update_xaxes(title_text="Gas Price (€/MWh_th)", row=1, col=col_idx)

    fig.update_yaxes(title_text="Est. aFRR-up Price (€/MW)", row=1, col=1)
    fig.update_layout(
        title=f"aFRR-up Opportunity Cost vs Gas Forward Price  (ETS = {FORECAST_ETS_PRICE:.0f} €/tCO₂)",
        height=500,
        legend=dict(orientation="h", y=-0.15),
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
  <p class="subtitle">Gas peaker plant (CCGT) — DE-LU 2021 | Model: aFRR = β₀ + β₁·CSS per regime</p>

  {cards}

  <footer>
    Generated by <code>src/analysis/dashboard.py</code> &mdash;
    aFRR Opportunity Cost Calculator &mdash; DE-LU 2021
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
) -> Path:
    """Generate a self-contained interactive Plotly HTML dashboard.

    Args:
        df: DataFrame with all market_regimes.csv columns.
        outputs_reports: Directory to save HTML dashboard into.
        outputs_models: Not used directly; estimate_afrr_price uses config paths.

    Returns:
        Path to saved HTML file.
    """
    outputs_reports.mkdir(parents=True, exist_ok=True)
    out_path = outputs_reports / "dashboard.html"

    charts = [
        ("1. Historical aFRR-up Capacity Prices", _fig_historical_afrr(df)),
        ("2. CSS vs aFRR-up Prices by Regime", _fig_css_vs_affr(df)),
        ("3. Opportunity Cost Forecast by Gas Forward Price", _fig_forecast_curves()),
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
