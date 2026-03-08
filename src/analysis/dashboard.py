"""
Interactive Plotly dashboard for aFRR opportunity cost analysis.

Produces a self-contained HTML file (GitHub Pages-compatible) with 5 interactive charts:
1. CSS vs aFRR scatter by regime
2. Regime distribution (bar + time scatter)
3. Model diagnostics (3×3 grid)
4. Coefficient comparison with CIs
5. Forecast curves (aFRR vs DA price)

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
# Internal helpers
# ---------------------------------------------------------------------------

def _load_model(regime: str, outputs_models: Path):
    """Load pickled OLS model for a regime."""
    path = outputs_models / f"regime_{regime}_model.pkl"
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _fig_to_div(fig: go.Figure, div_id: str) -> str:
    """Convert Plotly figure to an HTML div (no full HTML wrapper, no inline JS lib)."""
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, div_id=div_id)


# ---------------------------------------------------------------------------
# Figure 1: CSS vs aFRR
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
        # Trend line
        if len(sub) > 1:
            coeffs = np.polyfit(sub["css"], y_clipped, 1)
            x_range = np.linspace(sub["css"].min(), sub["css"].max(), 200)
            fig.add_trace(go.Scatter(
                x=x_range,
                y=np.polyval(coeffs, x_range),
                mode="lines",
                name=f"{regime.capitalize()} trend",
                line=dict(color=REGIME_COLORS[regime], width=2, dash="solid"),
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
# Figure 2: Regime distribution
# ---------------------------------------------------------------------------

def _fig_regime_distribution(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.25, 0.75],
        subplot_titles=["Regime Counts", "Regime Over Time"],
    )

    # Bar chart
    counts = df["regime"].value_counts().reindex(REGIME_ORDER)
    total = counts.sum()
    fig.add_trace(go.Bar(
        x=REGIME_ORDER,
        y=counts.values,
        marker_color=[REGIME_COLORS[r] for r in REGIME_ORDER],
        text=[f"{c:,} ({100*c/total:.1f}%)" for c in counts.values],
        textposition="outside",
        showlegend=False,
    ), row=1, col=1)

    # Time scatter
    regime_num = df["regime"].map({"low": 0, "medium": 1, "high": 2})
    idx = df.index
    if hasattr(idx, "tz") and idx.tz is not None:
        idx = idx.tz_localize(None)

    for regime in REGIME_ORDER:
        mask = df["regime"] == regime
        sub_idx = idx[mask]
        fig.add_trace(go.Scatter(
            x=sub_idx,
            y=regime_num[mask],
            mode="markers",
            name=regime.capitalize(),
            marker=dict(color=REGIME_COLORS[regime], size=3, opacity=0.6),
        ), row=1, col=2)

    fig.update_yaxes(
        tickvals=[0, 1, 2],
        ticktext=["Low", "Medium", "High"],
        row=1, col=2,
    )
    fig.update_layout(title="Market Regime Distribution", height=450)
    return fig


# ---------------------------------------------------------------------------
# Figure 3: Model diagnostics
# ---------------------------------------------------------------------------

def _fig_model_diagnostics(outputs_models: Path) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=[
            f"{regime.capitalize()} — {diag}"
            for diag in ["Residuals vs Fitted", "Normal Q-Q", "Scale-Location"]
            for regime in REGIME_ORDER
        ],
        # plotly make_subplots: row-major order for subplot_titles
        # titles should be: row 0 across cols, then row 1 across cols, etc.
    )
    # Re-order: we want row=diag, col=regime
    # subplot_titles order is row-major, so let's just rely on add_trace positions

    row_labels = ["Residuals vs Fitted", "Normal Q-Q", "Scale-Location"]

    for col_idx, regime in enumerate(REGIME_ORDER, start=1):
        model = _load_model(regime, outputs_models)
        resid = model.resid.values
        fitted = model.fittedvalues.values
        std_resid = (resid - resid.mean()) / (resid.std() + 1e-9)
        color = REGIME_COLORS[regime]

        # Row 1: Residuals vs Fitted
        fig.add_trace(go.Scatter(
            x=fitted, y=resid, mode="markers",
            marker=dict(color=color, size=3, opacity=0.4),
            name=f"{regime}", showlegend=False,
        ), row=1, col=col_idx)
        fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=col_idx)

        # Row 2: Q-Q
        (osm, osr), (slope, intercept, _) = scipy.stats.probplot(resid, fit=True)
        fig.add_trace(go.Scatter(
            x=osm, y=osr, mode="markers",
            marker=dict(color=color, size=3, opacity=0.4),
            name=f"{regime}", showlegend=False,
        ), row=2, col=col_idx)
        x_ref = np.array([osm[0], osm[-1]])
        fig.add_trace(go.Scatter(
            x=x_ref, y=slope * x_ref + intercept,
            mode="lines", line=dict(color="black", width=1.5),
            showlegend=False,
        ), row=2, col=col_idx)

        # Row 3: Scale-Location
        sqrt_abs = np.sqrt(np.abs(std_resid))
        sort_idx = np.argsort(fitted)
        smoothed = lowess(sqrt_abs[sort_idx], fitted[sort_idx], frac=0.3)
        fig.add_trace(go.Scatter(
            x=fitted, y=sqrt_abs, mode="markers",
            marker=dict(color=color, size=3, opacity=0.4),
            name=f"{regime}", showlegend=False,
        ), row=3, col=col_idx)
        fig.add_trace(go.Scatter(
            x=smoothed[:, 0], y=smoothed[:, 1],
            mode="lines", line=dict(color="red", width=1.5),
            showlegend=False,
        ), row=3, col=col_idx)

    fig.update_layout(
        title="OLS Model Diagnostics by Regime",
        height=900,
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 4: Coefficient comparison
# ---------------------------------------------------------------------------

def _fig_coefficient_comparison(outputs_models: Path) -> go.Figure:
    predictors = ["css", "da_price_eur_mwh", "ccgt_generation_mw"]
    predictor_labels = ["CSS", "DA Price", "CCGT Gen (×1000 MW)"]

    fig = go.Figure()
    for regime in REGIME_ORDER:
        model = _load_model(regime, outputs_models)
        p = model.params
        ci = model.conf_int()

        values = []
        errors_minus = []
        errors_plus = []
        for pred in predictors:
            scale = 1000.0 if pred == "ccgt_generation_mw" else 1.0
            val = p[pred] * scale
            lo = ci.loc[pred, 0] * scale
            hi = ci.loc[pred, 1] * scale
            values.append(val)
            errors_minus.append(val - lo)
            errors_plus.append(hi - val)

        fig.add_trace(go.Bar(
            name=f"{regime.capitalize()}",
            x=predictor_labels,
            y=values,
            error_y=dict(
                type="data",
                symmetric=False,
                array=errors_plus,
                arrayminus=errors_minus,
            ),
            marker_color=REGIME_COLORS[regime],
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="black")
    fig.update_layout(
        title="OLS Coefficient Comparison by Market Regime (with 95% CI)",
        yaxis_title="Coefficient Value",
        barmode="group",
        legend=dict(orientation="h", y=-0.15),
        height=500,
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 5: Forecast curves
# ---------------------------------------------------------------------------

def _fig_forecast_curves(outputs_models: Path) -> go.Figure:
    metadata = _load_metadata()
    da_range = np.linspace(FORECAST_DA_MIN, FORECAST_DA_MAX, FORECAST_DA_STEPS)

    fig = go.Figure()
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
        fig.add_trace(go.Scatter(
            x=da_range, y=prices,
            mode="lines",
            name=f"{regime.capitalize()} regime",
            line=dict(color=REGIME_COLORS[regime], width=2.5),
        ))

    fig.add_annotation(
        text=(
            f"Gas: {FORECAST_GAS_PRICE:.0f} €/MWh<br>"
            f"ETS: {FORECAST_ETS_PRICE:.0f} €/tCO₂<br>"
            f"CCGT: regime mean"
        ),
        xref="paper", yref="paper",
        x=0.99, y=0.97,
        showarrow=False,
        align="right",
        bgcolor="lightyellow",
        bordercolor="gray",
        borderwidth=1,
        font=dict(size=11),
    )
    fig.update_layout(
        title="Forecast: aFRR-up Opportunity Cost vs DA Price",
        xaxis_title="DA Price Forecast (€/MWh)",
        yaxis_title="Estimated aFRR-up Price (€/MW)",
        legend=dict(orientation="h", y=-0.15),
        height=500,
    )
    return fig


# ---------------------------------------------------------------------------
# HTML wrapper
# ---------------------------------------------------------------------------

def _html_wrapper(divs: list[tuple[str, str]]) -> str:
    """Build a full HTML page wrapping the Plotly divs.

    Args:
        divs: List of (title, html_div) tuples.

    Returns:
        Full HTML string.
    """
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
    h1 {{
      color: #1a237e;
      border-bottom: 3px solid #1a237e;
      padding-bottom: 10px;
    }}
    h2 {{
      color: #283593;
      margin-top: 0;
    }}
    .card {{
      background: white;
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
      padding: 24px;
      margin-bottom: 28px;
    }}
    .subtitle {{
      color: #666;
      font-size: 14px;
      margin-top: -8px;
      margin-bottom: 24px;
    }}
    footer {{
      margin-top: 40px;
      font-size: 12px;
      color: #9e9e9e;
      text-align: center;
      border-top: 1px solid #ddd;
      padding-top: 16px;
    }}
  </style>
</head>
<body>
  <h1>aFRR-up Opportunity Cost Dashboard</h1>
  <p class="subtitle">Gas peaker plant (CCGT) aFRR spin-up participation — DE-LU 2021</p>

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
        outputs_models: Directory containing regime pkl files and metadata JSON.

    Returns:
        Path to saved HTML file.
    """
    outputs_reports.mkdir(parents=True, exist_ok=True)
    out_path = outputs_reports / "dashboard.html"

    charts = [
        ("1. CSS vs aFRR-up Prices by Regime", _fig_css_vs_affr(df)),
        ("2. Market Regime Distribution", _fig_regime_distribution(df)),
        ("3. OLS Model Diagnostics", _fig_model_diagnostics(outputs_models)),
        ("4. Coefficient Comparison", _fig_coefficient_comparison(outputs_models)),
        ("5. Forecast Curves", _fig_forecast_curves(outputs_models)),
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
