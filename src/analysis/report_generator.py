"""
HTML report generator for aFRR opportunity cost analysis.

Narrative structure (5 sections):
1. Definitions — CSS formula, regime bands, aFRR market description
2. Periods / Model Fitting — per-year OLS rationale + CSS vs aFRR scatter PNG
3. Insights from Fitted Models — regime behaviour, 2022 anomaly + historical plot PNG
4. Forecasts — Jan/Feb 2026 weekly forecast PNGs (W1–W5 Jan, W1–W4 Feb)
5. Insights from Forecasts — BESS gap explanation, trend fidelity notes

Usage:
    python src/analysis/report_generator.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.utils.config import (
    DATA_PROCESSED,
    OUTPUTS_MODELS,
    OUTPUTS_PLOTS,
    OUTPUTS_REPORTS,
)
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

REGIME_ORDER: list[str] = ["high", "medium", "low"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_metadata(outputs_models: Path) -> dict:
    path = outputs_models / "regime_metadata.json"
    with open(path) as fh:
        return json.load(fh)


def _img(src: str, alt: str = "", width: str = "100%") -> str:
    return f'<img src="{src}" alt="{alt}" style="width:{width};max-width:900px;display:block;margin:16px auto;border-radius:4px;box-shadow:0 2px 6px rgba(0,0,0,0.15);">'


# ---------------------------------------------------------------------------
# CSS styles
# ---------------------------------------------------------------------------

def _css_styles() -> str:
    return """
    <style>
      body { font-family: Georgia, 'Times New Roman', serif; max-width: 1000px;
             margin: 40px auto; padding: 0 28px; background: #fafafa; color: #222; line-height: 1.7; }
      h1 { color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 10px;
           font-size: 1.7rem; }
      h2 { color: #283593; margin-top: 48px; border-left: 5px solid #3949ab;
           padding-left: 14px; font-size: 1.25rem; }
      h3 { color: #37474f; margin-top: 24px; font-size: 1.05rem; }
      p  { margin: 10px 0; }
      code { background: #eceff1; padding: 2px 5px; border-radius: 3px;
             font-family: 'Courier New', monospace; font-size: 0.92em; }
      .callout { background: #e8eaf6; border-left: 5px solid #3949ab;
                 padding: 14px 18px; border-radius: 4px; margin: 18px 0; }
      .warn    { background: #fff8e1; border-left: 5px solid #f9a825;
                 padding: 12px 16px; border-radius: 4px; margin: 12px 0; font-size: 0.95em; }
      table { border-collapse: collapse; width: 100%; margin: 16px 0;
              background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); font-size: 0.93em; }
      th { background: #37474f; color: white; padding: 9px 13px; text-align: left; }
      td { padding: 7px 13px; border-bottom: 1px solid #e0e0e0; }
      tr:hover { background: #f5f5f5; }
      .regime-high   { color: #1565C0; font-weight: 600; }
      .regime-medium { color: #E65100; font-weight: 600; }
      .regime-low    { color: #B71C1C; font-weight: 600; }
      footer { margin-top: 60px; font-size: 12px; color: #9e9e9e;
               border-top: 1px solid #ddd; padding-top: 16px; }
    </style>
    """


# ---------------------------------------------------------------------------
# Section 1 — Definitions
# ---------------------------------------------------------------------------

def _build_definitions() -> str:
    return """
    <h2>1. Definitions</h2>

    <h3>Market Regimes</h3>
    <p>
      Each hour can be classified into three market
      regimes based on total gas generation in the DE-LU bidding zone, using fixed MW
      thresholds derived from K-Means clustering of the 2020–2026 historical generation profile:
    </p>
    <table>
      <tr><th>Regime</th><th>gas Generation</th><th>Reserve Supply</th></tr>
      <tr>
        <td class="regime-low">Low</td>
        <td>&lt; 5,858 MW</td>
        <td>Scarce — few plants online, mostly efficient CCGT generators</td>
      </tr>
      <tr>
        <td class="regime-medium">Medium</td>
        <td>5,858 – 10,546 MW</td>
        <td>Balanced — moderate number of providers</td>
      </tr>
      <tr>
        <td class="regime-high">High</td>
        <td>≥ 10,546 MW</td>
        <td>Abundant — many plants online, including less efficient gas powered reciprocating engines</td>
      </tr>
    </table>
    <p>
      Thresholds are midpoints between adjacent K-Means centroids
      (low ≈ 3,776 MW; medium ≈ 7,941 MW; high ≈ 13,150 MW).
    </p>

    <h3>Clean Spark Spread (CSS)</h3>
    <p>
      The CSS measures the profitability of selling electricity in the day-ahead (DA) market
      rather than holding back capacity for the balancing reserve:
    </p>
    <div class="callout">
      <code>CSS = DA_Price − Gas_Price / η − ETS_Price × intensity / η</code>
      <br>
      <small>η = 0.50 (thermal efficiency), intensity = 0.202 tCO₂/MWh<sub>th</sub></small>
    </div>
    <p>
      A high CSS means spot-market generation is highly profitable, creating a strong
      opportunity cost for committing capacity to the reserve.
    </p>

    <h3>aFRR-up Market</h3>
    <p>
      Automated Frequency Restoration Reserve (aFRR) is a balancing capacity product procured
      day-ahead by the TSO (TenneT in DE-LU). Generators submit hourly bids (€/MW); the
      accepted bids are settled pay-as-bid. The reported hourly aFRR-up price is the
      capacity-weighted average of accepted upward bids in that hour.
    </p>
    <h3>Role of R² in OLS regression</h3>
    <p>
      An R² of 0.287 means the regressors explain 28.7% of the variance in Y; the remaining 71.3% is unexplained.
    </p>
    
    """


# ---------------------------------------------------------------------------
# Section 2 — Periods / Model Fitting
# ---------------------------------------------------------------------------

def _build_model_fitting(metadata: dict, plots_rel: str) -> str:
    # Collect per-year R² for table
    years = sorted(
        k for k in metadata if k.isdigit() and k not in ("high", "medium", "low")
    )
    rows = []
    for year in years:
        yr_meta = metadata[year]
        cells = f"<td>{year}</td>"
        for regime in REGIME_ORDER:
            r2 = yr_meta.get(regime, {}).get("rsquared", float("nan"))
            if r2 != r2:  # nan check
                cells += "<td>—</td>"
            else:
                cls = "style='color:#2e7d32;font-weight:600;'" if r2 >= 0.5 else "style='color:#e65100;'"
                cells += f"<td {cls}>{r2:.3f}</td>"
        rows.append(f"<tr>{cells}</tr>")

    rows_html = "".join(rows)

    regime_imgs = "".join(
        _img(
            f"{plots_rel}/css_vs_affr_{regime}.png",
            f"CSS vs aFRR — {regime.capitalize()} regime by year",
        )
        for regime in REGIME_ORDER
    )

    return f"""
    <h2>2. Periods &amp; Model Fitting</h2>
    <p>
      For each year from 2020 to 2025, and for each market regime, we fit a separate
      Ordinary Least Squares (OLS) model of the form:
    </p>
    <div class="callout">
      <code>aFRR_Price = β₀ + β₁ · CSS</code>
    </div>
    <p>
      The per-year approach is motivated by
      the structural breaks visible in German energy markets: COVID-19 demand suppression
      (2020–2021), the energy crisis triggered by Russia's invasion of Ukraine (2022–2023),
      and the subsequent normalisation (2024–2025).
      Fitting separate annual models allows β₁ to vary across these regimes without
      imposing a single structural relationship on fundamentally different market states.
    </p>
    <p>
      The regime label already captures gas generation levels, so CSS is chosen as sole
      predictor. Including DA price separately would double-count the information
      already embedded in CSS.
    </p>

    <h3>R² by Year and Regime</h3>
    <table>
      <tr><th>Year</th>
          <th class="regime-high">High</th>
          <th class="regime-medium">Medium</th>
          <th class="regime-low">Low</th></tr>
      {rows_html}
    </table>

    <h3>CSS vs aFRR Scatter — by Regime</h3>
    <p>
      Each chart below shows one market regime across all years, with one subplot per year
      and a linear trend line. This allows direct comparison of how the CSS–aFRR
      relationship evolves within a single regime from year to year.
    </p>
    {regime_imgs}
    """


# ---------------------------------------------------------------------------
# Section 3 — Insights from Fitted Models
# ---------------------------------------------------------------------------

def _build_model_insights(metadata: dict, plots_rel: str) -> str:
    # Pull flat compat keys for a representative year summary
    high_meta   = metadata.get("high", {})
    medium_meta = metadata.get("medium", {})
    low_meta    = metadata.get("low", {})

    def fmt_beta(m: dict) -> str:
        b = m.get("beta_1_css", float("nan"))
        return f"{b:+.4f}" if b == b else "—"

    return f"""
    <h2>3. Insights from Fitted Models</h2>

    <h3>Low Regime — No explaining power with CSS alone</h3>
    <p>
      The R² is strikingly weak throughout, rarely exceeding 0.019. When only efficient generators are providing balancing services their bids are independent of CSS.
    </p>
    <p>
      Low regime aFRR is concomitant with negative CSS. Understandable; only efficient Generators can run during low CSS periods(their η > 0.50).
      During medium and high regimes (where, η ~ 0.50), the modelled CSS begins reflecting their bidding.
    </p>
   
    <h3>High Regimes — Decisions based on CSS no more a coin flip</h3>
    <p>
      High regime consistently produces the strongest R² (excluding 2021 and 2022). 
      This suggests the model's predictors have the most explanatory power during high regime 
      — likely because high-regime periods exhibit competition dynamics.
    </p>

    <div class="warn">
      ⚠ 2021 model coefficients are anomalous due to the energy crisis from unusually low gas storage levels entering winter and post COVID demand recovery.
      Similarly for 2022 model.
    </div>

    {_img(f"{plots_rel}/historical_afrr_prices.png", "Historical aFRR prices coloured by regime")}
    """


# ---------------------------------------------------------------------------
# Section 4 — Forecasts
# ---------------------------------------------------------------------------

def _build_forecasts(plots_rel: str) -> str:
    jan_imgs = "".join(
        _img(f"{plots_rel}/forecast_curves_jan_w{w}.png", f"January 2026 Week {w} forecast")
        for w in range(1, 6)
    )
    feb_imgs = "".join(
        _img(f"{plots_rel}/forecast_curves_feb_w{w}.png", f"February 2026 Week {w} forecast")
        for w in range(1, 5)
    )

    return f"""
    <h2>4. Forecasts</h2>
    <p>
      For January and February 2026, we use the known gas generation (from ENTSO-E
      transparency data) to assign each PTU to a market regime, then apply the 2025
      annual model coefficients to forecast the aFRR-up capacity price as a function of
      the prevailing gas price and DA price. Actual aFRR prices (where available) are
      plotted alongside the forecasts for comparison.
    </p>
    <p>
      Each chart covers one calendar week.
    </p>

    <h3>January 2026</h3>
    {jan_imgs}

    <h3>February 2026</h3>
    {feb_imgs}
    """


# ---------------------------------------------------------------------------
# Section 5 — Insights from Forecasts
# ---------------------------------------------------------------------------

def _build_forecast_insights() -> str:
    return """
    <h2>5. Insights from Forecasts</h2>

    <h3>Forecasted prices are generally higher than actuals</h3>
    <p>
      Across both January and February 2026, the model-derived forecasts tend to sit
      above the observed aFRR-up prices. This is an expected structural bias: the models
      are calibrated on historical data when Battery Energy Storage Systems (BESS) had a
      smaller footprint in the aFRR market. Since 2024, BESS capacity participating in
      aFRR has grown substantially. BESS operators face near-zero marginal costs and bid
      aggressively at low prices, compressing the market clearing price independently of
      CSS. The models cannot capture this effect because BESS participation data was not
      included in the training features.
    </p>

    <h3>The model captures the price trend</h3>
    <p>
      Despite the systematic upward bias, the model correctly reproduces the intra-week
      shape of aFRR prices — the rise and fall of prices in response to changing gas
      forward curves and DA price levels. The level shift introduced by BESS could in principle be
      corrected with a BESS-participation adjustment term.
    </p>
    <p>
      This agreement can be explained as follows: The PTUs in the months of Jan and Feb are gas dominant i.e. high regimes, which is expected when renewables are low in winter months. 
      We observed high R² for this regime.
    </p>

    <h3>March 2026 Analysis</h3>
    <p>
      aFRR capacity prices for March 2026 are not yet available from Regelleistung.net
      at the time of writing. Forecast charts for March
      should be interesting, as gas prices will be high but the regimes might be predominantly low to mid.
    </p>
    """


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def generate_report(
    df: pd.DataFrame,
    outputs_reports: Path = OUTPUTS_REPORTS,
    outputs_models: Path = OUTPUTS_MODELS,
    outputs_plots: Path = OUTPUTS_PLOTS,
) -> Path:
    """Generate a narrative HTML report.

    Args:
        df: DataFrame with regime labels and market data columns (unused in new template
            beyond passing to section builders; retained for API compatibility).
        outputs_reports: Directory to save HTML report into.
        outputs_models: Directory containing regime_metadata.json.
        outputs_plots: Directory containing generated PNG plots.

    Returns:
        Path to saved HTML file.
    """
    outputs_reports.mkdir(parents=True, exist_ok=True)
    out_path = outputs_reports / "market_regime_report.html"

    metadata = _load_metadata(outputs_models)

    # Relative path from the HTML file's directory (outputs/reports/) to outputs/plots/
    import os
    plots_rel = os.path.relpath(outputs_plots, outputs_reports).replace("\\", "/")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>aFRR-up Opportunity Cost — Market Regime Report</title>
  {_css_styles()}
</head>
<body>
  <h1>aFRR-up Opportunity Cost: Market Regime Report</h1>
  <p style="color:#666;font-size:0.92em;">
    Gas peaker plant (gas) — DE-LU bidding zone, 2020–2026 |
    Model: <code>aFRR_Price = β₀ + β₁·CSS</code> (fitted per year × regime)
  </p>

  {_build_definitions()}
  {_build_model_fitting(metadata, plots_rel)}
  {_build_model_insights(metadata, plots_rel)}
  {_build_forecasts(plots_rel)}
  {_build_forecast_insights()}

  <footer>
    Generated by <code>src/analysis/report_generator.py</code> &mdash;
    aFRR Opportunity Cost Calculator &mdash; DE-LU 2020–2026
  </footer>
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    logger.info("Saved report: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    csv_path = DATA_PROCESSED / "market_regimes.csv"
    logger.info("Loading %s", csv_path)
    df_main = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    path = generate_report(df_main)
    print(f"Report saved to: {path}")
