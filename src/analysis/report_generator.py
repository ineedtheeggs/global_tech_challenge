"""
HTML report generator for aFRR opportunity cost analysis.

Produces a self-contained HTML report with:
1. What the model found — CSS slope per regime with interpretation
2. Regime snapshot — CCGT band, PTU count, mean aFRR price, mean CSS, R²
3. Model reliability — brief R² summary and data notes
4. Opportunity cost scenarios — aFRR estimates across gas × DA price grid per regime

Usage:
    python src/analysis/report_generator.py
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.stattools import durbin_watson

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.models.predictions import estimate_afrr_price
from src.utils.config import (
    DATA_PROCESSED,
    FORECAST_DA_SCENARIOS,
    FORECAST_ETS_PRICE,
    OUTPUTS_MODELS,
    OUTPUTS_REPORTS,
)
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

REGIME_ORDER: list[str] = ["high", "medium", "low"]
REGIME_DESCRIPTIONS: dict[str, str] = {
    "high": "Many plants online (>75th %ile CCGT gen) — abundant reserve supply",
    "medium": "Moderate generation (25–75th %ile CCGT gen) — balanced reserve supply",
    "low": "Few plants online (<25th %ile CCGT gen) — scarce reserve supply",
}

# Gas prices (rows) and DA prices (columns) for the scenario table
SCENARIO_GAS_PRICES: list[float] = [25.0, 35.0, 45.0, 55.0, 65.0, 75.0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_all_models(outputs_models: Path) -> dict:
    models = {}
    for regime in REGIME_ORDER:
        path = outputs_models / f"regime_{regime}_model.pkl"
        with open(path, "rb") as fh:
            models[regime] = pickle.load(fh)
    return models


def _load_metadata(outputs_models: Path) -> dict:
    path = outputs_models / "regime_metadata.json"
    with open(path) as fh:
        return json.load(fh)


def _extract_stats(model, regime: str, df: pd.DataFrame, metadata: dict) -> dict:
    subset = df[df["regime"] == regime]
    css_coef = float(model.params.get("css", float("nan")))
    ci = model.conf_int()
    css_ci = (float(ci.loc["css", 0]), float(ci.loc["css", 1])) if "css" in ci.index else (float("nan"), float("nan"))
    css_pval = float(model.pvalues.get("css", float("nan")))
    dw = float(durbin_watson(model.resid))

    return {
        "regime": regime,
        "css_coef": css_coef,
        "css_ci_lo": css_ci[0],
        "css_ci_hi": css_ci[1],
        "css_pval": css_pval,
        "rsquared": float(model.rsquared),
        "nobs": int(model.nobs),
        "durbin_watson": dw,
        "mean_affr": float(subset["affr_price_eur_mw"].mean()),
        "mean_css": float(subset["css"].mean()),
        "ccgt_mean_mw": metadata[regime].get("ccgt_mean_mw", float("nan")),
    }


# ---------------------------------------------------------------------------
# CSS styles
# ---------------------------------------------------------------------------

def _css_styles() -> str:
    return """
    <style>
      body { font-family: Arial, sans-serif; max-width: 1100px; margin: 40px auto;
             padding: 0 20px; background: #f8f9fa; color: #333; }
      h1 { color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 10px; }
      h2 { color: #283593; margin-top: 40px; border-left: 4px solid #283593;
           padding-left: 12px; }
      .summary-box { background: #e8eaf6; border-left: 5px solid #3949ab;
                     padding: 16px 20px; border-radius: 4px; margin: 16px 0; }
      .note { background: #fff8e1; border-left: 5px solid #f9a825;
              padding: 12px 16px; border-radius: 4px; margin: 8px 0; font-size: 14px; }
      table { border-collapse: collapse; width: 100%; margin: 16px 0;
              background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
      th { background: #37474f; color: white; padding: 10px 14px;
           text-align: left; font-weight: 600; }
      td { padding: 8px 14px; border-bottom: 1px solid #e0e0e0; font-size: 14px; }
      tr:hover { background: #f5f5f5; }
      .regime-high   { color: #1565C0; font-weight: 600; }
      .regime-medium { color: #E65100; font-weight: 600; }
      .regime-low    { color: #B71C1C; font-weight: 600; }
      .good { color: #2e7d32; font-weight: 600; }
      .warn { color: #e65100; font-weight: 600; }
      .scenario-table th { background: #455a64; }
      .scenario-table td { text-align: center; }
      .scenario-table td:first-child { text-align: left; font-weight: 600; }
      footer { margin-top: 60px; font-size: 12px; color: #9e9e9e;
               border-top: 1px solid #ddd; padding-top: 16px; }
    </style>
    """


def _fmt_pval(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.4f}"


# ---------------------------------------------------------------------------
# Section 1: What the model found
# ---------------------------------------------------------------------------

def _build_findings(stats_list: list[dict]) -> str:
    rows = []
    for s in stats_list:
        regime = s["regime"]
        coef = s["css_coef"]
        direction = "↑" if coef > 0 else "↓"
        if abs(coef) < 0.1:
            sensitivity = "Weak"
        elif abs(coef) < 0.25:
            sensitivity = "Moderate"
        else:
            sensitivity = "Strong"

        rows.append(
            f"<tr>"
            f"<td class='regime-{regime}'>{regime.capitalize()}</td>"
            f"<td>{REGIME_DESCRIPTIONS[regime]}</td>"
            f"<td><b>{coef:+.4f}</b> {direction}</td>"
            f"<td>[{s['css_ci_lo']:.4f}, {s['css_ci_hi']:.4f}]</td>"
            f"<td>{_fmt_pval(s['css_pval'])}</td>"
            f"<td>{sensitivity}</td>"
            f"</tr>"
        )

    return f"""
    <h2>1. What the Model Found</h2>
    <div class='summary-box'>
      <p>One OLS model per market regime:
         <code>aFRR_Price = β₀ + β₁ · CSS</code>
         where CSS = DA_Price − Gas_Price/η − ETS·intensity/η.</p>
      <p>The CSS slope β₁ quantifies how sensitive aFRR bid prices are to
         plant profitability. Regime classification already captures CCGT
         generation levels, so CSS is the sole predictor.</p>
    </div>
    <table>
      <tr>
        <th>Regime</th>
        <th>Market Condition</th>
        <th>β₁ (CSS slope, €/MW per €/MWh)</th>
        <th>95% CI</th>
        <th>p-value</th>
        <th>Sensitivity</th>
      </tr>
      {''.join(rows)}
    </table>
    <p style='font-size:13px;color:#555;'>
      A positive β₁ means higher CSS (more profitable spot sales) → higher aFRR bid prices.
      The Low regime has the strongest relationship because scarce reserve supply amplifies
      the effect of plant opportunity costs.
    </p>
    """


# ---------------------------------------------------------------------------
# Section 2: Regime snapshot
# ---------------------------------------------------------------------------

def _build_regime_snapshot(stats_list: list[dict]) -> str:
    rows = []
    for s in stats_list:
        regime = s["regime"]
        r2_cls = "good" if s["rsquared"] >= 0.5 else "warn"
        rows.append(
            f"<tr>"
            f"<td class='regime-{regime}'>{regime.capitalize()}</td>"
            f"<td>{s['nobs']:,}</td>"
            f"<td>{s['ccgt_mean_mw']:.0f} MW</td>"
            f"<td>{s['mean_affr']:.2f}</td>"
            f"<td>{s['mean_css']:.2f}</td>"
            f"<td class='{r2_cls}'>{s['rsquared']:.3f}</td>"
            f"</tr>"
        )

    return f"""
    <h2>2. Regime Snapshot</h2>
    <table>
      <tr>
        <th>Regime</th>
        <th>PTUs</th>
        <th>Mean CCGT Gen</th>
        <th>Mean aFRR Price (€/MW)</th>
        <th>Mean CSS (€/MWh)</th>
        <th>R²</th>
      </tr>
      {''.join(rows)}
    </table>
    """


# ---------------------------------------------------------------------------
# Section 3: Model reliability
# ---------------------------------------------------------------------------

def _build_reliability(stats_list: list[dict]) -> str:
    items = []
    for s in stats_list:
        r2_cls = "good" if s["rsquared"] >= 0.5 else "warn"
        items.append(
            f"<li><b class='regime-{s['regime']}'>{s['regime'].capitalize()}</b>: "
            f"R² = <span class='{r2_cls}'>{s['rsquared']:.3f}</span> "
            f"(N = {s['nobs']:,})</li>"
        )

    return f"""
    <h2>3. Model Reliability</h2>
    <ul>{''.join(items)}</ul>
    <div class='note'>
      ℹ Training period: Jan 2022–Dec 2023 (DE-LU, post-energy crisis, BESS expansion period).
      R² values below 0.50 reflect high PTU-level noise in aFRR prices — the model captures
      the structural CSS relationship, not short-term intra-day volatility.
    </div>
    """


# ---------------------------------------------------------------------------
# Section 4: Opportunity cost scenarios
# ---------------------------------------------------------------------------

def _build_scenario_table(regime: str) -> str:
    da_headers = "".join(
        f"<th>DA = {da:.0f} €/MWh</th>" for da in FORECAST_DA_SCENARIOS
    )

    rows = []
    for gas in SCENARIO_GAS_PRICES:
        cells = f"<td>Gas = {gas:.0f} €/MWh_th</td>"
        for da in FORECAST_DA_SCENARIOS:
            try:
                price = estimate_afrr_price(
                    da_price=da,
                    gas_price=gas,
                    eu_ets_price=FORECAST_ETS_PRICE,
                    regime=regime,
                )
                cells += f"<td>{price:.1f}</td>"
            except Exception:
                cells += "<td>—</td>"
        rows.append(f"<tr>{cells}</tr>")

    return f"""
    <h3>Regime: <span class='regime-{regime}'>{regime.capitalize()}</span></h3>
    <table class='scenario-table'>
      <tr><th>Gas / DA Price</th>{da_headers}</tr>
      {''.join(rows)}
    </table>
    """


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def generate_report(
    df: pd.DataFrame,
    outputs_reports: Path = OUTPUTS_REPORTS,
    outputs_models: Path = OUTPUTS_MODELS,
) -> Path:
    """Generate a self-contained HTML report.

    Args:
        df: DataFrame with regime labels and market data columns.
        outputs_reports: Directory to save HTML report into.
        outputs_models: Directory containing regime pkl files and metadata JSON.

    Returns:
        Path to saved HTML file.
    """
    outputs_reports.mkdir(parents=True, exist_ok=True)
    out_path = outputs_reports / "market_regime_report.html"

    models = _load_all_models(outputs_models)
    metadata = _load_metadata(outputs_models)

    stats_list = [
        _extract_stats(models[regime], regime, df, metadata)
        for regime in REGIME_ORDER
    ]

    scenario_sections = "".join(_build_scenario_table(regime) for regime in REGIME_ORDER)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>aFRR Opportunity Cost — Market Regime Report</title>
  {_css_styles()}
</head>
<body>
  <h1>aFRR-up Opportunity Cost: Market Regime Report</h1>
  <p style='color:#666;font-size:14px;'>
    Gas peaker plant (CCGT) — DE-LU bidding zone, 2022–2023 |
    Model: <code>aFRR_Price = β₀ + β₁·CSS</code> (fitted per regime)
  </p>

  {_build_findings(stats_list)}
  {_build_regime_snapshot(stats_list)}
  {_build_reliability(stats_list)}

  <h2>4. Opportunity Cost Scenarios</h2>
  <p>Estimated aFRR-up bid price (€/MW) for combinations of gas forward price and
     DA price forecast. ETS fixed at {FORECAST_ETS_PRICE:.0f} €/tCO₂.</p>
  {scenario_sections}

  <footer>
    Generated by <code>src/analysis/report_generator.py</code> &mdash;
    aFRR Opportunity Cost Calculator &mdash; DE-LU 2022–2023
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
