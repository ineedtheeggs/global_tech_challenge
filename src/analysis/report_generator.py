"""
HTML report generator for aFRR opportunity cost analysis.

Produces a self-contained HTML report with:
- Executive summary
- Regime distribution table
- Model metrics table (R², DW, F-stat per regime)
- Coefficient tables with significance highlights
- VIF tables with multicollinearity warnings

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
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.utils.config import DATA_PROCESSED, OUTPUTS_MODELS, OUTPUTS_REPORTS
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

REGIME_ORDER: list[str] = ["high", "medium", "low"]
REGIME_COLORS: dict[str, str] = {
    "high": "#2196F3",
    "medium": "#FF9800",
    "low": "#F44336",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_all_models(outputs_models: Path) -> dict:
    """Load pickled OLS models for all three regimes.

    Args:
        outputs_models: Directory containing regime pkl files.

    Returns:
        Dict mapping regime name to fitted statsmodels OLS results object.
    """
    models = {}
    for regime in REGIME_ORDER:
        path = outputs_models / f"regime_{regime}_model.pkl"
        with open(path, "rb") as fh:
            models[regime] = pickle.load(fh)
    return models


def _load_metadata(outputs_models: Path) -> dict:
    """Load regime_metadata.json sidecar.

    Args:
        outputs_models: Directory containing regime_metadata.json.

    Returns:
        Dict with per-regime statistics.
    """
    path = outputs_models / "regime_metadata.json"
    with open(path) as fh:
        return json.load(fh)


def _extract_model_stats(model, regime: str, metadata: dict) -> dict:
    """Extract key statistics from a fitted OLS model.

    Args:
        model: Fitted statsmodels OLS results object.
        regime: Regime name ('high', 'medium', 'low').
        metadata: Loaded regime_metadata.json content.

    Returns:
        Dict with R², Adj R², DW, F-stat, p-value, N, params, conf_int, pvalues, VIF.
    """
    resid = model.resid.values
    dw = durbin_watson(resid)
    ci = model.conf_int()
    exog_names = model.model.exog_names
    vif = {
        name: float(variance_inflation_factor(model.model.exog, i))
        for i, name in enumerate(exog_names)
    }

    return {
        "regime": regime,
        "rsquared": model.rsquared,
        "rsquared_adj": model.rsquared_adj,
        "durbin_watson": dw,
        "fstat": model.fvalue,
        "f_pvalue": model.f_pvalue,
        "nobs": int(model.nobs),
        "params": model.params.to_dict(),
        "conf_int": {
            name: (float(ci.loc[name, 0]), float(ci.loc[name, 1]))
            for name in exog_names
        },
        "pvalues": model.pvalues.to_dict(),
        "vif": vif,
        "ccgt_mean_mw": metadata[regime].get("ccgt_mean_mw", float("nan")),
        "ccgt_std_mw": metadata[regime].get("ccgt_std_mw", float("nan")),
        "n_train": metadata[regime].get("n_train", 0),
        "n_test": metadata[regime].get("n_test", 0),
    }


# ---------------------------------------------------------------------------
# HTML builders
# ---------------------------------------------------------------------------

def _css_styles() -> str:
    return """
    <style>
      body { font-family: Arial, sans-serif; max-width: 1200px; margin: 40px auto;
             padding: 0 20px; background: #f8f9fa; color: #333; }
      h1 { color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 10px; }
      h2 { color: #283593; margin-top: 40px; border-left: 4px solid #283593;
           padding-left: 12px; }
      h3 { color: #37474f; }
      .summary-box { background: #e8eaf6; border-left: 5px solid #3949ab;
                     padding: 16px 20px; border-radius: 4px; margin: 16px 0; }
      .warning { background: #fff3e0; border-left: 5px solid #ef6c00;
                 padding: 12px 16px; border-radius: 4px; margin: 8px 0; font-size: 14px; }
      table { border-collapse: collapse; width: 100%; margin: 16px 0;
              background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
      th { background: #37474f; color: white; padding: 10px 14px;
           text-align: left; font-weight: 600; }
      td { padding: 8px 14px; border-bottom: 1px solid #e0e0e0; font-size: 14px; }
      tr:hover { background: #f5f5f5; }
      .sig   { background-color: #e8f5e9; }
      .nosig { background-color: #ffebee; }
      .warn-vif { background-color: #fff3e0; }
      .regime-high   { color: #1565C0; font-weight: 600; }
      .regime-medium { color: #E65100; font-weight: 600; }
      .regime-low    { color: #B71C1C; font-weight: 600; }
      .metric-good { color: #2e7d32; font-weight: 600; }
      .metric-warn { color: #e65100; font-weight: 600; }
      footer { margin-top: 60px; font-size: 12px; color: #9e9e9e;
               border-top: 1px solid #ddd; padding-top: 16px; }
    </style>
    """


def _fmt_pval(p: float) -> str:
    if p < 0.001:
        return "<0.001"
    return f"{p:.4f}"


def _regime_class(regime: str) -> str:
    return f"regime-{regime}"


def _build_executive_summary(stats_list: list[dict]) -> str:
    rows = []
    for s in stats_list:
        r2_class = "metric-good" if s["rsquared"] >= 0.5 else "metric-warn"
        dw_class = "metric-good" if 1.8 <= s["durbin_watson"] <= 2.2 else "metric-warn"
        rows.append(
            f"<li><span class='{_regime_class(s['regime'])}'>{s['regime'].capitalize()} regime</span>: "
            f"R²=<span class='{r2_class}'>{s['rsquared']:.3f}</span>, "
            f"Adj R²={s['rsquared_adj']:.3f}, "
            f"DW=<span class='{dw_class}'>{s['durbin_watson']:.2f}</span>, "
            f"N={s['nobs']:,}</li>"
        )

    warnings = []
    for s in stats_list:
        if s["rsquared"] < 0.5:
            warnings.append(
                f"<div class='warning'>⚠ <b>{s['regime'].capitalize()} regime</b>: "
                f"R²={s['rsquared']:.3f} is below the 0.50 target. "
                f"The 2021-only training window limits variance explained; "
                f"a longer dataset should improve fit.</div>"
            )
        if not (1.8 <= s["durbin_watson"] <= 2.2):
            direction = "positive" if s["durbin_watson"] < 1.8 else "negative"
            warnings.append(
                f"<div class='warning'>⚠ <b>{s['regime'].capitalize()} regime</b>: "
                f"DW={s['durbin_watson']:.2f} suggests {direction} autocorrelation. "
                f"Model captures broad trend; intra-day PTU noise is not modelled.</div>"
            )

    return f"""
    <h2>1. Executive Summary</h2>
    <div class='summary-box'>
      <p>Three OLS models were fitted, one per market regime, on 2021 DE-LU PTU data.
         The Clean Spark Spread (CSS), day-ahead price, and CCGT generation are used as
         predictors for aFRR-up capacity prices. Key results:</p>
      <ul>{''.join(rows)}</ul>
    </div>
    {''.join(warnings)}
    """


def _build_regime_table(stats_list: list[dict]) -> str:
    rows = []
    for s in stats_list:
        rows.append(
            f"<tr>"
            f"<td class='{_regime_class(s['regime'])}'>{s['regime'].capitalize()}</td>"
            f"<td>{s['nobs']:,}</td>"
            f"<td>{s['n_train']:,}</td>"
            f"<td>{s['n_test']:,}</td>"
            f"<td>{s['ccgt_mean_mw']:.0f}</td>"
            f"<td>{s['ccgt_std_mw']:.0f}</td>"
            f"</tr>"
        )
    return f"""
    <h2>2. Regime Distribution</h2>
    <table>
      <tr>
        <th>Regime</th><th>Total PTUs</th><th>Train</th><th>Test</th>
        <th>CCGT Mean (MW)</th><th>CCGT Std (MW)</th>
      </tr>
      {''.join(rows)}
    </table>
    """


def _build_metrics_table(stats_list: list[dict]) -> str:
    metrics = [
        ("R²", lambda s: f"{s['rsquared']:.4f}", lambda s: s["rsquared"] >= 0.5),
        ("Adj R²", lambda s: f"{s['rsquared_adj']:.4f}", None),
        ("Durbin-Watson", lambda s: f"{s['durbin_watson']:.3f}",
         lambda s: 1.8 <= s["durbin_watson"] <= 2.2),
        ("F-statistic", lambda s: f"{s['fstat']:.2f}", None),
        ("F p-value", lambda s: _fmt_pval(s["f_pvalue"]), lambda s: s["f_pvalue"] < 0.05),
        ("N obs", lambda s: f"{s['nobs']:,}", None),
    ]

    header_cells = "".join(
        f"<th class='{_regime_class(s['regime'])}'>{s['regime'].capitalize()}</th>"
        for s in stats_list
    )

    rows = []
    for label, fmt_fn, good_fn in metrics:
        cells = f"<td><b>{label}</b></td>"
        for s in stats_list:
            val = fmt_fn(s)
            cls = ""
            if good_fn is not None:
                cls = "metric-good" if good_fn(s) else "metric-warn"
            cells += f"<td class='{cls}'>{val}</td>"
        rows.append(f"<tr>{cells}</tr>")

    return f"""
    <h2>3. Model Performance Metrics</h2>
    <table>
      <tr><th>Metric</th>{header_cells}</tr>
      {''.join(rows)}
    </table>
    """


def _build_coefficient_table(stats: dict) -> str:
    regime = stats["regime"]
    params = stats["params"]
    ci = stats["conf_int"]
    pvals = stats["pvalues"]

    predictors = list(params.keys())
    rows = []
    for pred in predictors:
        p = pvals.get(pred, 1.0)
        sig_class = "sig" if p < 0.05 else "nosig"
        ci_lo, ci_hi = ci.get(pred, (float("nan"), float("nan")))
        rows.append(
            f"<tr class='{sig_class}'>"
            f"<td><code>{pred}</code></td>"
            f"<td>{params[pred]:.6f}</td>"
            f"<td>[{ci_lo:.6f}, {ci_hi:.6f}]</td>"
            f"<td>{_fmt_pval(p)}</td>"
            f"<td>{'✓' if p < 0.05 else '✗'}</td>"
            f"</tr>"
        )

    return f"""
    <h3>Regime: <span class='{_regime_class(regime)}'>{regime.capitalize()}</span></h3>
    <table>
      <tr>
        <th>Predictor</th><th>Coefficient</th><th>95% CI</th>
        <th>p-value</th><th>Significant (p&lt;0.05)</th>
      </tr>
      {''.join(rows)}
    </table>
    <p style='font-size:12px;color:#777;'>
      Green rows: p&lt;0.05. Red rows: p≥0.05 (not statistically significant at 5% level).
    </p>
    """


def _build_vif_table(stats: dict) -> str:
    regime = stats["regime"]
    vif = stats["vif"]
    rows = []
    for name, val in vif.items():
        cls = "warn-vif" if val > 5 else ""
        rows.append(
            f"<tr class='{cls}'>"
            f"<td><code>{name}</code></td>"
            f"<td>{val:.2f}</td>"
            f"<td>{'⚠ High (expected for const)' if val > 5 else 'OK'}</td>"
            f"</tr>"
        )
    return f"""
    <h3>VIF — Regime: <span class='{_regime_class(regime)}'>{regime.capitalize()}</span></h3>
    <table>
      <tr><th>Variable</th><th>VIF</th><th>Status</th></tr>
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
    """Generate a self-contained HTML model summary report.

    Args:
        df: DataFrame with regime labels (used for distribution counts; models
            loaded independently from disk).
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
        _extract_model_stats(models[regime], regime, metadata)
        for regime in REGIME_ORDER
    ]

    coeff_sections = "".join(_build_coefficient_table(s) for s in stats_list)
    vif_sections = "".join(_build_vif_table(s) for s in stats_list)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>aFRR Opportunity Cost — Market Regime Report</title>
  {_css_styles()}
</head>
<body>
  <h1>aFRR-up Opportunity Cost: Market Regime Analysis Report</h1>
  <p style='color:#666;font-size:14px;'>
    Gas peaker plant (CCGT) aFRR spin-up participation analysis — DE-LU bidding zone, 2021
  </p>

  {_build_executive_summary(stats_list)}
  {_build_regime_table(stats_list)}
  {_build_metrics_table(stats_list)}

  <h2>4. Model Coefficients</h2>
  <p>Model specification:
     <code>aFRR_Price = β₀ + β₁·CSS + β₂·DA_Price + β₃·CCGT_Gen + ε</code>
     (fitted separately per regime)
  </p>
  {coeff_sections}

  <h2>5. Variance Inflation Factors (Multicollinearity Check)</h2>
  <p>VIF &gt; 5 flagged in orange. High VIF for <code>const</code> is expected and
     does not indicate multicollinearity among predictors.</p>
  {vif_sections}

  <footer>
    Generated by <code>src/analysis/report_generator.py</code> &mdash;
    aFRR Opportunity Cost Calculator &mdash; DE-LU 2021
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
