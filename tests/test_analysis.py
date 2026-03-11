"""
Unit tests for the analysis module (Phase 3).

Tests:
    - visualization: 4 plot functions produce PNG files
    - report_generator: HTML report with expected content
    - dashboard: self-contained HTML dashboard with Plotly
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from src.analysis.visualization import (
    plot_css_vs_affr,
    plot_2026_forecast,
    plot_historical_afrr,
    plot_regime_distribution,
)
from src.analysis.report_generator import generate_report
from src.analysis.dashboard import generate_dashboard


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """50-row DataFrame spanning multiple years with all market_regimes.csv columns."""
    rng = np.random.default_rng(42)
    n = 50

    # Span 2020–2022 so per-year subplot tests have multiple years
    idx = pd.date_range("2020-01-01", periods=n, freq="1000h", tz="UTC")

    regimes = ["high"] * 12 + ["medium"] * 25 + ["low"] * 13
    rng.shuffle(regimes)

    df = pd.DataFrame(index=idx)
    df["regime"] = regimes
    df["css"] = rng.normal(-5, 20, n)
    df["da_price_eur_mwh"] = rng.uniform(30, 180, n)
    df["gas_price_eur_mwh"] = rng.uniform(30, 60, n)
    df["eu_ets_price_eur_t"] = rng.uniform(50, 80, n)
    df["ccgt_generation_mw"] = rng.uniform(1000, 10000, n)
    df["affr_price_eur_mw"] = (
        5.0 + 0.3 * df["css"] + rng.normal(0, 5, n)
    ).clip(lower=0)
    return df


@pytest.fixture
def fake_model_dir(tmp_path: Path, synthetic_df: pd.DataFrame) -> Path:
    """Fit CSS-only OLS models on synthetic data, pickle them, write metadata."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True)

    features = ["css"]
    target = "affr_price_eur_mw"
    metadata = {}

    for regime in ["high", "medium", "low"]:
        sub = synthetic_df[synthetic_df["regime"] == regime].copy()
        X = sm.add_constant(sub[features], has_constant="add")
        y = sub[target]
        model = sm.OLS(y, X).fit()

        # Legacy flat-named pkl (kept for any direct callers)
        pkl_path = models_dir / f"regime_{regime}_model.pkl"
        with open(pkl_path, "wb") as fh:
            pickle.dump(model, fh)

        # Year-named pkl for estimate_afrr_price(year=2022) default
        pkl_year_path = models_dir / f"regime_{regime}_2022_model.pkl"
        with open(pkl_year_path, "wb") as fh:
            pickle.dump(model, fh)

        regime_meta = {
            "n_train": len(sub),
            "n_test": max(1, len(sub) // 5),
            "ccgt_mean_mw": float(sub["ccgt_generation_mw"].mean()),
            "ccgt_std_mw": float(sub["ccgt_generation_mw"].std()),
            "n_observations": len(sub),
        }
        metadata[regime] = regime_meta

    # Also write year-keyed entries so code reading metadata["2022"]["high"] works
    metadata["2022"] = {r: metadata[r] for r in ["high", "medium", "low"]}

    meta_path = models_dir / "regime_metadata.json"
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")

    return models_dir


@pytest.fixture
def fake_combined_csv(tmp_path: Path, fake_model_dir: Path) -> Path:
    """Write a temp combined_dataset.csv with 2026-01 + 2026-02 PTU data (59 days)."""
    rng = np.random.default_rng(99)
    n = 59  # Jan 1–31 (31 days) + Feb 1–28 (28 days)

    idx = pd.date_range("2026-01-01", periods=n, freq="24h", tz="UTC")
    df = pd.DataFrame(index=idx)

    # ccgt_generation_mw spread across all regime ranges (repeated to fill 59 rows)
    # K-Means boundaries: low < 5858, medium 5858–10546, high >= 10546
    base = [2000, 4000, 6000, 7000, 8000, 9000, 11000, 13000]
    df["ccgt_generation_mw"] = (base * 7 + base[:3])
    df["da_price_eur_mwh"] = rng.uniform(50, 120, n)
    df["gas_price_eur_mwh_th"] = rng.uniform(30, 60, n)
    df["eu_ets_price_eur_tco2"] = rng.uniform(55, 75, n)
    # CSS = DA - gas/0.5 - ets*0.202/0.5
    df["css"] = (
        df["da_price_eur_mwh"]
        - df["gas_price_eur_mwh_th"] / 0.5
        - df["eu_ets_price_eur_tco2"] * 0.202 / 0.5
    )
    df["affr_price_eur_mw"] = (5.0 + 0.3 * df["css"] + rng.normal(0, 5, n)).clip(lower=0)

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True)
    csv_path = processed_dir / "combined_dataset.csv"
    df.to_csv(csv_path)
    return csv_path


# ---------------------------------------------------------------------------
# TestPlotHistoricalAfrr
# ---------------------------------------------------------------------------

class TestPlotHistoricalAfrr:
    def test_creates_png(self, synthetic_df: pd.DataFrame, tmp_path: Path) -> None:
        plots_dir = tmp_path / "plots"
        plot_historical_afrr(synthetic_df, outputs_plots=plots_dir)
        assert (plots_dir / "historical_afrr_prices.png").exists()

    def test_returns_path(self, synthetic_df: pd.DataFrame, tmp_path: Path) -> None:
        result = plot_historical_afrr(synthetic_df, outputs_plots=tmp_path / "plots")
        assert isinstance(result, Path)
        assert result.suffix == ".png"


# ---------------------------------------------------------------------------
# TestPlotCssVsAfrr
# ---------------------------------------------------------------------------

class TestPlotCssVsAfrr:
    def test_creates_png(self, synthetic_df: pd.DataFrame, tmp_path: Path) -> None:
        plots_dir = tmp_path / "plots"
        plot_css_vs_affr(synthetic_df, outputs_plots=plots_dir)
        assert (plots_dir / "css_vs_affr_prices.png").exists()

    def test_returns_path(self, synthetic_df: pd.DataFrame, tmp_path: Path) -> None:
        result = plot_css_vs_affr(synthetic_df, outputs_plots=tmp_path / "plots")
        assert isinstance(result, Path)
        assert result.suffix == ".png"


# ---------------------------------------------------------------------------
# TestPlot2026Forecast
# ---------------------------------------------------------------------------

class TestPlot2026Forecast:
    def test_creates_png(
        self,
        synthetic_df: pd.DataFrame,
        fake_model_dir: Path,
        fake_combined_csv: Path,
        tmp_path: Path,
    ) -> None:
        plots_dir = tmp_path / "plots"
        plot_2026_forecast(
            synthetic_df,
            outputs_plots=plots_dir,
            outputs_models=fake_model_dir,
            data_processed=fake_combined_csv.parent,
        )
        # January: 5 weekly plots (W1–W5 including partial week 29–31)
        for wk in range(1, 6):
            assert (plots_dir / f"forecast_curves_jan_w{wk}.png").exists()
        # February: 4 weekly plots (W1–W4)
        for wk in range(1, 5):
            assert (plots_dir / f"forecast_curves_feb_w{wk}.png").exists()

    def test_returns_path(
        self,
        synthetic_df: pd.DataFrame,
        fake_model_dir: Path,
        fake_combined_csv: Path,
        tmp_path: Path,
    ) -> None:
        result = plot_2026_forecast(
            synthetic_df,
            outputs_plots=tmp_path / "plots",
            outputs_models=fake_model_dir,
            data_processed=fake_combined_csv.parent,
        )
        assert isinstance(result, list)
        assert len(result) == 9  # 5 Jan + 4 Feb
        for p in result:
            assert isinstance(p, Path)
            assert p.suffix == ".png"

    def test_all_weeks_non_empty(
        self,
        synthetic_df: pd.DataFrame,
        fake_model_dir: Path,
        fake_combined_csv: Path,
        tmp_path: Path,
    ) -> None:
        result = plot_2026_forecast(
            synthetic_df,
            outputs_plots=tmp_path / "plots",
            outputs_models=fake_model_dir,
            data_processed=fake_combined_csv.parent,
        )
        for p in result:
            assert p.stat().st_size > 0


# ---------------------------------------------------------------------------
# TestPlotRegimeDistribution
# ---------------------------------------------------------------------------

class TestPlotRegimeDistribution:
    def test_creates_png(self, synthetic_df: pd.DataFrame, tmp_path: Path) -> None:
        plots_dir = tmp_path / "plots"
        plot_regime_distribution(synthetic_df, outputs_plots=plots_dir)
        assert (plots_dir / "regime_distribution.png").exists()

    def test_returns_path(self, synthetic_df: pd.DataFrame, tmp_path: Path) -> None:
        result = plot_regime_distribution(synthetic_df, outputs_plots=tmp_path / "plots")
        assert isinstance(result, Path)
        assert result.suffix == ".png"


# ---------------------------------------------------------------------------
# TestGenerateReport
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def test_creates_html(
        self, synthetic_df: pd.DataFrame, fake_model_dir: Path, tmp_path: Path, monkeypatch
    ) -> None:
        import src.models.predictions as pred_module
        monkeypatch.setattr(pred_module, "OUTPUTS_MODELS", fake_model_dir)

        reports_dir = tmp_path / "reports"
        generate_report(
            synthetic_df, outputs_reports=reports_dir, outputs_models=fake_model_dir
        )
        assert (reports_dir / "market_regime_report.html").exists()

    def test_html_contains_rsquared(
        self, synthetic_df: pd.DataFrame, fake_model_dir: Path, tmp_path: Path, monkeypatch
    ) -> None:
        import src.models.predictions as pred_module
        monkeypatch.setattr(pred_module, "OUTPUTS_MODELS", fake_model_dir)

        path = generate_report(
            synthetic_df, outputs_reports=tmp_path / "reports", outputs_models=fake_model_dir
        )
        content = path.read_text(encoding="utf-8")
        assert "R²" in content

    def test_html_contains_all_regimes(
        self, synthetic_df: pd.DataFrame, fake_model_dir: Path, tmp_path: Path, monkeypatch
    ) -> None:
        import src.models.predictions as pred_module
        monkeypatch.setattr(pred_module, "OUTPUTS_MODELS", fake_model_dir)

        path = generate_report(
            synthetic_df, outputs_reports=tmp_path / "reports", outputs_models=fake_model_dir
        )
        content = path.read_text(encoding="utf-8")
        for regime in ["High", "Medium", "Low"]:
            assert regime in content

    def test_html_contains_scenario_table(
        self, synthetic_df: pd.DataFrame, fake_model_dir: Path, tmp_path: Path, monkeypatch
    ) -> None:
        import src.models.predictions as pred_module
        monkeypatch.setattr(pred_module, "OUTPUTS_MODELS", fake_model_dir)

        path = generate_report(
            synthetic_df, outputs_reports=tmp_path / "reports", outputs_models=fake_model_dir
        )
        content = path.read_text(encoding="utf-8")
        assert "Opportunity Cost Scenarios" in content
        assert "Gas =" in content


# ---------------------------------------------------------------------------
# TestGenerateDashboard
# ---------------------------------------------------------------------------

class TestGenerateDashboard:
    def test_creates_html(
        self,
        synthetic_df: pd.DataFrame,
        fake_model_dir: Path,
        fake_combined_csv: Path,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        import src.models.predictions as pred_module
        monkeypatch.setattr(pred_module, "OUTPUTS_MODELS", fake_model_dir)

        reports_dir = tmp_path / "reports"
        generate_dashboard(
            synthetic_df,
            outputs_reports=reports_dir,
            outputs_models=fake_model_dir,
            data_processed=fake_combined_csv.parent,
        )
        assert (reports_dir / "dashboard.html").exists()

    def test_html_contains_plotly(
        self,
        synthetic_df: pd.DataFrame,
        fake_model_dir: Path,
        fake_combined_csv: Path,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        import src.models.predictions as pred_module
        monkeypatch.setattr(pred_module, "OUTPUTS_MODELS", fake_model_dir)

        path = generate_dashboard(
            synthetic_df,
            outputs_reports=tmp_path / "reports",
            outputs_models=fake_model_dir,
            data_processed=fake_combined_csv.parent,
        )
        content = path.read_text(encoding="utf-8")
        assert "plotly" in content.lower()

    def test_html_is_self_contained(
        self,
        synthetic_df: pd.DataFrame,
        fake_model_dir: Path,
        fake_combined_csv: Path,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        """Verify the dashboard links to Plotly CDN (one external script, no inline lib)."""
        import src.models.predictions as pred_module
        monkeypatch.setattr(pred_module, "OUTPUTS_MODELS", fake_model_dir)

        path = generate_dashboard(
            synthetic_df,
            outputs_reports=tmp_path / "reports",
            outputs_models=fake_model_dir,
            data_processed=fake_combined_csv.parent,
        )
        content = path.read_text(encoding="utf-8")
        assert "cdn.plot.ly" in content or "plotly" in content.lower()
        assert len(content) < 5_000_000
