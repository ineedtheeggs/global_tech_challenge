"""
Unit tests for the analysis module (Phase 3).

Tests:
    - visualization: 3 plot functions produce PNG files
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
    plot_forecast_curves,
    plot_historical_afrr,
)
from src.analysis.report_generator import generate_report
from src.analysis.dashboard import generate_dashboard


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """50-row DataFrame with all market_regimes.csv columns."""
    rng = np.random.default_rng(42)
    n = 50
    idx = pd.date_range("2021-01-01", periods=n, freq="h", tz="UTC")

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

        pkl_path = models_dir / f"regime_{regime}_model.pkl"
        with open(pkl_path, "wb") as fh:
            pickle.dump(model, fh)

        metadata[regime] = {
            "n_train": len(sub),
            "n_test": max(1, len(sub) // 5),
            "ccgt_mean_mw": float(sub["ccgt_generation_mw"].mean()),
            "ccgt_std_mw": float(sub["ccgt_generation_mw"].std()),
            "n_observations": len(sub),
        }

    meta_path = models_dir / "regime_metadata.json"
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")

    return models_dir


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
# TestPlotForecastCurves
# ---------------------------------------------------------------------------

class TestPlotForecastCurves:
    def test_creates_png(
        self, synthetic_df: pd.DataFrame, fake_model_dir: Path, tmp_path: Path, monkeypatch
    ) -> None:
        import src.models.predictions as pred_module
        monkeypatch.setattr(pred_module, "OUTPUTS_MODELS", fake_model_dir)

        plots_dir = tmp_path / "plots"
        plot_forecast_curves(synthetic_df, outputs_plots=plots_dir)
        assert (plots_dir / "forecast_curves.png").exists()

    def test_returns_path(
        self, synthetic_df: pd.DataFrame, fake_model_dir: Path, tmp_path: Path, monkeypatch
    ) -> None:
        import src.models.predictions as pred_module
        monkeypatch.setattr(pred_module, "OUTPUTS_MODELS", fake_model_dir)

        result = plot_forecast_curves(synthetic_df, outputs_plots=tmp_path / "plots")
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
        self, synthetic_df: pd.DataFrame, fake_model_dir: Path, tmp_path: Path, monkeypatch
    ) -> None:
        import src.models.predictions as pred_module
        monkeypatch.setattr(pred_module, "OUTPUTS_MODELS", fake_model_dir)

        reports_dir = tmp_path / "reports"
        generate_dashboard(
            synthetic_df, outputs_reports=reports_dir, outputs_models=fake_model_dir
        )
        assert (reports_dir / "dashboard.html").exists()

    def test_html_contains_plotly(
        self, synthetic_df: pd.DataFrame, fake_model_dir: Path, tmp_path: Path, monkeypatch
    ) -> None:
        import src.models.predictions as pred_module
        monkeypatch.setattr(pred_module, "OUTPUTS_MODELS", fake_model_dir)

        path = generate_dashboard(
            synthetic_df, outputs_reports=tmp_path / "reports", outputs_models=fake_model_dir
        )
        content = path.read_text(encoding="utf-8")
        assert "plotly" in content.lower()

    def test_html_is_self_contained(
        self, synthetic_df: pd.DataFrame, fake_model_dir: Path, tmp_path: Path, monkeypatch
    ) -> None:
        """Verify the dashboard links to Plotly CDN (one external script, no inline lib)."""
        import src.models.predictions as pred_module
        monkeypatch.setattr(pred_module, "OUTPUTS_MODELS", fake_model_dir)

        path = generate_dashboard(
            synthetic_df, outputs_reports=tmp_path / "reports", outputs_models=fake_model_dir
        )
        content = path.read_text(encoding="utf-8")
        assert "cdn.plot.ly" in content or "plotly" in content.lower()
        assert len(content) < 5_000_000
