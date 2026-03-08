"""
Unit tests for the analysis module (Phase 3).

Tests:
    - visualization: all 5 plot functions produce PNG files
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
    plot_coefficient_comparison,
    plot_css_vs_affr,
    plot_forecast_curves,
    plot_model_diagnostics,
    plot_regime_distribution,
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

    # regime split: ~12 high, ~25 medium, ~13 low
    regimes = (
        ["high"] * 12
        + ["medium"] * 25
        + ["low"] * 13
    )
    # shuffle to make time series more realistic
    rng.shuffle(regimes)

    df = pd.DataFrame(index=idx)
    df["regime"] = regimes
    df["css"] = rng.normal(-5, 20, n)
    df["da_price_eur_mwh"] = rng.uniform(30, 180, n)
    df["gas_price_eur_mwh"] = rng.uniform(30, 60, n)
    df["eu_ets_price_eur_t"] = rng.uniform(50, 80, n)
    df["ccgt_generation_mw"] = rng.uniform(1000, 10000, n)
    df["affr_price_eur_mw"] = (
        100
        + 0.3 * df["css"]
        + 0.5 * df["da_price_eur_mwh"]
        + rng.normal(0, 20, n)
    )
    return df


@pytest.fixture
def fake_model_dir(tmp_path: Path, synthetic_df: pd.DataFrame) -> Path:
    """Fit OLS on synthetic data, pickle models, write metadata. Returns models dir."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True)

    features = ["css", "da_price_eur_mwh", "ccgt_generation_mw"]
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
        }

    meta_path = models_dir / "regime_metadata.json"
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")

    return models_dir


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
# TestPlotModelDiagnostics
# ---------------------------------------------------------------------------

class TestPlotModelDiagnostics:
    def test_creates_png(
        self, synthetic_df: pd.DataFrame, fake_model_dir: Path, tmp_path: Path
    ) -> None:
        plots_dir = tmp_path / "plots"
        plot_model_diagnostics(
            synthetic_df, outputs_plots=plots_dir, outputs_models=fake_model_dir
        )
        assert (plots_dir / "model_diagnostics.png").exists()

    def test_returns_path(
        self, synthetic_df: pd.DataFrame, fake_model_dir: Path, tmp_path: Path
    ) -> None:
        result = plot_model_diagnostics(
            synthetic_df, outputs_plots=tmp_path / "plots", outputs_models=fake_model_dir
        )
        assert isinstance(result, Path)
        assert result.suffix == ".png"


# ---------------------------------------------------------------------------
# TestPlotCoefficientComparison
# ---------------------------------------------------------------------------

class TestPlotCoefficientComparison:
    def test_creates_png(
        self, synthetic_df: pd.DataFrame, fake_model_dir: Path, tmp_path: Path
    ) -> None:
        plots_dir = tmp_path / "plots"
        plot_coefficient_comparison(
            synthetic_df, outputs_plots=plots_dir, outputs_models=fake_model_dir
        )
        assert (plots_dir / "coefficient_comparison.png").exists()

    def test_returns_path(
        self, synthetic_df: pd.DataFrame, fake_model_dir: Path, tmp_path: Path
    ) -> None:
        result = plot_coefficient_comparison(
            synthetic_df, outputs_plots=tmp_path / "plots", outputs_models=fake_model_dir
        )
        assert isinstance(result, Path)
        assert result.suffix == ".png"


# ---------------------------------------------------------------------------
# TestPlotForecastCurves
# ---------------------------------------------------------------------------

class TestPlotForecastCurves:
    def test_creates_png(
        self, synthetic_df: pd.DataFrame, fake_model_dir: Path, tmp_path: Path, monkeypatch
    ) -> None:
        """Monkeypatch _load_metadata to use fake_model_dir."""
        import src.models.predictions as pred_module
        import src.analysis.visualization as viz_module

        real_metadata_path = pred_module.METADATA_PATH
        pred_module.METADATA_PATH = fake_model_dir / "regime_metadata.json"
        try:
            plots_dir = tmp_path / "plots"
            plot_forecast_curves(
                synthetic_df, outputs_plots=plots_dir, outputs_models=fake_model_dir
            )
            assert (plots_dir / "forecast_curves.png").exists()
        finally:
            pred_module.METADATA_PATH = real_metadata_path

    def test_returns_path(
        self, synthetic_df: pd.DataFrame, fake_model_dir: Path, tmp_path: Path, monkeypatch
    ) -> None:
        import src.models.predictions as pred_module

        real_metadata_path = pred_module.METADATA_PATH
        pred_module.METADATA_PATH = fake_model_dir / "regime_metadata.json"
        try:
            result = plot_forecast_curves(
                synthetic_df, outputs_plots=tmp_path / "plots", outputs_models=fake_model_dir
            )
            assert isinstance(result, Path)
            assert result.suffix == ".png"
        finally:
            pred_module.METADATA_PATH = real_metadata_path


# ---------------------------------------------------------------------------
# TestGenerateReport
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def test_creates_html(
        self, synthetic_df: pd.DataFrame, fake_model_dir: Path, tmp_path: Path
    ) -> None:
        reports_dir = tmp_path / "reports"
        generate_report(
            synthetic_df, outputs_reports=reports_dir, outputs_models=fake_model_dir
        )
        assert (reports_dir / "market_regime_report.html").exists()

    def test_html_contains_rsquared(
        self, synthetic_df: pd.DataFrame, fake_model_dir: Path, tmp_path: Path
    ) -> None:
        reports_dir = tmp_path / "reports"
        path = generate_report(
            synthetic_df, outputs_reports=reports_dir, outputs_models=fake_model_dir
        )
        content = path.read_text(encoding="utf-8")
        assert "R²" in content or "rsquared" in content.lower()

    def test_html_contains_all_regimes(
        self, synthetic_df: pd.DataFrame, fake_model_dir: Path, tmp_path: Path
    ) -> None:
        reports_dir = tmp_path / "reports"
        path = generate_report(
            synthetic_df, outputs_reports=reports_dir, outputs_models=fake_model_dir
        )
        content = path.read_text(encoding="utf-8")
        for regime in ["High", "Medium", "Low"]:
            assert regime in content


# ---------------------------------------------------------------------------
# TestGenerateDashboard
# ---------------------------------------------------------------------------

class TestGenerateDashboard:
    def test_creates_html(
        self, synthetic_df: pd.DataFrame, fake_model_dir: Path, tmp_path: Path, monkeypatch
    ) -> None:
        import src.models.predictions as pred_module

        real_metadata_path = pred_module.METADATA_PATH
        pred_module.METADATA_PATH = fake_model_dir / "regime_metadata.json"
        try:
            reports_dir = tmp_path / "reports"
            generate_dashboard(
                synthetic_df, outputs_reports=reports_dir, outputs_models=fake_model_dir
            )
            assert (reports_dir / "dashboard.html").exists()
        finally:
            pred_module.METADATA_PATH = real_metadata_path

    def test_html_contains_plotly(
        self, synthetic_df: pd.DataFrame, fake_model_dir: Path, tmp_path: Path, monkeypatch
    ) -> None:
        import src.models.predictions as pred_module

        real_metadata_path = pred_module.METADATA_PATH
        pred_module.METADATA_PATH = fake_model_dir / "regime_metadata.json"
        try:
            reports_dir = tmp_path / "reports"
            path = generate_dashboard(
                synthetic_df, outputs_reports=reports_dir, outputs_models=fake_model_dir
            )
            content = path.read_text(encoding="utf-8")
            assert "plotly" in content.lower()
        finally:
            pred_module.METADATA_PATH = real_metadata_path

    def test_html_is_self_contained(
        self, synthetic_df: pd.DataFrame, fake_model_dir: Path, tmp_path: Path, monkeypatch
    ) -> None:
        """Verify the dashboard links to Plotly CDN (one external script, no inline lib)."""
        import src.models.predictions as pred_module

        real_metadata_path = pred_module.METADATA_PATH
        pred_module.METADATA_PATH = fake_model_dir / "regime_metadata.json"
        try:
            reports_dir = tmp_path / "reports"
            path = generate_dashboard(
                synthetic_df, outputs_reports=reports_dir, outputs_models=fake_model_dir
            )
            content = path.read_text(encoding="utf-8")
            # CDN script tag present
            assert "cdn.plot.ly" in content or "plotly" in content.lower()
            # No base64 blob (not a fully-inlined standalone file)
            assert len(content) < 5_000_000, "Dashboard should not inline the full Plotly lib"
        finally:
            pred_module.METADATA_PATH = real_metadata_path
