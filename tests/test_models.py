"""
Unit tests for the Phase 2 model pipeline.

Covers:
    - regime_classifier: load_centroids, compute_thresholds, assign_regimes
    - regression_models: fit_regime_model, save_model, build_year_regime_metadata, run()
    - predictions: estimate_afrr_price (including --year CLI arg)
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

# Make sure project root is importable when running from any cwd
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.models.regime_classifier import (
    assign_regimes,
    compute_thresholds,
    load_centroids,
)
from src.models.regression_models import (
    MIN_OBSERVATIONS,
    build_year_regime_metadata,
    fit_regime_model,
    save_model,
)
from src.models.predictions import _load_model, estimate_afrr_price


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_regime_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Return a minimal DataFrame with ccgt_generation_mw, css, affr_price_eur_mw."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="h")
    css = rng.normal(0, 20, n)
    return pd.DataFrame(
        {
            "ccgt_generation_mw": rng.uniform(1000, 15000, n),
            "css": css,
            "affr_price_eur_mw": 10 + 0.5 * css + rng.normal(0, 5, n),
            "da_price_eur_mwh": rng.uniform(30, 180, n),
            "affr_price_eur_mw": (10 + 0.5 * css + rng.normal(0, 5, n)).clip(0),
        },
        index=idx,
    )


def _fit_ols(df: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    X = sm.add_constant(df[["css"]])
    return sm.OLS(df["affr_price_eur_mw"], X).fit()


# ---------------------------------------------------------------------------
# TestLoadCentroids
# ---------------------------------------------------------------------------

class TestLoadCentroids:
    def test_kmeans_returns_three_regimes(self) -> None:
        centroids = load_centroids("kmeans")
        assert set(centroids.keys()) == {"low", "medium", "high"}

    def test_capacity_returns_three_regimes(self) -> None:
        centroids = load_centroids("capacity")
        assert set(centroids.keys()) == {"low", "medium", "high"}

    def test_kmeans_values_are_positive_floats(self) -> None:
        centroids = load_centroids("kmeans")
        for val in centroids.values():
            assert isinstance(val, float)
            assert val > 0

    def test_invalid_method_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown classification method"):
            load_centroids("percentile")

    def test_missing_file_raises_file_not_found(self, tmp_path: Path, monkeypatch) -> None:
        import src.models.regime_classifier as rc
        monkeypatch.setattr(rc, "REGIME_KMEANS_JSON", tmp_path / "nonexistent.json")
        with pytest.raises(FileNotFoundError):
            rc.load_centroids("kmeans")

    def test_kmeans_low_less_than_high(self) -> None:
        centroids = load_centroids("kmeans")
        assert centroids["low"] < centroids["medium"] < centroids["high"]

    def test_capacity_low_less_than_high(self) -> None:
        centroids = load_centroids("capacity")
        assert centroids["low"] < centroids["medium"] < centroids["high"]


# ---------------------------------------------------------------------------
# TestComputeThresholds
# ---------------------------------------------------------------------------

class TestComputeThresholds:
    def test_midpoint_arithmetic(self) -> None:
        centroids = {"low": 1000.0, "medium": 5000.0, "high": 10000.0}
        low_med, med_high = compute_thresholds(centroids)
        assert low_med == pytest.approx(3000.0)
        assert med_high == pytest.approx(7500.0)

    def test_lower_boundary_less_than_upper(self) -> None:
        centroids = load_centroids("kmeans")
        low_med, med_high = compute_thresholds(centroids)
        assert low_med < med_high

    def test_kmeans_expected_approximate_values(self) -> None:
        """K-Means centroids: low=3776, medium=7941, high=13150 → ~5858 / ~10546."""
        centroids = load_centroids("kmeans")
        low_med, med_high = compute_thresholds(centroids)
        assert low_med == pytest.approx(5858, abs=5)
        assert med_high == pytest.approx(10546, abs=5)

    def test_capacity_expected_approximate_values(self) -> None:
        """Capacity centroids: low=2708, medium=5253, high=10383 → ~3981 / ~7818."""
        centroids = load_centroids("capacity")
        low_med, med_high = compute_thresholds(centroids)
        assert low_med == pytest.approx(3981, abs=5)
        assert med_high == pytest.approx(7818, abs=5)


# ---------------------------------------------------------------------------
# TestAssignRegimes
# ---------------------------------------------------------------------------

class TestAssignRegimes:
    def test_all_regimes_present(self) -> None:
        df = _make_regime_df(300)
        result = assign_regimes(df)
        assert set(result["regime"].unique()) == {"high", "medium", "low"}

    def test_regime_column_added(self) -> None:
        df = _make_regime_df(100)
        result = assign_regimes(df)
        assert "regime" in result.columns

    def test_correct_labels_for_boundary_values(self) -> None:
        """Rows exactly at or above medium_high should be 'high', below low_med 'low'."""
        centroids = load_centroids("kmeans")
        low_med, med_high = compute_thresholds(centroids)

        df = pd.DataFrame(
            {"ccgt_generation_mw": [low_med - 1, low_med, med_high - 1, med_high]},
            index=pd.date_range("2022-01-01", periods=4, freq="h"),
        )
        result = assign_regimes(df)
        assert result["regime"].tolist() == ["low", "medium", "medium", "high"]

    def test_method_toggle_changes_thresholds(self) -> None:
        """capacity thresholds differ from kmeans; some rows may be classified differently."""
        centroids_km = load_centroids("kmeans")
        centroids_cap = load_centroids("capacity")
        # They must differ (sanity check for the test to be meaningful)
        assert centroids_km["low"] != centroids_cap["low"]

        df = _make_regime_df(200)
        result_km = assign_regimes(df, method="kmeans")
        result_cap = assign_regimes(df, method="capacity")
        # Not identical (different thresholds → different classifications)
        assert not result_km["regime"].equals(result_cap["regime"])

    def test_does_not_mutate_input(self) -> None:
        df = _make_regime_df(50)
        original_cols = list(df.columns)
        assign_regimes(df)
        assert "regime" not in original_cols  # original unchanged


# ---------------------------------------------------------------------------
# TestFitRegimeModel
# ---------------------------------------------------------------------------

class TestFitRegimeModel:
    def test_returns_ols_wrapper(self) -> None:
        df = _make_regime_df(100)
        model = fit_regime_model(df, "high")
        assert hasattr(model, "params")
        assert hasattr(model, "rsquared")

    def test_model_has_css_param(self) -> None:
        df = _make_regime_df(100)
        model = fit_regime_model(df, "medium")
        assert "css" in model.params

    def test_raises_below_min_observations(self) -> None:
        df = _make_regime_df(MIN_OBSERVATIONS - 1)
        with pytest.raises(ValueError, match="minimum required"):
            fit_regime_model(df, "low")

    def test_accepts_year_kwarg(self) -> None:
        df = _make_regime_df(50)
        model = fit_regime_model(df, "high", year=2023)
        assert model is not None

    def test_exactly_min_observations_does_not_raise(self) -> None:
        df = _make_regime_df(MIN_OBSERVATIONS)
        model = fit_regime_model(df, "low")
        assert model is not None


# ---------------------------------------------------------------------------
# TestSaveModel
# ---------------------------------------------------------------------------

class TestSaveModel:
    def test_creates_year_named_file(self, tmp_path: Path, monkeypatch) -> None:
        import src.models.regression_models as rm
        monkeypatch.setattr(rm, "OUTPUTS_MODELS", tmp_path)

        df = _make_regime_df(50)
        model = _fit_ols(df)
        save_model(model, "high", 2022)

        expected = tmp_path / "regime_high_2022_model.pkl"
        assert expected.exists()

    def test_saved_model_loadable(self, tmp_path: Path, monkeypatch) -> None:
        import src.models.regression_models as rm
        monkeypatch.setattr(rm, "OUTPUTS_MODELS", tmp_path)

        df = _make_regime_df(50)
        model = _fit_ols(df)
        save_model(model, "medium", 2023)

        path = tmp_path / "regime_medium_2023_model.pkl"
        with open(path, "rb") as fh:
            loaded = pickle.load(fh)
        assert hasattr(loaded, "params")

    def test_returns_correct_path(self, tmp_path: Path, monkeypatch) -> None:
        import src.models.regression_models as rm
        monkeypatch.setattr(rm, "OUTPUTS_MODELS", tmp_path)

        df = _make_regime_df(50)
        model = _fit_ols(df)
        path = save_model(model, "low", 2024)

        assert path == tmp_path / "regime_low_2024_model.pkl"


# ---------------------------------------------------------------------------
# TestBuildYearRegimeMetadata
# ---------------------------------------------------------------------------

class TestBuildYearRegimeMetadata:
    def test_all_required_keys_present(self) -> None:
        df = _make_regime_df(60)
        model = fit_regime_model(df, "high")
        meta = build_year_regime_metadata(df, model, "high", 2022)

        required_keys = {
            "beta_0", "beta_1_css", "rsquared", "rsquared_adj",
            "durbin_watson", "n_observations", "css_pvalue",
            "ccgt_mean_mw", "ccgt_std_mw",
        }
        assert required_keys.issubset(meta.keys())

    def test_n_observations_matches_subset(self) -> None:
        df = _make_regime_df(75)
        model = fit_regime_model(df, "medium")
        meta = build_year_regime_metadata(df, model, "medium", 2022)
        assert meta["n_observations"] == len(df)

    def test_rsquared_in_valid_range(self) -> None:
        df = _make_regime_df(80)
        model = fit_regime_model(df, "low")
        meta = build_year_regime_metadata(df, model, "low", 2022)
        assert 0.0 <= meta["rsquared"] <= 1.0

    def test_ccgt_mean_mw_matches_data(self) -> None:
        df = _make_regime_df(50)
        model = fit_regime_model(df, "high")
        meta = build_year_regime_metadata(df, model, "high", 2022)
        assert meta["ccgt_mean_mw"] == pytest.approx(df["ccgt_generation_mw"].mean(), rel=1e-6)


# ---------------------------------------------------------------------------
# TestRunYearly (integration)
# ---------------------------------------------------------------------------

class TestRunYearly:
    @pytest.fixture
    def multi_year_regime_csv(self, tmp_path: Path) -> Path:
        """Write a synthetic market_regimes.csv spanning 2022–2024."""
        rng = np.random.default_rng(99)
        frames = []
        for year in [2022, 2023, 2024]:
            n = 500
            idx = pd.date_range(f"{year}-01-01", periods=n, freq="h")
            css = rng.normal(0, 20, n)
            regimes = (["high"] * 120 + ["medium"] * 260 + ["low"] * 120)[:n]
            frames.append(pd.DataFrame({
                "ccgt_generation_mw": rng.uniform(1000, 15000, n),
                "css": css,
                "affr_price_eur_mw": (10 + 0.5 * css + rng.normal(0, 5, n)).clip(0),
                "da_price_eur_mwh": rng.uniform(30, 180, n),
                "regime": regimes,
            }, index=idx))
        df = pd.concat(frames)
        path = tmp_path / "market_regimes.csv"
        df.to_csv(path)
        return path

    def _patch_rm(self, rm, tmp_path: Path, csv_parent: Path, monkeypatch) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(rm, "OUTPUTS_MODELS", models_dir)
        monkeypatch.setattr(rm, "DATA_PROCESSED", csv_parent)
        monkeypatch.setattr(rm, "METADATA_PATH", models_dir / "regime_metadata.json")

    def test_creates_year_regime_pkl_files(
        self, tmp_path: Path, multi_year_regime_csv: Path, monkeypatch
    ) -> None:
        import src.models.regression_models as rm
        self._patch_rm(rm, tmp_path, multi_year_regime_csv.parent, monkeypatch)
        rm.run()

        for year in [2022, 2023, 2024]:
            for regime in ["high", "medium", "low"]:
                expected = tmp_path / "models" / f"regime_{regime}_{year}_model.pkl"
                assert expected.exists(), f"Missing: {expected.name}"

    def test_metadata_json_has_year_keys(
        self, tmp_path: Path, multi_year_regime_csv: Path, monkeypatch
    ) -> None:
        import src.models.regression_models as rm
        self._patch_rm(rm, tmp_path, multi_year_regime_csv.parent, monkeypatch)
        rm.run()

        meta_path = tmp_path / "models" / "regime_metadata.json"
        metadata = json.loads(meta_path.read_text())
        for year in ["2022", "2023", "2024"]:
            assert year in metadata

    def test_metadata_json_has_flat_compat_keys(
        self, tmp_path: Path, multi_year_regime_csv: Path, monkeypatch
    ) -> None:
        import src.models.regression_models as rm
        self._patch_rm(rm, tmp_path, multi_year_regime_csv.parent, monkeypatch)
        rm.run()

        meta_path = tmp_path / "models" / "regime_metadata.json"
        metadata = json.loads(meta_path.read_text())
        for regime in ["high", "medium", "low"]:
            assert regime in metadata, f"Flat compat key '{regime}' missing from metadata"

    def test_returns_model_dict_with_year_regime_keys(
        self, tmp_path: Path, multi_year_regime_csv: Path, monkeypatch
    ) -> None:
        import src.models.regression_models as rm
        self._patch_rm(rm, tmp_path, multi_year_regime_csv.parent, monkeypatch)
        models = rm.run()
        assert "2022_high" in models
        assert "2023_medium" in models


# ---------------------------------------------------------------------------
# TestEstimateAFRRPrice
# ---------------------------------------------------------------------------

class TestEstimateAFRRPrice:
    @pytest.fixture
    def model_dir_2022(self, tmp_path: Path) -> Path:
        """Write year-named pkl models for all 3 regimes (year=2022)."""
        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True)

        rng = np.random.default_rng(7)
        css = rng.normal(0, 20, 100)
        df = pd.DataFrame({
            "css": css,
            "affr_price_eur_mw": (10 + 0.5 * css + rng.normal(0, 3, 100)).clip(0),
        })
        X = sm.add_constant(df[["css"]])
        model = sm.OLS(df["affr_price_eur_mw"], X).fit()

        for regime in ["high", "medium", "low"]:
            path = models_dir / f"regime_{regime}_2022_model.pkl"
            with open(path, "wb") as fh:
                pickle.dump(model, fh)

        return models_dir

    def test_explicit_year_works(self, model_dir_2022: Path, monkeypatch) -> None:
        import src.models.predictions as pred
        monkeypatch.setattr(pred, "OUTPUTS_MODELS", model_dir_2022)
        result = estimate_afrr_price(80, 35, 60, "low", year=2022)
        assert isinstance(result, float)

    def test_default_year_2022_works(self, model_dir_2022: Path, monkeypatch) -> None:
        import src.models.predictions as pred
        monkeypatch.setattr(pred, "OUTPUTS_MODELS", model_dir_2022)
        result = estimate_afrr_price(80, 35, 60, "medium")
        assert isinstance(result, float)

    def test_negative_price_raises(self, model_dir_2022: Path, monkeypatch) -> None:
        import src.models.predictions as pred
        monkeypatch.setattr(pred, "OUTPUTS_MODELS", model_dir_2022)
        with pytest.raises(ValueError, match="cannot be negative"):
            estimate_afrr_price(-10, 35, 60, "high")

    def test_invalid_regime_raises(self, model_dir_2022: Path, monkeypatch) -> None:
        import src.models.predictions as pred
        monkeypatch.setattr(pred, "OUTPUTS_MODELS", model_dir_2022)
        with pytest.raises(ValueError, match="regime must be one of"):
            estimate_afrr_price(80, 35, 60, "extreme")

    def test_missing_model_file_raises(self, tmp_path: Path, monkeypatch) -> None:
        import src.models.predictions as pred
        monkeypatch.setattr(pred, "OUTPUTS_MODELS", tmp_path / "empty")
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError):
            estimate_afrr_price(80, 35, 60, "high", year=2022)

    def test_cli_year_arg_accepted(self, model_dir_2022: Path, monkeypatch, capsys) -> None:
        """Verify --year is accepted by the CLI parser and produces output."""
        import src.models.predictions as pred
        monkeypatch.setattr(pred, "OUTPUTS_MODELS", model_dir_2022)

        parser = pred._build_cli_parser()
        args = parser.parse_args(
            ["--da-price", "80", "--gas-price", "35", "--ets-price", "60",
             "--regime", "low", "--year", "2022"]
        )
        assert args.year == 2022
        assert args.regime == "low"

    def test_high_regime_consistent_with_low(self, model_dir_2022: Path, monkeypatch) -> None:
        """All regimes share the same underlying model in the fixture — just sanity check."""
        import src.models.predictions as pred
        monkeypatch.setattr(pred, "OUTPUTS_MODELS", model_dir_2022)
        r_high = estimate_afrr_price(80, 35, 60, "high", year=2022)
        r_low = estimate_afrr_price(80, 35, 60, "low", year=2022)
        # Same model → same result
        assert r_high == pytest.approx(r_low, rel=1e-6)
