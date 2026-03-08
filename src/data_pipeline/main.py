"""
Data pipeline entry point.

Orchestrates the full pipeline:
    1. Load all 5 raw datasets (from API/web or cached CSVs)
    2. Clean & align to hourly UTC
    3. Engineer features (CSS, time columns)
    4. Save processed outputs
    5. Log summary statistics

Usage:
    python src/data_pipeline/main.py

Environment variables required:
    ENTSOE_API_TOKEN — ENTSO-E Transparency API token

Config toggle (src/utils/config.py):
    FORCE_REDOWNLOAD = True   # Re-fetch from APIs even if raw CSVs exist
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path when run as a script
PROJECT_ROOT = Path(__file__).parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline.data_cleaner import clean_dataset
from src.data_pipeline.data_loader import (
    AFRRBidPriceLoader,
    ENTSOEClient,
    EUETSLoader,
    GasPriceLoader,
)
from src.data_pipeline.feature_engineer import engineer_features
from src.utils.config import (
    DATA_PROCESSED,
    DATA_RAW,
    ENTSOE_API_TOKEN,
    FORCE_REDOWNLOAD,
)
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def _load_or_error(path: Path, name: str) -> pd.Series:
    """
    Load a cached series from CSV, or raise a clear error if it doesn't exist.

    Used as fallback when API clients are skipped.
    """
    if path.exists():
        logger.info("Loading cached %s from %s", name, path.name)
        return pd.read_csv(path, index_col=0, parse_dates=True).squeeze()
    raise FileNotFoundError(
        f"Raw data file not found: {path}\n"
        f"Either set the required credentials and re-run, or manually place\n"
        f"the data at {path}.\nSee data/references/data_sources.md for details."
    )


def run_pipeline() -> pd.DataFrame:
    """
    Execute the full data collection and processing pipeline.

    Returns:
        The final processed DataFrame with all features.
    """
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("aFRR Opportunity Cost — Data Pipeline")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Phase 1: Load raw data
    # ------------------------------------------------------------------
    logger.info("Phase 1: Loading raw datasets …")

    # --- ENTSO-E (DA prices + CCGT generation) ---
    da_path = DATA_RAW / "da_prices.csv"
    ccgt_path = DATA_RAW / "ccgt_generation.csv"

    if ENTSOE_API_TOKEN and (FORCE_REDOWNLOAD or not da_path.exists() or not ccgt_path.exists()):
        try:
            entsoe = ENTSOEClient()
            da_prices = entsoe.fetch_da_prices()
            ccgt_generation = entsoe.fetch_ccgt_generation()
        except Exception as exc:
            logger.error("ENTSO-E fetch failed: %s", exc)
            logger.info("Falling back to cached files …")
            da_prices = _load_or_error(da_path, "DA prices")
            ccgt_generation = _load_or_error(ccgt_path, "CCGT generation")
    else:
        if not ENTSOE_API_TOKEN:
            logger.warning(
                "ENTSOE_API_TOKEN not set — skipping API fetch, loading from cache."
            )
        da_prices = _load_or_error(da_path, "DA prices")
        ccgt_generation = _load_or_error(ccgt_path, "CCGT generation")

    # --- aFRR prices (static CSV) ---
    affr_path = DATA_RAW / "affr_prices.csv"
    if FORCE_REDOWNLOAD or not affr_path.exists():
        loader = AFRRBidPriceLoader()
        affr_prices = loader.load()
    else:
        affr_prices = _load_or_error(affr_path, "aFRR prices")

    # --- EU-ETS (static CSV — must be manually downloaded) ---
    ets_loader = EUETSLoader()
    eu_ets_prices = ets_loader.load()

    # --- Gas prices (static CSV — must be manually downloaded) ---
    gas_loader = GasPriceLoader()
    gas_prices = gas_loader.load()

    # ------------------------------------------------------------------
    # Phase 2: Clean & align
    # ------------------------------------------------------------------
    logger.info("Phase 2: Cleaning and aligning …")
    df_clean = clean_dataset(
        da_prices=da_prices,
        affr_prices=affr_prices,
        ccgt_generation=ccgt_generation,
        eu_ets_prices=eu_ets_prices,
        gas_prices=gas_prices,
    )

    # ------------------------------------------------------------------
    # Phase 3: Feature engineering
    # ------------------------------------------------------------------
    logger.info("Phase 3: Engineering features …")
    df_features = engineer_features(df_clean)

    # ------------------------------------------------------------------
    # Phase 4: Save outputs
    # ------------------------------------------------------------------
    logger.info("Phase 4: Saving processed data …")

    combined_path = DATA_PROCESSED / "combined_dataset.csv"
    df_features.to_csv(combined_path)
    logger.info("Saved combined dataset: %s (%d rows, %d cols)",
                combined_path, len(df_features), len(df_features.columns))

    # ------------------------------------------------------------------
    # Phase 5: Summary statistics
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Pipeline complete. Summary statistics:")
    logger.info("=" * 60)
    summary = df_features[
        ["da_price_eur_mwh", "affr_price_eur_mw", "ccgt_generation_mw",
         "eu_ets_price_eur_tco2", "gas_price_eur_mwh_th", "css"]
    ].describe()

    for col in summary.columns:
        logger.info(
            "%s → mean=%.2f, std=%.2f, min=%.2f, max=%.2f",
            col,
            summary.loc["mean", col],
            summary.loc["std", col],
            summary.loc["min", col],
            summary.loc["max", col],
        )

    missing_rates = df_features.isna().mean() * 100
    logger.info("Missing rates (%%): \n%s", missing_rates[missing_rates > 0].to_string())

    return df_features


if __name__ == "__main__":
    df = run_pipeline()
    print("\nFirst 5 rows of combined dataset:")
    print(df.head().to_string())
    print(f"\nShape: {df.shape}")
