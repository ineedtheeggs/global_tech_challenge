"""
Configuration constants for the aFRR Opportunity Cost Calculator.

All secrets are loaded from environment variables. Set them before running:
    export ENTSOE_API_TOKEN="your_token_here"

Or place them in a .env file in the project root (never committed).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (if it exists)
load_dotenv(Path(__file__).parents[2] / ".env")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_REFERENCES = PROJECT_ROOT / "data" / "references"
OUTPUTS_MODELS = PROJECT_ROOT / "outputs" / "models"
OUTPUTS_PLOTS = PROJECT_ROOT / "outputs" / "plots"
OUTPUTS_REPORTS = PROJECT_ROOT / "outputs" / "reports"
OUTPUTS_DATA = PROJECT_ROOT / "outputs" / "data"

# ---------------------------------------------------------------------------
# API credentials
# ---------------------------------------------------------------------------
ENTSOE_API_TOKEN: str | None = os.getenv("ENTSOE_API_TOKEN")

# ---------------------------------------------------------------------------
# Market parameters
# ---------------------------------------------------------------------------
BIDDING_ZONE = "DE_LU"              # ENTSO-E bidding zone code

ANALYSIS_START = "2021-01-01"       # CCGT data available from ~2021-01-09; NaN rows dropped
ANALYSIS_END = "2021-11-30"         # aFRR prices become forward-filled (unreliable) after this

# CCGT plant assumptions
EFFICIENCY = 0.50                   # 50% thermal efficiency
CARBON_INTENSITY = 0.202            # tCO₂ per MWh_thermal gas input

# ---------------------------------------------------------------------------
# Market regime thresholds (percentiles of CCGT generation)
# ---------------------------------------------------------------------------
REGIME_THRESHOLDS = {
    "high": 0.75,   # > 75th percentile → High generation regime
    "medium": 0.25, # 25th–75th percentile → Medium
    "low": 0.00,    # < 25th percentile → Low generation regime
}

# ---------------------------------------------------------------------------
# Data quality parameters
# ---------------------------------------------------------------------------
MAX_MISSING_RATE = 0.05     # Maximum fraction of missing values allowed
OUTLIER_THRESHOLD = 3.0     # Z-score threshold for outlier detection
MAX_INTERPOLATION_GAP = 2   # Max consecutive hours to interpolate (DA/CCGT)

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
OLS_FIT_METHOD = "pinv"     # Pseudo-inverse for numerical robustness
TEST_SIZE = 0.20
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Pipeline control
# ---------------------------------------------------------------------------
FORCE_REDOWNLOAD = False    # If True, re-fetch even if raw CSVs exist

# ---------------------------------------------------------------------------
# Phase 3 forecast scenario defaults
# ---------------------------------------------------------------------------
FORECAST_GAS_PRICE: float = 45.0      # €/MWh thermal (Mar 2026 TTF)
FORECAST_ETS_PRICE: float = 65.0      # €/tCO₂  (Mar 2026 EUA)
FORECAST_DA_MIN: float = 50.0         # €/MWh — set to week_min - 10
FORECAST_DA_MAX: float = 200.0        # €/MWh — set to week_max + 10
FORECAST_DA_STEPS: int = 100

# ---------------------------------------------------------------------------
# Regelleistung.net aFRR download settings
# ---------------------------------------------------------------------------
# Base URL for aFRR positive capacity (SRL-positive) data
REGELLEISTUNG_BASE_URL = "https://www.regelleistung.net/ext/data/"
REGELLEISTUNG_PRODUCT = "SRL"       # Secondary reserve (aFRR)
REGELLEISTUNG_DIRECTION = "positive"

# ---------------------------------------------------------------------------
# Pre-compiled aFRR bid prices CSV (replaces regelleistung.net download)
# ---------------------------------------------------------------------------
# Path to static CSV with individual bids per 4-hour product block (2020–2023)
AFRR_CSV_PATH = Path("/home/anurag/GitRepos/SideScripts/AFRR value finder/AFRR_Bid_Prices_2020-2023.csv")
