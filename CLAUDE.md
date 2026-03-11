# .CLAUDE.md - Global Tech Challenge Project

## Project Overview
This project quantifies the opportunity costs for gas peaker plants participating in aFRR (Automated Frequency Restoration Reserve) spin-up markets under varying market conditions. The analysis uses linear regression models to estimate the relationship between Clean Spark Spread (CSS) and aFRR-up prices across three distinct market regimes (high, medium, low gas generation).

**Core Objective:** Build a calculator that estimates aFRR-up capacity bid prices as a function of gas forward prices, DA forecasts, and market regime conditions.

---

## Project Structure

```
global-tech-challenge/
│
├── .CLAUDE.md                          # Claude Code context file
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore rules
│
├── data/
│   ├── raw/
│   │   ├── da_prices.csv               # Day-ahead market prices
│   │   ├── affr_prices.csv             # aFRR spin-up market prices (pre-aggregated hourly)
│   │   ├── AFRR_Bid_Prices_2020-2026.csv  # Raw bid-level aFRR source
│   │   ├── ccgt_generation.csv         # CCGT generation per PTU
│   │   ├── eu_ets_prices.csv           # Carbon allowance prices
│   │   └── gas_prices.csv              # Gas forward prices
│   ├── processed/
│   │   ├── combined_dataset.csv        # Merged all features
│   │   └── market_regimes.csv          # PTU regime classifications
│   └── references/
│       ├── data_sources.md             # External data source documentation
│       ├── data_dictionary.csv         # Column definitions
│       ├── regime_kmeans.json          # K-Means cluster centroids (low/medium/high MW)
│       └── regime_capacity.json        # Capacity-thirds centroids (alternative method)
│
├── src/
│   ├── __init__.py
│   │
│   ├── data_pipeline/
│   │   ├── __init__.py
│   │   ├── data_loader.py              # Load from external sources (ENTSO-E, Regelleistung, ETS, Gas)
│   │   ├── data_cleaner.py             # Handle missing values, outliers
│   │   ├── feature_engineer.py         # Create CSS, spreads, etc.
│   │   └── main.py                     # Execute full pipeline
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── regime_classifier.py        # Classify PTUs into regimes (centroid-based MW thresholds)
│   │   ├── regression_models.py        # Fit OLS models per year × regime (21 models)
│   │   └── predictions.py              # Generate opportunity cost estimates
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── visualization.py            # Plots for regime conditions
│   │   └── report_generator.py         # Automated report generation
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py                   # Configuration constants
│       └── logging_setup.py            # Logging utilities
│
├── tests/
│   ├── __init__.py
│   ├── test_data_pipeline.py           # Test data loading/cleaning (37 tests)
│   ├── test_models.py                  # Test regime classifier, regression models, predictions (39 tests)
│   └── test_analysis.py                # Test analysis functions and report (13 tests)
│
├── outputs/
│   ├── models/                         # Saved model files (pickle)
│   │   ├── regime_high_<year>_model.pkl     # One per year (2020–2026)
│   │   ├── regime_medium_<year>_model.pkl
│   │   ├── regime_low_<year>_model.pkl
│   │   └── regime_metadata.json             # Nested by year + flat compat keys
│   ├── plots/                          # Generated visualizations
│   │   ├── historical_afrr_prices.png       # aFRR time series coloured by regime
│   │   ├── css_vs_affr_prices.png           # CSS vs aFRR scatter (coloured by regime)
│   │   ├── regime_distribution.png          # Stacked bar of regime share per year
│   │   ├── forecast_curves_jan_w1.png       # Jan 2026 weekly forecast (W1–W5)
│   │   ├── forecast_curves_jan_w2.png
│   │   ├── forecast_curves_jan_w3.png
│   │   ├── forecast_curves_jan_w4.png
│   │   ├── forecast_curves_jan_w5.png
│   │   ├── forecast_curves_feb_w1.png       # Feb 2026 weekly forecast (W1–W4)
│   │   ├── forecast_curves_feb_w2.png
│   │   ├── forecast_curves_feb_w3.png
│   │   └── forecast_curves_feb_w4.png
│   ├── reports/
│   │   └── market_regime_report.html   # Narrative HTML report
│   └── data/
│
└── venv/                               # Virtual environment (not committed)
```

---

## Key Concepts & Domain Knowledge

### Clean Spark Spread (CSS)
```
carbon_cost = eu_ets_price * CARBON_INTENSITY / EFFICIENCY   # (0.202 / 0.50)
CSS = DA_Price - (Gas_Price / EFFICIENCY) - carbon_cost
```
- Represents the **opportunity cost** for gas peaker plants
- Higher CSS = higher profitability of gas generation
- Used as primary predictor for aFRR bid behavior

### aFRR Spin-up Markets
- Automated Frequency Restoration Reserve market
- Balancing capacity market where generators offer reserve capacity
- Gas peakers are key participants
- Prices vary based on reserve scarcity and plant opportunity costs

### Market Regimes (3 classifications)

Regimes are assigned using **fixed MW thresholds** derived from K-Means centroids
(default) or capacity-based thirds. Boundaries are the midpoints between adjacent centroids.

| Regime | K-Means Centroid | Boundary (K-Means) | Reserve Supply | aFRR Price Sensitivity |
|--------|------------------|--------------------|----------------|------------------------|
| **Low** | ~3,776 MW | < 5,858 MW | Scarce | High CSS sensitivity |
| **Medium** | ~7,941 MW | 5,858 – 10,546 MW | Balanced | Moderate sensitivity |
| **High** | ~13,150 MW | ≥ 10,546 MW | Abundant | Low CSS sensitivity |

**Rationale:** Fewer plants online = fewer reserve providers = higher prices with steeper CSS correlation.

Toggle classification method in `config.py`: `REGIME_CLASSIFICATION_METHOD = "kmeans"` or `"capacity"`.

---

## Technical Requirements

### Python Stack
- **Python:** 3.10 (venv at `venv/`)
- **Data:** pandas, NumPy
- **Modeling:** statsmodels (OLS)
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Notebooks:** Jupyter
- **Testing:** pytest, pytest-cov

### Installation
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Development Workflow

### Phase 1: Data Pipeline ✅
```bash
python src/data_pipeline/main.py
```
**Outputs:**
- `data/processed/combined_dataset.csv` — all features merged
- Summary statistics in logs

### Phase 2: Model Development ✅
```bash
python src/models/regime_classifier.py
python src/models/regression_models.py
```
**Outputs:**
- `data/processed/market_regimes.csv` — PTU regime labels
- `outputs/models/regime_{regime}_{year}_model.pkl` — 21 pkl files (7 years × 3 regimes)
- `outputs/models/regime_metadata.json` — per-year nested stats + flat compat keys from most recent complete year

### Phase 3: Analysis & Visualization ✅
```bash
python src/analysis/visualization.py
python src/analysis/report_generator.py
```
**Outputs:**
- `outputs/plots/historical_afrr_prices.png` — aFRR time series coloured by regime
- `outputs/plots/css_vs_affr_prices.png` — CSS vs aFRR scatter (coloured by regime)
- `outputs/plots/regime_distribution.png` — stacked bar of regime share per year
- `outputs/plots/forecast_curves_{month}_w{N}.png` — weekly forecast plots (Jan W1–W5, Feb W1–W4)
- `outputs/reports/market_regime_report.html` — narrative HTML report (5 sections: definitions → model fitting → model insights → forecasts → forecast insights)

### Testing & Validation
```bash
# Run all tests (92 total)
venv/bin/pytest tests/ -v

# Run with coverage
venv/bin/pytest tests/ --cov=src/ --cov-report=html

# Run specific test file
venv/bin/pytest tests/test_models.py -v
```

---

## Model Specification

### Regression Model Template
```
aFRR_Price = β₀ + β₁·CSS + ε
```

**Fitted separately for each year × regime** (21 models total) to capture both
regime-level and inter-year variation.

**Rationale for single-predictor design:**
- DA price is already embedded in CSS — including it separately would double-count
- Market regimes already isolate CCGT generation effects → CCGT_Gen dropped as predictor

### Model File Naming
```
outputs/models/regime_{regime}_{year}_model.pkl
  e.g. regime_high_2022_model.pkl
       regime_medium_2023_model.pkl
       regime_low_2024_model.pkl
```

### Metadata JSON Structure
```json
{
  "2022": {
    "high":   { "beta_0": ..., "beta_1_css": ..., "rsquared": ..., "durbin_watson": ..., ... },
    "medium": { ... },
    "low":    { ... }
  },
  "2023": { ... },
  "high":   { ... },   // flat compat keys = most recent complete year
  "medium": { ... },
  "low":    { ... }
}
```

### Validation Metrics
- **R² > 0.50** target per model
- **Residuals:** Normally distributed, homoscedastic
- **Durbin-Watson:** 1.8–2.2 target (DW < 1 = positive autocorrelation — flagged in report)

---

## Data Sources & Access

| Dataset | Source | Format | Notes |
|---------|--------|--------|-------|
| DA Prices | ENTSO-E API (`entsoe-py`) | hourly CSV | `export ENTSOE_API_TOKEN=...` |
| aFRR Prices | Static CSV (2020–2026) | hourly CSV | `data/raw/affr_prices.csv` |
| CCGT Generation | ENTSO-E API | hourly CSV | same token |
| EU-ETS Prices | Ember Climate (manual) | daily CSV | `data/raw/eu_ets_prices.csv` |
| Gas Prices | energy-charts.info (manual) | weekly CSV | `data/raw/gas_prices.csv` |

See `data/references/data_sources.md` for detailed access instructions.

---

## Important Limitations & Assumptions

1. **CCGT-focused analysis:**
   - Model captures only CCGT player actions
   - BESS impact is excluded (requires separate model)
   - Results don't predict absolute aFRR prices, only CSS sensitivity

2. **Historical analysis:**
   - Model fitted on historical data ≠ future predictions without caution
   - Market structure changes (e.g., new BESS entrants) invalidate model

3. **Linear relationship assumption:**
   - Assumes linear CSS-aFRR relationship within regimes
   - Non-linear effects possible during extreme market conditions

4. **Data quality:**
   - Missing data → interpolation strategy documented in `data_cleaner.py`
   - Outliers → removal criteria in `feature_engineer.py`

---

## Key Deliverables

### 1. Market Regime Classifier ✅
- **Input:** CCGT generation time series
- **Output:** Regime labels (High/Medium/Low) per PTU using fixed MW thresholds
- **File:** `src/models/regime_classifier.py`
- **Reference data:** `data/references/regime_kmeans.json`, `regime_capacity.json`

### 2. Regression Models (21 models) ✅
- **Format:** `outputs/models/regime_{regime}_{year}_model.pkl`
- **Coverage:** 7 years × 3 regimes = up to 21 models
- **Sidecar:** `outputs/models/regime_metadata.json` (nested by year + flat compat keys)

### 3. Opportunity Cost Calculator ✅
- **Type:** Python function + CLI tool
- **Inputs:** `da_price`, `gas_price`, `eu_ets_price`, `regime`, `year` (default 2022)
- **Output:** Estimated aFRR-up opportunity cost (€/MW)
- **File:** `src/models/predictions.py`
- **API:** `estimate_afrr_price(da_price, gas_price, eu_ets_price, regime, year=2022)`
- **CLI:** `python src/models/predictions.py --da-price 80 --gas-price 35 --ets-price 60 --regime low --year 2022`

### 4. Analysis Report ✅
- **Report:** `outputs/reports/market_regime_report.html` — 5 sections: definitions → model fitting → model insights → forecasts → forecast insights

### 5. Visualizations (12 plots) ✅
- `historical_afrr_prices.png` — aFRR time series coloured by market regime
- `css_vs_affr_prices.png` — CSS vs aFRR prices scatter plot (colored by regime)
- `regime_distribution.png` — stacked bar of regime share per year
- `forecast_curves_jan_w1.png` … `forecast_curves_jan_w5.png` — Jan 2026 weekly forecast (W1–W5)
- `forecast_curves_feb_w1.png` … `forecast_curves_feb_w4.png` — Feb 2026 weekly forecast (W1–W4)

---

## Testing Strategy

### Unit Tests (89 total, all passing)
```bash
venv/bin/pytest tests/test_data_pipeline.py -v   # 37 tests: data loading/cleaning
venv/bin/pytest tests/test_models.py -v           # 39 tests: regime classifier, regression, predictions
venv/bin/pytest tests/test_analysis.py -v         # 13 tests: visualization and report
```

### What is Tested
- **Data Pipeline:** Missing values, outliers, merges, AFRR loader
- **Regime Classifier:** `load_centroids`, `compute_thresholds`, `assign_regimes` boundary logic, method toggle
- **Regression Models:** `fit_regime_model`, `save_model`, `build_year_regime_metadata`, per-year `run()` integration
- **Predictions:** `estimate_afrr_price` with explicit/default year, CLI `--year` arg, error handling
- **Analysis:** PNG plots, HTML report content

### Coverage Target
- Minimum **80% code coverage**
- Every public function tested
- Edge cases documented in test docstrings

---

## Useful Commands

```bash
# Full pipeline (all 3 phases)
python src/data_pipeline/main.py
python src/models/regime_classifier.py
python src/models/regression_models.py
python src/analysis/visualization.py
python src/analysis/report_generator.py

# Predict opportunity cost (CLI)
python src/models/predictions.py \
    --da-price 80 --gas-price 35 --ets-price 60 --regime low --year 2022

# Run all tests
venv/bin/pytest tests/ -v --cov=src/
```

---

## Configuration

Key settings in `src/utils/config.py`:

```python
BIDDING_ZONE = "DE_LU"

# Analysis window (full available range)
ANALYSIS_START = "2020-01-01"
ANALYSIS_END   = "2026-03-09"

# CCGT plant assumptions
EFFICIENCY       = 0.50    # 50% thermal efficiency
CARBON_INTENSITY = 0.202   # tCO₂ per MWh_thermal

# Regime classification
REGIME_CLASSIFICATION_METHOD = "kmeans"  # or "capacity"
REGIME_KMEANS_JSON   = DATA_REFERENCES / "regime_kmeans.json"
REGIME_CAPACITY_JSON = DATA_REFERENCES / "regime_capacity.json"

# (Legacy percentile thresholds — kept for documentation only)
REGIME_THRESHOLDS = {"high": 0.75, "medium": 0.25, "low": 0.00}

# Model hyperparameters
OLS_FIT_METHOD = "pinv"   # Pseudo-inverse for numerical robustness
MIN_OBSERVATIONS = 20     # Minimum rows per (year, regime) to fit a model

# Data validation
MAX_MISSING_RATE   = 0.05   # Max 5% missing values per column
OUTLIER_THRESHOLD  = 3.0    # Z-score threshold

# Forecast scenario constants (used by predictions.py and weekly forecast plots)
FORECAST_ETS_PRICE      = 65.0                      # Fixed EU-ETS (€/tCO₂)
FORECAST_GAS_MIN        = 20.0                       # €/MWh thermal
FORECAST_GAS_MAX        = 100.0                      # €/MWh thermal
FORECAST_GAS_STEPS      = 100
FORECAST_DA_SCENARIOS   = [50.0, 80.0, 120.0, 160.0]  # €/MWh
```

---

## References & Links

- **ENTSO-E Data:** https://transparency.entsoe.eu/
- **EU-ETS:** https://ec.europa.eu/clima/ets/
- **Statsmodels OLS:** https://www.statsmodels.org/stable/regression.html
- **Market Design:** ENTSO-E Network Codes on Electricity Balancing

---

## Contact & Updates

- **Project Owner:** [Your Name]
- **Version:** 2.0.0 (Phase 2 overhaul — per-year models, centroid-based regime classification)

---

**This file is committed to version control so the entire team benefits from these guidelines.**
