# aFRR Opportunity Cost Calculator

Quantifies opportunity costs for gas peaker plants (CCGT) participating in aFRR spin-up (balancing) markets in the DE-LU bidding zone. Uses OLS regression to model the relationship between Clean Spark Spread (CSS) and aFRR-up capacity prices across three market regime conditions (High / Medium / Low gas generation).

> The motivation and initial vision for this analysis is documented in \[`Resources/Vision.odt`](Resources/Vision.odt).

\---

## Background

Gas prices are a key driver of opportunity costs for gas peakers bidding into aFRR markets. When CSS (a proxy for plant profitability) rises, plants demand higher compensation to withhold capacity for balancing. This sensitivity also depends on how many gas plants are online — a "regime" effect that this model captures via three fixed MW thresholds derived from K-Means clustering of historical CCGT generation data.

This project covers the period **2020–2026**, selected to isolate CCGT-dominated aFRR price formation (post-EU-ETS carbon pricing, pre-BESS boom).

> Side scripts used for preprocessing (aFRR bid aggregation, regime centroid derivation via K-Means) are not included in this repository for brevity. Their outputs are committed as reference files in `data/references/`.

\---

## Prerequisites

* Python 3.10
* ENTSO-E API token (for DA prices and CCGT generation data)

```bash
# Clone the repo, then:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set ENTSO-E API token
export ENTSOE\_API\_TOKEN="your\_token\_here"
```

\---

## Project Structure

```
global\_tech\_challenge/
├── data/
│   ├── raw/                        # Source CSVs (DA, aFRR, ETS, gas, CCGT)
│   ├── processed/                  # combined\_dataset.csv, market\_regimes.csv
│   └── references/                 # Data source docs, regime centroid JSONs
├── src/
│   ├── data\_pipeline/              # Data loading, cleaning, feature engineering
│   ├── models/                     # Regime classifier, OLS models, predictions CLI
│   ├── analysis/                   # Visualizations and HTML report
│   └── utils/                      # Config constants, logging
├── tests/                          # 92 unit tests (pytest)
├── outputs/
│   ├── models/                     # Pickled OLS models + regime\_metadata.json
│   ├── plots/                      # PNG visualisations and forecast curves
│   └── reports/                    # market\_regime\_report.html
├── Resources/                      # Vision doc, report template
└── Claude Prompts/                 # LLM prompts used during analysis development
```

\---

## Phase 1 — Data Collection \& Preparation

### Data Sources

|Dataset|Source|Granularity|File|
|-|-|-|-|
|DA Prices|ENTSO-E API (`entsoe-py`)|Hourly|`data/raw/da\_prices.csv`|
|CCGT Generation|ENTSO-E API (`entsoe-py`)|Hourly|`data/raw/ccgt\_generation.csv`|
|aFRR Prices|regelleistung.net (4-hour blocks → forward-filled hourly)|Hourly|`data/raw/affr\_prices.csv`|
|EU-ETS Carbon Price|Yearly average — used as constant per year|Yearly|`data/raw/eu\_ets\_prices.csv`|
|Gas Prices|Dutch TTF Natural Gas Futures (Investing.com)|Daily → forward-filled|`data/raw/gas\_prices.csv`|

**Notes:**

* 2020 CCGT generation data returned near-zero values from ENTSO-E; SMARD.de data was used for that year instead.
* aFRR prices were aggregated from raw 4-hour capacity bid data using a capacity-weighted average (€/MW) via a preprocessing script not included in this repo.
* For manual download instructions and accepted column name formats, see [`data/references/data\_sources.md`](data/references/data_sources.md).

### Run the pipeline

```bash
python src/data\_pipeline/main.py
```

**Outputs:**

* `data/processed/combined\_dataset.csv` — all features merged into one file

\---

## Phase 2 — Modelling \& Coefficient Generation

### Market Regimes

Each one hour interval is classified into one of three regimes based on total CCGT generation using fixed MW thresholds derived from K-Means clustering:

|Regime|CCGT Generation|Reserve Supply|aFRR Price Sensitivity|
|-|-|-|-|
|**Low**|< 5,858 MW|Scarce|High|
|**Medium**|5,858 – 10,546 MW|Balanced|Moderate|
|**High**|≥ 10,546 MW|Abundant|Low|

K-Means centroids: Low ≈ 3,776 MW, Medium ≈ 7,941 MW, High ≈ 13,150 MW.
Centroid reference files: `data/references/regime\_kmeans.json`, `data/references/regime\_capacity.json`.
Toggle method in `src/utils/config.py`: `REGIME\_CLASSIFICATION\_METHOD = "kmeans"` or `"capacity"`.

### OLS Model

For each year × regime combination, a simple OLS model is fitted:

```
aFRR\_Price = β₀ + β₁·CSS + ε

where:
  CSS = DA\_Price − (Gas\_Price / η) − (ETS\_Price × 0.202 / η)
  η   = 0.50  (Thermal efficiency)
```

Up to 21 models are produced (7 years × 3 regimes).

```bash
python src/models/regime\_classifier.py
python src/models/regression\_models.py
```

**Outputs:**

* `data/processed/market\_regimes.csv` — PTU-level regime labels
* `outputs/models/regime\_{regime}\_{year}\_model.pkl` — fitted OLS models (e.g. `regime\_low\_2022\_model.pkl`)
* `outputs/models/regime\_metadata.json` — R², Durbin-Watson, coefficients per year × regime

\---

## Phase 3 — Evaluation \& Forecasting

### Visualisations \& Report

```bash
python src/analysis/visualization.py
python src/analysis/report\_generator.py
```

**Outputs — Plots (`outputs/plots/`):**

* `historical\_afrr\_prices.png` — aFRR time series coloured by market regime
* `css\_vs\_affr\_prices.png` — CSS vs aFRR scatter (coloured by regime)
* `regime\_distribution.png` — stacked bar chart of regime share per year
* `forecast\_curves\_jan\_w1.png` … `forecast\_curves\_jan\_w5.png` — Jan 2026 weekly forecast curves
* `forecast\_curves\_feb\_w1.png` … `forecast\_curves\_feb\_w4.png` — Feb 2026 weekly forecast curves

**Outputs — Reports (`outputs/reports/`):**

* `market\_regime\_report.html` — narrative HTML report (5 sections: regime definitions → model fitting → model insights → forecasts → forecast insights)

### Opportunity Cost Calculator (CLI)

Estimate the aFRR-up bid price for given market inputs:

```bash
python src/models/predictions.py \\
    --da-price 80 \\
    --gas-price 35 \\
    --ets-price 60 \\
    --regime low \\
    --year 2022
```

|Argument|Unit|Description|
|-|-|-|
|`--da-price`|€/MWh|Day-ahead electricity price|
|`--gas-price`|€/MWh\_th|Gas forward price (thermal)|
|`--ets-price`|€/tCO₂|EU-ETS carbon allowance price|
|`--regime`|—|`high`, `medium`, or `low`|
|`--year`|—|Model year to use (default: 2022)|

Or call from Python:

```python
from src.models.predictions import estimate\_afrr\_price

price = estimate\_afrr\_price(da\_price=80, gas\_price=35, eu\_ets\_price=60, regime="low", year=2022)
print(f"Estimated aFRR price: {price:.2f} €/MW")
```

\---

## Testing

```bash
# Run all 92 tests
venv/bin/pytest tests/ -v

# With coverage report
venv/bin/pytest tests/ --cov=src/ --cov-report=html
```

|Test file|Coverage|
|-|-|
|`tests/test\_data\_pipeline.py`|Data loading, cleaning, feature engineering (37 tests)|
|`tests/test\_models.py`|Regime classifier, OLS models, predictions (39 tests)|
|`tests/test\_analysis.py`|Visualizations, report (16 tests)|

\---

## Limitations

* **BESS excluded:** Battery storage bids are not modelled. Results reflect CCGT-only opportunity costs — a partial view of aFRR price formation.
* **Historical model:** Coefficients fitted on 2020–2026 data. Structural market changes (e.g. new BESS entrants, grid topology changes) may invalidate extrapolations.

\---

## Claude Prompts

LLM prompts used during the development of this analysis are stored in [`Claude Prompts/`](Claude%2520Prompts/).

## Disclosure

I have had to edit and add inputs to files developed by Claude. Not because it did not do what I intended but also because I learned faults in my analysis and assumptions through its executions. Hand written edits are, to name a few instances, present in codes/reports to reflect details I wanted to personally include. Like this one.



