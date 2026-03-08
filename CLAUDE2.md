# CLAUDE2.md — aFRR Opportunity Cost Calculator

## Purpose
Quantify opportunity costs for German/DE-LU gas peaker plants (CCGTs) participating
in aFRR spin-up capacity markets. Fits per-regime OLS models (high/medium/low CCGT
generation) relating Clean Spark Spread to aFRR-up bid prices.

---

## Tech Stack
| Layer | Libraries |
|---|---|
| Data | pandas ≥2.0, numpy ≥1.24 |
| APIs | entsoe-py ≥0.6, requests ≥2.31 |
| Modelling | statsmodels ≥0.14, scikit-learn ≥1.3 |
| Viz | matplotlib, seaborn, plotly + kaleido |
| Testing | pytest ≥7.4, pytest-cov ≥4.1 |
| Config | python-dotenv ≥1.0 |

Python 3.10+. Virtual env at `venv/`.

---

## Key Directories
| Path | Purpose |
|---|---|
| `src/utils/config.py` | Single source of truth for all constants and env vars |
| `src/data_pipeline/` | Phase 1: load → clean → feature-engineer → save |
| `src/models/` | Phase 2: regime classifier + per-regime OLS (stub) |
| `src/analysis/` | Phase 3: visualizations, dashboard, reports (stub) |
| `data/raw/` | Downloaded CSVs (some manual, some auto-cached) |
| `data/processed/` | Cleaned + engineered DataFrames |
| `tests/` | 32 unit tests for data pipeline |
| `logs/pipeline.log` | Runtime log file |

---

## Environment Setup
```bash
source venv/bin/activate
export ENTSOE_API_TOKEN="<your-token>"   # required for DA/CCGT fetch
```

---

## Essential Commands
```bash
# Run full data pipeline
python src/data_pipeline/main.py

# Run all tests
venv/bin/pytest tests/ -v

# Run with coverage
venv/bin/pytest tests/ -v --cov=src/ --cov-report=html
```

---

## Data Sources — Manual Downloads Required
Two CSVs must be placed manually before running the pipeline:
- `data/raw/eu_ets_prices.csv` — Ember Climate (daily ETS prices)
- `data/raw/gas_prices.csv` — energy-charts.info (gas forward prices)

See `data/references/data_sources.md` for exact URLs and column formats.

---

## Phase Status
- **Phase 1 (data pipeline):** Complete — `src/data_pipeline/`
- **Phase 2 (models):** Pending — `src/models/`
- **Phase 3 (analysis):** Pending — `src/analysis/`

---

## Additional Documentation
Check these files when working in the relevant area:

| Topic | File |
|---|---|
| Architectural patterns & design decisions | `.claude/docs/architectural_patterns.md` |
| Data source URLs and column specs | `data/references/data_sources.md` |
| Full project spec & model equations | `CLAUDE.md` |
