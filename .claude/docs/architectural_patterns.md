# Architectural Patterns

Patterns confirmed across multiple files in `src/`.

---

## 1. Centralized Configuration
All constants live in `src/utils/config.py` — never hardcoded elsewhere.
- Paths use `pathlib.Path` rooted at `PROJECT_ROOT = Path(__file__).parents[2]`
- `ENTSOE_API_TOKEN` loaded from env via `python-dotenv` at import time
- Boolean flag `FORCE_REDOWNLOAD` gates cache bypass across all loaders

Consumers import directly: e.g., `from src.utils.config import EFFICIENCY, ANALYSIS_START`

---

## 2. Logger Factory Pattern
Every module gets a named logger via the same factory:
```
logger = get_logger(__name__)   # src/utils/logging_setup.py:get_logger()
```
Used in: `data_loader.py`, `data_cleaner.py`, `feature_engineer.py`, `main.py`.
The factory adds console + file handlers exactly once (`if logger.handlers` guard) and
writes to `logs/pipeline.log`.

---

## 3. Classes for Stateful Loaders, Functions for Pure Transformations
- **Classes** (`src/data_pipeline/data_loader.py`): `ENTSOEClient`, `RegelleistungClient`,
  `EUETSLoader`, `GasPriceLoader` — each holds API credentials / URL templates / column
  candidates as instance state; exposes one public `fetch_*` method.
- **Functions** (`data_cleaner.py`, `feature_engineer.py`): stateless pure functions that
  take a Series/DataFrame and return a new one. No classes.

---

## 4. CSV Caching / Skip-if-exists
All four loaders guard expensive fetches with the same helper pattern:
```python
# src/data_pipeline/data_loader.py — _cached()
if _cached(out_path) and not FORCE_REDOWNLOAD:
    return pd.read_csv(out_path, index_col=0, parse_dates=True).squeeze()
```
`_cached(path)` returns `True` when the file exists and has non-zero size.
All loaders write their result to `data/raw/<name>.csv` before returning, ensuring
subsequent runs skip the network call.

---

## 5. UTC DatetimeIndex as the Canonical Data Structure
Every Series returned by a loader and every DataFrame produced by cleaners/engineers
uses a tz-aware UTC `DatetimeIndex` at hourly frequency.
- `data_cleaner.py:build_hourly_index()` generates the canonical index
- `data_cleaner.py:align_to_index()` coerces all inputs to it (handles naive, local, and
  already-UTC inputs)
- `.copy()` is called before any in-place mutation to preserve immutability

---

## 6. Orchestrator Functions Compose Pure Steps
Both `data_cleaner.py:clean_dataset()` and `feature_engineer.py:engineer_features()`
are orchestrators that call smaller pure functions in sequence, logging each step.
`src/data_pipeline/main.py:run_pipeline()` is the top-level orchestrator that wires
all four loaders → cleaner → engineer → CSV save.

---

## 7. Graceful Degradation with Fallback Loading
`main.py` follows a try-API → fallback-to-cache pattern:
- Attempt live fetch; on failure, log an error and load from the cached CSV if it exists
- Helper `_load_or_error()` encapsulates this fallback so the orchestrator stays readable

---

## 8. Validation Returns vs. Side-Effect Logging
- Functions that **fix** data return the corrected object (`fill_interpolated`, `remove_outliers`)
- Functions that **check** data log a warning and return nothing (`validate_missing_rate`)
- `remove_outliers` returns a `tuple[pd.Series, pd.Series]` (cleaned, mask) so callers
  can inspect what was removed — `src/data_pipeline/data_cleaner.py:remove_outliers()`

---

## 9. Test Organization — One Class per Function
`tests/test_data_pipeline.py` groups tests into classes named after the function under
test (e.g., `TestBuildHourlyIndex`, `TestCalculateCSS`).
Shared test data is provided by module-level pytest fixtures (`sample_index`,
`clean_series`, `sample_df`). No mocking — tests use real pandas objects.

---

## 10. Column Naming Convention — Descriptive with Units
All DataFrame/Series names include unit suffixes:
- `da_price_eur_mwh`, `affr_price_eur_mw`, `gas_price_eur_mwh_th`, `eu_ets_price_eur_t`
- Engineered: `css` (€/MWh), `hour_of_day`, `day_of_week`, `css_lag_1h`

This convention is consistent across loaders, cleaner, feature engineer, and tests.
