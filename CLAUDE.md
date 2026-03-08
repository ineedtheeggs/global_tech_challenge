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
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore rules
│
├── data/
│   ├── raw/
│   │   ├── da_prices.csv               # Day-ahead market prices
│   │   ├── affr_prices.csv             # aFRR spin-up market prices
│   │   ├── ccgt_generation.csv         # CCGT generation per PTU
│   │   ├── eu_ets_prices.csv           # Carbon allowance prices
│   │   └── gas_prices.csv              # Gas forward prices
│   ├── processed/
│   │   ├── combined_dataset.csv        # Merged all features
│   │   ├── train_test_split.csv        # Train/test datasets
│   │   └── market_regimes.csv          # PTU regime classifications
│   └── references/
│       ├── data_sources.md             # External data source documentation
│       └── data_dictionary.csv         # Column definitions
│
├── src/
│   ├── __init__.py
│   │
│   ├── data_pipeline/
│   │   ├── __init__.py
│   │   ├── data_loader.py              # Load from external sources
│   │   ├── data_cleaner.py             # Handle missing values, outliers
│   │   ├── feature_engineer.py         # Create CSS, spreads, etc.
│   │   └── main.py                     # Execute full pipeline
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── regime_classifier.py        # Classify PTUs into regimes
│   │   ├── regression_models.py        # Fit OLS models per regime
│   │   ├── predictions.py              # Generate opportunity cost estimates
│   │   └── model_validation.py         # R², residual analysis, diagnostics
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── statistics.py               # Correlation analysis, summary stats
│   │   ├── visualization.py            # Plots for regime conditions
│   │   ├── dashboard.py                # Interactive dashboard
│   │   └── report_generator.py         # Automated report generation
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py                   # Configuration constants
│       ├── logging_setup.py            # Logging utilities
│       └── helpers.py                  # General utility functions
│
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb   # Data exploration
│   ├── 02_regime_analysis.ipynb        # Market regime investigation
│   ├── 03_model_development.ipynb      # Model fitting & diagnostics
│   └── 04_results_interpretation.ipynb # Results & insights
│
├── tests/
│   ├── __init__.py
│   ├── test_data_pipeline.py           # Test data loading/cleaning
│   ├── test_regime_classifier.py       # Test regime classification logic
│   ├── test_models.py                  # Test regression models
│   └── test_analysis.py                # Test analysis functions
│
├── outputs/
│   ├── models/                         # Saved model files (pickle)
│   │   ├── regime_high_model.pkl
│   │   ├── regime_medium_model.pkl
│   │   └── regime_low_model.pkl
│   ├── plots/                          # Generated visualizations
│   │   ├── regime_distribution.png
│   │   ├── css_vs_affr_prices.png
│   │   └── model_diagnostics.png
│   ├── reports/
│   │   ├── market_regime_report.html   # Automated HTML report
│   │   └── model_summary.txt           # Text summary of results
│   └── data/
│       └── opportunity_cost_estimates.csv  # Final estimates
│
└── venv/                               # Virtual environment (not committed)
```

---

## Key Concepts & Domain Knowledge

### Clean Spark Spread (CSS)
```
CSS = DA_Price + Gas_Price + Carbon_Tax - Efficiency_Loss
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

| Regime | Gas Generation | Reserve Supply | aFRR Price Sensitivity |
|--------|----------------|-----------------|----------------------|
| **High** | Many plants online (>75th %ile) | Abundant | Low CSS sensitivity |
| **Medium** | Moderate generation (25-75th %ile) | Balanced | Moderate sensitivity |
| **Low** | Few plants online (<25th %ile) | Scarce | High CSS sensitivity |

**Rationale:** Fewer plants online = fewer reserve providers = higher prices with steeper CSS correlation.

---

## Technical Requirements

### Python Stack
- **Python:** 3.11+
- **Data:** pandas, NumPy
- **Modeling:** scikit-learn, statsmodels
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Notebooks:** Jupyter
- **Testing:** pytest, pytest-cov
- **IDE:** VS Code (Remote-WSL extension)

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

### Phase 1: Data Pipeline
```bash
python src/data_pipeline/main.py
```
**Outputs:**
- `data/processed/combined_dataset.csv` - Features + labels
- `data/processed/market_regimes.csv` - Regime classifications
- Summary statistics in logs

### Phase 2: Model Development
```bash
python src/models/regime_classifier.py
python src/models/regression_models.py
python src/models/model_validation.py
```
**Outputs:**
- `outputs/models/*.pkl` - Trained models (3 regime models)
- `outputs/plots/model_diagnostics.png` - Residual plots, Q-Q plots
- Model coefficients and R² values

### Phase 3: Analysis & Visualization
```bash
python src/analysis/visualization.py
python src/analysis/dashboard.py
python src/analysis/report_generator.py
```
**Outputs:**
- `outputs/plots/*.png` - Market condition visualizations
- `outputs/reports/market_regime_report.html` - Interactive HTML report
- `outputs/data/opportunity_cost_estimates.csv` - Final estimates

### Testing & Validation
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/ --cov-report=html

# Run specific test file
pytest tests/test_models.py -v
```

---

## Claude Code Usage

When using Claude Code in this project:

```bash
# Start session in project directory
cd ~/projects/global-tech-challenge
claude
```

### Common Commands

**Code Generation:**
```
/generate "Create a function to calculate CSS from DA price, gas price, and carbon tax"
/generate "Write data validation for market regime classification"
/generate "Build visualization for regime distributions"
```

**Code Review & Analysis:**
```
/ask "Review the regression model assumptions and suggest diagnostics"
/review src/models/regression_models.py
/ask "What's the correlation structure between our predictors?"
```

**Testing & Debugging:**
```
/generate "Create comprehensive unit tests for data_loader.py"
/ask "Why might the Low regime model have high residuals in Q4?"
/test  # Run pytest if tests exist
```

**Documentation:**
```
/ask "Generate docstrings for all functions in data_pipeline module"
/generate "Create a detailed README for the analysis module"
```

**Project Questions:**
```
/ask "Explain the market regime classification logic"
/ask "What assumptions does the linear regression model make?"
/ask "How should we handle outliers in CCGT generation data?"
```

---

## Coding Standards

### Python Style
- Follow **PEP 8** (line length: 100 chars)
- Use **type hints** for all function parameters and returns
- Write **docstrings** for all modules, classes, and public functions
- Use **meaningful variable names** (no single letters except `i` in loops)

### Example Function Style
```python
def calculate_css(
    da_price: float,
    gas_price: float,
    carbon_tax: float,
    efficiency: float = 0.5
) -> float:
    """
    Calculate Clean Spark Spread (CSS) for CCGT plant.
    
    Args:
        da_price: Day-ahead market price (€/MWh)
        gas_price: Gas forward price (€/MWh thermal)
        carbon_tax: EU-ETS carbon allowance price (€/ton CO2)
        efficiency: Plant efficiency factor (default 0.5 = 50%)
    
    Returns:
        CSS value in €/MWh
    
    Raises:
        ValueError: If inputs are negative
    """
    if any(x < 0 for x in [da_price, gas_price, carbon_tax]):
        raise ValueError("Prices cannot be negative")
    
    carbon_cost = carbon_tax * 0.202 / efficiency  # ~0.2 ton CO2/MWh gas
    css = da_price - (gas_price / efficiency) - carbon_cost
    return css
```

### Git Commit Messages
- **Format:** Imperative mood, max 72 chars
- **Good:** `"Add regime classifier with percentile thresholds"`
- **Bad:** `"Added classifier"` or `"WIP: trying to fix something"`
- **Format:** 
  ```
  Add regime classification module
  
  - Classify PTUs into high/medium/low based on CCGT generation percentiles
  - Add unit tests for boundary conditions
  - Create visualization of regime distribution
  ```

---

## Model Specification

### Regression Model Template
```
aFRR_Price = β₀ + β₁*CSS + β₂*DA_Price + β₃*CCGT_Gen + ε
```

**Fitted separately for each regime** to capture interaction effects.

### Expected Coefficients
| Regime | β₁ (CSS) | Direction | Interpretation |
|--------|----------|-----------|-----------------|
| High | Low (~0.05-0.10) | Weak | Abundant reserves suppress CSS effect |
| Medium | Moderate (~0.15-0.25) | Medium | Mixed supply conditions |
| Low | High (~0.35-0.50) | Strong | Scarce reserves amplify CSS effect |

### Validation Metrics
- **R² > 0.50** for each regime model
- **Residuals:** Normally distributed, homoscedastic
- **Durbin-Watson:** 1.8-2.2 (check for autocorrelation)
- **VIF < 5** for all predictors (multicollinearity check)

---

## Data Sources & Access

| Dataset | Source | Format | Update Frequency |
|---------|--------|--------|------------------|
| DA Prices | ENTSO-E Transparency | CSV/API | Hourly |
| aFRR Prices | ENTSO-E or TSO | CSV/API | Per PTU |
| CCGT Generation | ENTSO-E Transparency | CSV/API | Per PTU |
| EU-ETS Prices | ICE/BloombergNEF | CSV/web scrape | Daily |
| Gas Prices | ICE/Refinitiv | CSV/API | Daily/Weekly |

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
   - Missing data → interpolation strategy documented in data_cleaner.py
   - Outliers → removal criteria explained in feature_engineer.py

---

## Key Deliverables

### 1. Market Regime Classifier
- **Input:** CCGT generation time series
- **Output:** Regime labels (High/Medium/Low) per PTU
- **File:** `src/models/regime_classifier.py`

### 2. Regression Models (3 models)
- **High regime model:** `outputs/models/regime_high_model.pkl`
- **Medium regime model:** `outputs/models/regime_medium_model.pkl`
- **Low regime model:** `outputs/models/regime_low_model.pkl`
- **Format:** Sklearn-compatible pickle files

### 3. Opportunity Cost Calculator
- **Type:** Function or CLI tool
- **Inputs:** DA price forecast, Gas forward price, Carbon tax, Market regime
- **Output:** Estimated aFRR-up opportunity cost (€/MWh)
- **Files:** `src/models/predictions.py` + CLI wrapper

### 4. Analysis Report
- **Format:** HTML interactive dashboard + PDF summary
- **Contents:** Model coefficients, regime distributions, forecast curves
- **Files:** `outputs/reports/market_regime_report.html`

### 5. Visualizations
- CSS vs aFRR prices scatter plot (colored by regime)
- Model diagnostic plots (residuals, Q-Q plot, scale-location)
- Regime distribution over time
- Coefficient comparison across regimes
- Forecast curves (DA price scenarios)

---

## Testing Strategy

### Unit Tests
```bash
pytest tests/test_data_pipeline.py -v  # Data loading/cleaning
pytest tests/test_models.py -v         # Model fitting & prediction
pytest tests/test_analysis.py -v       # Analysis functions
```

### What to Test
- **Data Pipeline:** Handles missing values, outliers, merges correctly
- **Regime Classifier:** Correctly assigns percentile-based regimes
- **Models:** Coefficients reasonable, predictions in expected ranges
- **Calculations:** CSS formula, metrics (R², VIF, DW) correct

### Coverage Target
- Minimum **80% code coverage**
- Every public function tested
- Edge cases documented in test docstrings

---

## Next Steps (For Claude Code)

When you first run `claude` in this project, ask:

```
/ask "Help me set up the complete data pipeline from raw CSV files to processed datasets"
/generate "Create the data_loader.py module to read ENTSO-E CSV files"
/generate "Build the regime_classifier.py with percentile-based logic"
/ask "What diagnostic plots should I create to validate the regression models?"
/generate "Create comprehensive test suite for the models module"
```

---

## Useful Commands

```bash
# Data processing
python src/data_pipeline/main.py

# Model training
python src/models/regression_models.py

# Generate visualizations
python src/analysis/visualization.py

# Run tests
pytest tests/ -v --cov=src/

# Jupyter exploration
jupyter notebook notebooks/

# Generate report
python src/analysis/report_generator.py
```

---

## Configuration

Key settings in `src/utils/config.py`:

```python
# Market regime thresholds (percentiles)
REGIME_THRESHOLDS = {
    'high': 0.75,      # Top 25% by CCGT generation
    'medium': 0.25,    # Middle 50%
    'low': 0.00        # Bottom 25%
}

# Time period for analysis
ANALYSIS_START = '2020-01-01'  # Post EU-ETS carbon pricing
ANALYSIS_END = '2023-12-31'    # Pre BESS boom (approx)

# Model hyperparameters
OLS_FIT_METHOD = 'pinv'        # Pseudo-inverse for robustness
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Data validation
MAX_MISSING_RATE = 0.05        # Max 5% missing values per column
OUTLIER_THRESHOLD = 3.0        # Z-score threshold
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
- **Created:** [Date]
- **Last Updated:** [Date]
- **Version:** 1.0.0

---

**This file is committed to version control so the entire team benefits from these guidelines.**
