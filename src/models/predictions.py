"""
aFRR opportunity cost calculator.

Provides a Python function and a CLI tool for estimating the aFRR-up bid price
given current market inputs and an assumed market regime.

Model:  aFRR_Price = β₀ + β₁·CSS    (regime-specific coefficients)
        CSS = DA_Price - Gas_Price/η - ETS·intensity/η

Usage (CLI):
    python src/models/predictions.py \\
        --da-price 80 \\
        --gas-price 35 \\
        --ets-price 60 \\
        --regime low

Usage (Python API):
    from src.models.predictions import estimate_afrr_price
    price = estimate_afrr_price(da_price=80, gas_price=35, eu_ets_price=60, regime="low")
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as a script from the project root
sys.path.insert(0, str(Path(__file__).parents[2]))

from src.utils.config import CARBON_INTENSITY, EFFICIENCY, OUTPUTS_MODELS
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

METADATA_PATH = OUTPUTS_MODELS / "regime_metadata.json"
VALID_REGIMES = ("high", "medium", "low")


def _load_model(regime: str, year: int = 2022):
    """Load the pickled OLS model for the given regime and year.

    Args:
        regime: One of 'high', 'medium', 'low'.
        year: Year of the model to load (default 2022 for backward compat).

    Returns:
        Fitted statsmodels OLS results object.

    Raises:
        FileNotFoundError: If the model file does not exist.
        ValueError: If regime is not valid.
    """
    if regime not in VALID_REGIMES:
        raise ValueError(f"regime must be one of {VALID_REGIMES}, got '{regime}'")

    model_path = OUTPUTS_MODELS / f"regime_{regime}_{year}_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Run regression_models.py first: python src/models/regression_models.py"
        )

    with open(model_path, "rb") as fh:
        return pickle.load(fh)


def _load_metadata() -> dict:
    """Load regime metadata JSON sidecar.

    Returns:
        Dict with per-regime statistics (including 'ccgt_mean_mw').

    Raises:
        FileNotFoundError: If metadata file does not exist.
    """
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_PATH}. "
            "Run regression_models.py first: python src/models/regression_models.py"
        )
    with open(METADATA_PATH) as fh:
        return json.load(fh)


def _compute_css(da_price: float, gas_price: float, eu_ets_price: float) -> float:
    """Compute Clean Spark Spread from market inputs.

    Args:
        da_price: Day-ahead electricity price (€/MWh).
        gas_price: Gas forward price (€/MWh thermal).
        eu_ets_price: EU-ETS carbon allowance price (€/tCO₂).

    Returns:
        CSS value in €/MWh.
    """
    carbon_cost = eu_ets_price * CARBON_INTENSITY / EFFICIENCY
    return da_price - (gas_price / EFFICIENCY) - carbon_cost


def estimate_afrr_price(
    da_price: float,
    gas_price: float,
    eu_ets_price: float,
    regime: str,
    year: int = 2022,
) -> float:
    """Estimate aFRR-up opportunity cost bid price.

    Computes CSS from the provided market inputs, then applies the
    regime-specific OLS model:  aFRR_Price = β₀ + β₁·CSS

    Args:
        da_price: Day-ahead electricity price (€/MWh).
        gas_price: Gas forward price (€/MWh thermal).
        eu_ets_price: EU-ETS carbon allowance price (€/tCO₂).
        regime: Market regime — one of 'high', 'medium', 'low'.
        year: Year of the model to use (default 2022 for backward compat).

    Returns:
        Estimated aFRR-up price in €/MW.

    Raises:
        ValueError: If regime is invalid or any price is negative.
        FileNotFoundError: If model files are missing.
    """
    if any(v < 0 for v in [da_price, gas_price, eu_ets_price]):
        raise ValueError("Prices cannot be negative.")
    if regime not in VALID_REGIMES:
        raise ValueError(f"regime must be one of {VALID_REGIMES}, got '{regime}'")

    model = _load_model(regime, year=year)
    css = _compute_css(da_price, gas_price, eu_ets_price)
    logger.debug("Computed CSS: %.4f €/MWh", css)

    feature_vector = pd.DataFrame([[1.0, css]], columns=["const", "css"])
    prediction = float(model.predict(feature_vector)[0])
    return prediction


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate aFRR-up opportunity cost bid price.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--da-price",
        type=float,
        required=True,
        metavar="EUR_MWH",
        help="Day-ahead electricity price (€/MWh).",
    )
    parser.add_argument(
        "--gas-price",
        type=float,
        required=True,
        metavar="EUR_MWH_TH",
        help="Gas forward price (€/MWh thermal).",
    )
    parser.add_argument(
        "--ets-price",
        type=float,
        required=True,
        metavar="EUR_TCO2",
        help="EU-ETS carbon allowance price (€/tCO₂).",
    )
    parser.add_argument(
        "--regime",
        type=str,
        required=True,
        choices=list(VALID_REGIMES),
        help="Market regime.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2022,
        metavar="YEAR",
        help="Year of the fitted model to use (default: 2022).",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    parser = _build_cli_parser()
    args = parser.parse_args()

    price = estimate_afrr_price(
        da_price=args.da_price,
        gas_price=args.gas_price,
        eu_ets_price=args.ets_price,
        regime=args.regime,
        year=args.year,
    )

    css = _compute_css(args.da_price, args.gas_price, args.ets_price)
    print(f"\n{'='*50}")
    print(f"  aFRR-up Opportunity Cost Estimate")
    print(f"{'='*50}")
    print(f"  DA price:        {args.da_price:>8.2f} €/MWh")
    print(f"  Gas price:       {args.gas_price:>8.2f} €/MWh_th")
    print(f"  ETS price:       {args.ets_price:>8.2f} €/tCO₂")
    print(f"  CSS (computed):  {css:>8.2f} €/MWh")
    print(f"  Regime:          {args.regime}")
    print(f"  Model year:      {args.year}")
    print(f"{'='*50}")
    print(f"  Estimated aFRR price: {price:>8.2f} €/MW")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
