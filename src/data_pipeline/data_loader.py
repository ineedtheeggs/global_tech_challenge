"""
Data loader module: four sub-clients for fetching raw datasets.

Clients:
    ENTSOEClient        - DA prices and CCGT generation via entsoe-py
    RegelleistungClient - aFRR capacity prices from regelleistung.net
    EUETSLoader         - EU-ETS carbon allowance prices from static CSV
    GasPriceLoader      - Gas (THE/TTF) day-ahead prices from static CSV

All clients return hourly pd.Series with a UTC DatetimeIndex and save their
output to data/raw/ as CSV files.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

from src.utils.config import (
    AFRR_CSV_PATH,
    ANALYSIS_END,
    ANALYSIS_START,
    BIDDING_ZONE,
    DATA_RAW,
    ENTSOE_API_TOKEN,
    FORCE_REDOWNLOAD,
    REGELLEISTUNG_BASE_URL,
)
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _cached(path: Path, force: bool = FORCE_REDOWNLOAD) -> bool:
    """Return True if the file already exists and we should skip fetching."""
    exists = path.exists() and path.stat().st_size > 0
    if exists and not force:
        logger.info("Using cached file: %s", path.name)
    return exists and not force


def _is_stale(path: Path, end: str | pd.Timestamp, tolerance_days: int = 1) -> bool:
    """
    Return True if the cached CSV does not cover up to `end`.

    Reads the last index value from the file and compares it to `end` minus
    `tolerance_days`.  Returns True (stale) on any read error so the caller
    falls back to a fresh fetch.

    Args:
        path: CSV file with a DatetimeIndex as the first column.
        end: Expected end of coverage (string or Timestamp).
        tolerance_days: How many days short of `end` is still considered fresh.

    Returns:
        True if the file needs updating, False if it is sufficiently up-to-date.
    """
    if not path.exists() or path.stat().st_size == 0:
        return True
    try:
        idx = pd.read_csv(path, index_col=0, parse_dates=True).index
        if idx.empty:
            return True
        last = idx[-1]
        if getattr(last, "tz", None) is None:
            last = last.tz_localize("UTC")
        end_ts = pd.Timestamp(end, tz="UTC") if not isinstance(end, pd.Timestamp) else end
        return last < end_ts - pd.Timedelta(days=tolerance_days)
    except Exception:
        return True


# ---------------------------------------------------------------------------
# 1. ENTSO-E client (DA prices + CCGT generation)
# ---------------------------------------------------------------------------

class ENTSOEClient:
    """
    Fetches Day-Ahead prices and CCGT (Natural Gas) generation from
    ENTSO-E Transparency Platform using the entsoe-py library.

    Requires ENTSOE_API_TOKEN environment variable.
    See: https://transparency.entsoe.eu/

    psr_type codes:
        B04 = Fossil Gas (CCGT / OCGT combined)
    """

    PSR_TYPE_GAS = "B04"

    def __init__(self, api_token: str | None = ENTSOE_API_TOKEN) -> None:
        if not api_token:
            raise ValueError(
                "ENTSOE_API_TOKEN is not set. "
                "Export it with: export ENTSOE_API_TOKEN='your_token'"
            )
        try:
            from entsoe import EntsoePandasClient
        except ImportError as exc:
            raise ImportError(
                "entsoe-py is not installed. Run: pip install entsoe-py"
            ) from exc

        self._client = EntsoePandasClient(api_key=api_token)
        self._start = pd.Timestamp(ANALYSIS_START, tz="UTC")
        self._end = pd.Timestamp(ANALYSIS_END, tz="UTC") + pd.Timedelta(days=1)

    def fetch_da_prices(self, out_path: Path | None = None) -> pd.Series:
        """
        Fetch Day-Ahead electricity prices for DE-LU bidding zone.

        If a cached file exists but doesn't cover the full analysis window,
        only the missing tail is fetched and appended (incremental update).

        Args:
            out_path: Optional path to save the CSV. Defaults to data/raw/da_prices.csv.

        Returns:
            Hourly pd.Series of DA prices in €/MWh with UTC DatetimeIndex.
        """
        out_path = out_path or (DATA_RAW / "da_prices.csv")
        DATA_RAW.mkdir(parents=True, exist_ok=True)

        existing: pd.Series | None = None
        fetch_start = self._start

        if not FORCE_REDOWNLOAD and out_path.exists() and out_path.stat().st_size > 0:
            existing = pd.read_csv(out_path, index_col=0, parse_dates=True).squeeze()
            if existing.index.tz is None:
                existing.index = existing.index.tz_localize("UTC")
            if not _is_stale(out_path, self._end):
                logger.info("DA prices up-to-date, using cached file: %s", out_path.name)
                return existing
            # Incremental: only fetch the missing window
            fetch_start = (existing.index[-1] + pd.Timedelta(hours=1)).normalize()
            logger.info(
                "DA prices cached to %s; fetching tail from %s …",
                existing.index[-1].date(), fetch_start.date(),
            )
        else:
            logger.info("Fetching DA prices for %s from ENTSO-E …", BIDDING_ZONE)

        series = self._client.query_day_ahead_prices(
            country_code=BIDDING_ZONE,
            start=fetch_start,
            end=self._end,
        )
        series = series.rename("da_price_eur_mwh")
        series.index = series.index.tz_convert("UTC")
        series = series.resample("h").mean()

        if existing is not None and not existing.empty:
            series = pd.concat([existing, series]).sort_index()
            series = series[~series.index.duplicated(keep="last")]

        series.to_csv(out_path, header=True)
        logger.info("Saved DA prices: %d rows → %s", len(series), out_path)
        return series

    def fetch_ccgt_generation(self, out_path: Path | None = None) -> pd.Series:
        """
        Fetch actual CCGT (Fossil Gas) generation for DE-LU.

        If a cached file exists but doesn't cover the full analysis window,
        only the missing tail is fetched and appended (incremental update).

        Args:
            out_path: Optional path to save the CSV.

        Returns:
            Hourly pd.Series of CCGT generation in MW with UTC DatetimeIndex.
        """
        out_path = out_path or (DATA_RAW / "ccgt_generation.csv")
        DATA_RAW.mkdir(parents=True, exist_ok=True)

        existing: pd.Series | None = None
        fetch_start = self._start

        if not FORCE_REDOWNLOAD and out_path.exists() and out_path.stat().st_size > 0:
            existing = pd.read_csv(out_path, index_col=0, parse_dates=True).squeeze()
            if existing.index.tz is None:
                existing.index = existing.index.tz_localize("UTC")
            if not _is_stale(out_path, self._end):
                logger.info("CCGT generation up-to-date, using cached file: %s", out_path.name)
                return existing
            fetch_start = (existing.index[-1] + pd.Timedelta(hours=1)).normalize()
            logger.info(
                "CCGT generation cached to %s; fetching tail from %s …",
                existing.index[-1].date(), fetch_start.date(),
            )
        else:
            logger.info("Fetching CCGT generation for %s from ENTSO-E …", BIDDING_ZONE)

        df = self._client.query_generation(
            country_code=BIDDING_ZONE,
            start=fetch_start,
            end=self._end,
            psr_type=self.PSR_TYPE_GAS,
        )
        df.index = df.index.tz_convert("UTC")

        # query_generation may return a DataFrame with Actual Aggregated column
        if isinstance(df, pd.DataFrame):
            if "Actual Aggregated" in df.columns:
                series = df["Actual Aggregated"]
            else:
                series = df.iloc[:, 0]
        else:
            series = df

        series = series.rename("ccgt_generation_mw")
        series = series.resample("h").mean()
        series = self._patch_from_smard(series)

        if existing is not None and not existing.empty:
            series = pd.concat([existing, series]).sort_index()
            series = series[~series.index.duplicated(keep="last")]

        series.to_csv(out_path, header=True)
        logger.info("Saved CCGT generation: %d rows → %s", len(series), out_path)
        return series

    @staticmethod
    def _patch_from_smard(series: pd.Series) -> pd.Series:
        """Back-fill early-2020 CCGT NaNs from the SMARD supplement file.

        Args:
            series: CCGT generation series potentially containing NaNs in early 2020.

        Returns:
            Series with NaN positions filled where SMARD data is available.
        """
        smard_path = DATA_RAW / "ccgt_generation_2020_smard.csv"
        if not smard_path.exists():
            logger.debug("SMARD supplement not found at %s; skipping patch.", smard_path)
            return series
        smard = pd.read_csv(smard_path, index_col=0, parse_dates=True).squeeze()
        smard.index = (
            smard.index.tz_localize("UTC") if smard.index.tz is None else smard.index
        )
        smard.name = series.name
        # Only fill NaN positions to avoid overwriting good ENTSO-E data
        mask = series.isna() & series.index.isin(smard.index)
        series.loc[mask] = smard.reindex(series.index[mask])
        patched = int(mask.sum())
        if patched:
            logger.info("Patched %d CCGT NaNs from SMARD supplement.", patched)
        return series


# ---------------------------------------------------------------------------
# 2. Regelleistung.net client (aFRR capacity prices)
# ---------------------------------------------------------------------------

class RegelleistungClient:
    """
    Downloads aFRR (SRL positive) capacity prices from regelleistung.net.

    Data is published as 4-hour product blocks. This client downloads yearly
    CSVs, parses them, and resamples to an hourly forward-filled series.

    Source: https://www.regelleistung.net/ext/data/
    No authentication required.
    """

    # Column name in the downloaded CSV that holds the capacity price
    PRICE_COL_CANDIDATES = [
        "CAPACITY_PRICE_[EUR/MW]",
        "CAPACITY_PRICE",
        "PRICE_EUR_MW",
        "capacity_price",
    ]

    def __init__(self) -> None:
        self._start_year = int(ANALYSIS_START[:4])
        self._end_year = int(ANALYSIS_END[:4])

    def _build_url(self, year: int) -> str:
        """Construct download URL for a given year's aFRR positive data."""
        # The regelleistung.net portal uses a specific URL pattern.
        # Format: /ext/data/?period=yearly&type=SRL&direction=positive&year=YYYY
        return (
            f"{REGELLEISTUNG_BASE_URL}"
            f"?period=yearly&type=SRL&direction=positive&year={year}"
        )

    def _parse_response(self, text: str) -> pd.DataFrame:
        """
        Parse the CSV text from regelleistung.net.

        The site uses semicolon separators and European decimal commas.
        """
        df = pd.read_csv(
            io.StringIO(text),
            sep=";",
            decimal=",",
            encoding="utf-8",
            on_bad_lines="warn",
        )
        df.columns = df.columns.str.strip()
        return df

    def _extract_price_series(self, df: pd.DataFrame) -> pd.Series:
        """
        Extract a time-indexed price series from the raw DataFrame.

        Handles multiple possible column naming conventions.
        """
        # Find the timestamp column
        ts_col = None
        for candidate in ["DATE_FROM", "DATETIME_FROM", "FROM", "date_from", "start"]:
            if candidate in df.columns:
                ts_col = candidate
                break
        if ts_col is None:
            raise ValueError(
                f"Could not find timestamp column in regelleistung data. "
                f"Available columns: {list(df.columns)}"
            )

        # Find the price column
        price_col = None
        for candidate in self.PRICE_COL_CANDIDATES:
            if candidate in df.columns:
                price_col = candidate
                break
        if price_col is None:
            raise ValueError(
                f"Could not find price column in regelleistung data. "
                f"Available columns: {list(df.columns)}"
            )

        series = pd.to_numeric(df[price_col], errors="coerce")
        series.index = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        series = series.dropna()
        series = series.sort_index()
        return series

    def fetch_affr_prices(self, out_path: Path | None = None) -> pd.Series:
        """
        Download and aggregate aFRR positive capacity prices.

        Args:
            out_path: Optional save path (defaults to data/raw/affr_prices.csv).

        Returns:
            Hourly pd.Series of aFRR capacity prices (€/MW/h) with UTC index.
        """
        out_path = out_path or (DATA_RAW / "affr_prices.csv")
        DATA_RAW.mkdir(parents=True, exist_ok=True)

        if _cached(out_path):
            return pd.read_csv(out_path, index_col=0, parse_dates=True).squeeze()

        all_series: list[pd.Series] = []

        for year in range(self._start_year, self._end_year + 1):
            url = self._build_url(year)
            logger.info("Downloading aFRR data for %d from %s …", year, url)
            try:
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                df = self._parse_response(response.text)
                series = self._extract_price_series(df)
                all_series.append(series)
                logger.info("  → %d records for %d", len(series), year)
            except requests.RequestException as exc:
                logger.warning("Failed to download aFRR data for %d: %s", year, exc)
            except (ValueError, KeyError) as exc:
                logger.warning("Failed to parse aFRR data for %d: %s", year, exc)

        if not all_series:
            logger.error(
                "No aFRR data could be downloaded. "
                "Please manually download from https://www.regelleistung.net "
                "and save to %s", out_path
            )
            return pd.Series(name="affr_price_eur_mw", dtype=float)

        combined = pd.concat(all_series).sort_index()
        combined = combined[~combined.index.duplicated(keep="first")]

        # Resample 4-hour blocks to hourly by forward-filling
        hourly_index = pd.date_range(
            start=ANALYSIS_START, end=ANALYSIS_END, freq="h", tz="UTC"
        )
        hourly = combined.reindex(hourly_index).ffill()
        hourly.name = "affr_price_eur_mw"

        hourly.to_csv(out_path, header=True)
        logger.info("Saved aFRR prices: %d rows → %s", len(hourly), out_path)
        return hourly


# ---------------------------------------------------------------------------
# 2b. AFRRBidPriceLoader — static CSV (replaces RegelleistungClient)
# ---------------------------------------------------------------------------

class AFRRBidPriceLoader:
    """
    Loads aFRR upward (POS_*) capacity prices from a static bid-level CSV.

    CSV columns used:
        date_from                  - YYYY-MM-DD date of the product block
        product                    - e.g. POS_00_04, POS_04_08 (upward blocks)
        capacity_price_[eur/mw]    - per-bid capacity price to aggregate
        allocated_capacity_[mw]    - MW weight for the weighted average

    Aggregation logic:
        1. Filter rows where product starts with 'POS' (upward aFRR spin-up).
        2. Group by (date_from, product) — one 4-hour block per day.
        3. Compute capacity-weighted mean of capacity_price_[eur/mw].
        4. Parse UTC timestamp: date_from + start hour from product code
           (POS_HH_XX → datetime(date, HH, 0, tzinfo=UTC)).
        5. Reindex to full hourly UTC range; forward-fill within each block.
    """

    PRICE_COL = "capacity_price_[eur/mw]"
    WEIGHT_COL = "allocated_capacity_[mw]"
    DATE_COL = "date_from"
    PRODUCT_COL = "product"

    @staticmethod
    def _parse_block_start(product: str) -> int:
        """Extract start hour from product code 'POS_HH_XX' → int(HH)."""
        parts = product.split("_")
        return int(parts[1])

    def load(self, csv_path: Path | None = None, out_path: Path | None = None) -> pd.Series:
        """
        Load and aggregate aFRR upward capacity prices to an hourly series.

        Args:
            csv_path:  Path to the bid-level CSV. Defaults to AFRR_CSV_PATH.
            out_path:  Where to save the result. Defaults to data/raw/affr_prices.csv.

        Returns:
            Hourly pd.Series named 'affr_price_eur_mw' with UTC DatetimeIndex.
        """
        csv_path = csv_path or AFRR_CSV_PATH
        out_path = out_path or (DATA_RAW / "affr_prices.csv")
        DATA_RAW.mkdir(parents=True, exist_ok=True)

        if _cached(out_path) and not _is_stale(out_path, ANALYSIS_END):
            return pd.read_csv(out_path, index_col=0, parse_dates=True).squeeze()

        if not csv_path.exists():
            raise FileNotFoundError(
                f"aFRR bid prices CSV not found: {csv_path}\n"
                "Place the pre-compiled CSV at the configured AFRR_CSV_PATH."
            )

        logger.info("Loading aFRR bid prices from %s …", csv_path)
        df = pd.read_csv(csv_path, low_memory=False)
        df.columns = df.columns.str.strip()

        # 1. Keep only upward (POS_*) products
        df = df[df[self.PRODUCT_COL].str.startswith("POS")].copy()

        # 2. Drop rows with missing price or weight
        df = df.dropna(subset=[self.PRICE_COL, self.WEIGHT_COL])
        df[self.PRICE_COL] = pd.to_numeric(df[self.PRICE_COL], errors="coerce")
        df[self.WEIGHT_COL] = pd.to_numeric(df[self.WEIGHT_COL], errors="coerce")
        df = df.dropna(subset=[self.PRICE_COL, self.WEIGHT_COL])

        # 3. Weighted-average price per (date_from, product) block
        def _wavg(group: pd.DataFrame) -> float:
            weights = group[self.WEIGHT_COL]
            total_w = weights.sum()
            if total_w == 0:
                return group[self.PRICE_COL].mean()
            return (group[self.PRICE_COL] * weights).sum() / total_w

        block_prices = (
            df.groupby([self.DATE_COL, self.PRODUCT_COL])
            .apply(_wavg, include_groups=False)
            .reset_index()
        )
        block_prices.columns = [self.DATE_COL, self.PRODUCT_COL, "price"]

        # 4. Parse UTC timestamps: date + start hour from product code
        block_prices["start_hour"] = block_prices[self.PRODUCT_COL].apply(
            self._parse_block_start
        )
        block_prices["timestamp"] = pd.to_datetime(
            block_prices[self.DATE_COL], utc=True
        ) + pd.to_timedelta(block_prices["start_hour"], unit="h")

        series = block_prices.set_index("timestamp")["price"].sort_index()
        series = series[~series.index.duplicated(keep="first")]

        # 5. Reindex to full hourly UTC; ffill fills each 4-hour block forward
        hourly_index = pd.date_range(
            start=ANALYSIS_START, end=ANALYSIS_END, freq="h", tz="UTC"
        )
        hourly = series.reindex(hourly_index).ffill()
        hourly.name = "affr_price_eur_mw"

        hourly.to_csv(out_path, header=True)
        logger.info("Saved aFRR prices: %d rows → %s", len(hourly), out_path)
        return hourly


# ---------------------------------------------------------------------------
# 3. EU-ETS carbon price loader
# ---------------------------------------------------------------------------

class EUETSLoader:
    """
    Loads EU-ETS carbon allowance (EUA) prices from a static CSV file.

    Actual CSV format (annual averages):
        Date,average_prices
        2012,7.387083333
        ...
        2023,84.59738281

    Date column contains integer years; price column is the annual average €/tCO₂.
    Each year's price is forward-filled to every hour in that calendar year.

    Place the downloaded file at: data/raw/eu_ets_prices.csv
    """

    EXPECTED_PRICE_COL_CANDIDATES = [
        "average_prices",
        "price_eur_tco2",
        "price",
        "Price",
        "EUA",
        "eua_spot",
        "carbon_price",
        "value",
    ]

    def load(self, csv_path: Path | None = None) -> pd.Series:
        """
        Load EUA annual-average prices and forward-fill to hourly resolution.

        Args:
            csv_path: Path to the raw CSV file. Defaults to data/raw/eu_ets_prices.csv.

        Returns:
            Hourly pd.Series of EUA prices (€/tCO₂) with UTC DatetimeIndex.
        """
        csv_path = csv_path or (DATA_RAW / "eu_ets_prices.csv")

        if not csv_path.exists():
            raise FileNotFoundError(
                f"EU-ETS price file not found: {csv_path}\n"
                "Download annual EUA average prices and save as: data/raw/eu_ets_prices.csv\n"
                "Required columns: Date (integer year), average_prices"
            )

        logger.info("Loading EU-ETS prices from %s …", csv_path)
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        # Identify price column (Date column always contains integer years)
        price_col = self._find_column(df, self.EXPECTED_PRICE_COL_CANDIDATES, "price")

        # Filter to analysis window and build a UTC-timestamped series (Jan 1 of each year)
        start_year = int(ANALYSIS_START[:4])
        end_year = int(ANALYSIS_END[:4])
        df["_year"] = pd.to_numeric(df["Date"], errors="coerce")
        df = df.dropna(subset=["_year"])
        df = df[(df["_year"] >= start_year) & (df["_year"] <= end_year)]

        prices = pd.to_numeric(df[price_col], errors="coerce")
        timestamps = df["_year"].apply(
            lambda y: pd.Timestamp(f"{int(y)}-01-01", tz="UTC")
        )
        series = prices.set_axis(timestamps).dropna().sort_index()
        series = series[~series.index.duplicated(keep="first")]

        # Warn if data doesn't cover the full analysis window so forward-fill is obvious
        last_year = int(series.index.max().year)
        end_year = int(ANALYSIS_END[:4])
        if last_year < end_year:
            logger.warning(
                "EU-ETS CSV only covers up to %d but analysis window runs to %d. "
                "Forward-filling %d year(s) with %.2f €/tCO₂ (last available value). "
                "Update data/raw/eu_ets_prices.csv with annual averages for %s to fix this.",
                last_year,
                end_year,
                end_year - last_year,
                float(series.iloc[-1]),
                ", ".join(str(y) for y in range(last_year + 1, end_year + 1)),
            )

        # Reindex to full hourly UTC index; ffill assigns each year's price to all hours
        hourly_index = pd.date_range(
            start=ANALYSIS_START, end=ANALYSIS_END, freq="h", tz="UTC"
        )
        hourly = series.reindex(hourly_index, method="ffill")
        hourly.name = "eu_ets_price_eur_tco2"

        logger.info(
            "Loaded EU-ETS prices: %d annual obs → %d hourly rows", len(series), len(hourly)
        )
        return hourly

    @staticmethod
    def _find_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
        for candidate in candidates:
            if candidate in df.columns:
                return candidate
        raise ValueError(
            f"Could not find {label} column in EU-ETS CSV. "
            f"Available: {list(df.columns)}. "
            f"Tried: {candidates}"
        )


# ---------------------------------------------------------------------------
# 4. Gas price loader (THE / TTF day-ahead)
# ---------------------------------------------------------------------------

class GasPriceLoader:
    """
    Loads gas day-ahead prices (THE or TTF hub) from a static CSV file.

    Actual CSV format (UTF-8 BOM, newest-first, MM/DD/YYYY dates):
        "Date","Price","Open","High","Low","Vol.","Change %"
        "12/29/2023","32.350","33.550","34.065","31.635","0.21K","-2.28%"
        ...

    Encoding: utf-8-sig (strips BOM automatically).
    Date column: MM/DD/YYYY format.
    Only the "Price" column is used; all other columns are ignored.
    Data arrives newest-first and is sorted ascending before reindexing.

    Place the downloaded file at: data/raw/gas_prices.csv
    Units must be €/MWh_thermal (convert from p/therm or $/MMBtu if needed).
    """

    EXPECTED_DATE_COL_CANDIDATES = ["Date", "date", "DATE", "Day", "day"]
    EXPECTED_PRICE_COL_CANDIDATES = [
        "Price",
        "price_eur_mwh_th",
        "price",
        "gas_price",
        "THE",
        "TTF",
        "value",
        "close",
        "Close",
    ]

    def load(self, csv_path: Path | None = None) -> pd.Series:
        """
        Load gas day-ahead prices and forward-fill to hourly resolution.

        Args:
            csv_path: Path to the raw CSV. Defaults to data/raw/gas_prices.csv.

        Returns:
            Hourly pd.Series of gas prices (€/MWh_th) with UTC DatetimeIndex.
        """
        csv_path = csv_path or (DATA_RAW / "gas_prices.csv")

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Gas price file not found: {csv_path}\n"
                "Download THE/TTF day-ahead prices and save as: data/raw/gas_prices.csv\n"
                "Required columns: Date (MM/DD/YYYY), Price"
            )

        logger.info("Loading gas prices from %s …", csv_path)
        # utf-8-sig strips the BOM that causes column-name corruption
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        df.columns = df.columns.str.strip()

        date_col = self._find_column(df, self.EXPECTED_DATE_COL_CANDIDATES, "date")
        price_col = self._find_column(df, self.EXPECTED_PRICE_COL_CANDIDATES, "price")

        prices = pd.to_numeric(df[price_col], errors="coerce")
        dates = pd.to_datetime(df[date_col], format="%m/%d/%Y", utc=True, errors="coerce")
        series = prices.set_axis(dates).dropna().sort_index()  # sort ascending
        series = series[~series.index.duplicated(keep="first")]

        # Forward-fill daily → hourly
        hourly_index = pd.date_range(
            start=ANALYSIS_START, end=ANALYSIS_END, freq="h", tz="UTC"
        )
        hourly = series.reindex(hourly_index, method="ffill")
        hourly.name = "gas_price_eur_mwh_th"

        logger.info(
            "Loaded gas prices: %d daily obs → %d hourly rows", len(series), len(hourly)
        )
        return hourly

    @staticmethod
    def _find_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
        for candidate in candidates:
            if candidate in df.columns:
                return candidate
        raise ValueError(
            f"Could not find {label} column in gas price CSV. "
            f"Available: {list(df.columns)}. "
            f"Tried: {candidates}"
        )
