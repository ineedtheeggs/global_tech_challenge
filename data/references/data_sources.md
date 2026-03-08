# Data Sources Reference

All 5 datasets required for the aFRR Opportunity Cost Calculator.

---

## 1. DA Prices — ENTSO-E Transparency Platform

- **URL:** https://transparency.entsoe.eu/
- **Access:** Free API (requires registration for token)
- **Registration:** https://transparency.entsoe.eu/usrm/user/register
- **Method:** `entsoe-py` Python client
- **Zone:** DE-LU (Germany-Luxembourg bidding zone)
- **Granularity:** Hourly
- **Years:** 2020–2023
- **Token setup:**
  ```bash
  export ENTSOE_API_TOKEN="your_token_here"
  # Or add to .env file (never commit .env)
  ```

---

## 2. aFRR Prices — regelleistung.net

- **URL:** https://www.regelleistung.net/ext/data/
- **Access:** Public, no authentication required
- **Data:** SRL (Sekundärregelleistung = aFRR) positive capacity prices
- **Format:** Semicolon-separated CSV, European decimal commas
- **Granularity:** 4-hour product blocks → resampled to hourly
- **All 4 German TSOs:** 50Hertz, Amprion, TenneT, TransnetBW
- **Manual download path (if API fails):**
  1. Go to https://www.regelleistung.net/ext/data/
  2. Select: Product = SRL, Direction = Positive, Period = Yearly
  3. Download CSV for each year 2020–2023
  4. Concatenate and save to `data/raw/affr_prices.csv`

---

## 3. EU-ETS Carbon Prices

- **Primary source (free):** Ember Climate Carbon Price Viewer
  - URL: https://ember-climate.org/data/data-tools/carbon-price-viewer/
  - Select "EUA Spot Price" → Export CSV
  - Save to `data/raw/eu_ets_prices.csv`
  - Required columns: `date`, `price_eur_tco2`

- **Alternative source:** ICE Endex EUA Daily Futures Settlement
  - URL: https://www.theice.com/marketdata/reports/ReportCenter.shtml#report/103
  - Requires ICE account (free registration)

- **Unit:** €/tCO₂ (euros per tonne of CO₂)
- **Granularity:** Daily → forward-filled to hourly

---

## 4. Gas Prices (THE Day-Ahead)

- **Primary source (free):** energy-charts.info
  - URL: https://www.energy-charts.info/charts/gas_spot_price/chart.htm
  - Select "THE Day-Ahead" → Export CSV
  - Save to `data/raw/gas_prices.csv`
  - Required columns: `date`, `price_eur_mwh_th`

- **Alternative sources:**
  - TTF Day-Ahead via investing.com (free, registration may be required)
  - ENTSO-G (https://transparency.entsog.eu/) — API available
  - Ember Climate gas price dataset

- **Unit:** €/MWh_th (euros per MWh thermal — NOT electrical)
- **Granularity:** Daily → forward-filled to hourly
- **Note:** THE (Trading Hub Europe) is the German gas hub, equivalent to TTF for Germany

---

## 5. CCGT Generation — ENTSO-E Transparency Platform

- **URL:** https://transparency.entsoe.eu/
- **Same API token** as DA prices
- **Query:** Actual Generation Per Production Type
- **PSR Type:** B04 = Fossil Gas (includes both CCGT and OCGT)
- **Zone:** DE-LU
- **Granularity:** Hourly
- **Note:** ENTSO-E reports B04 as the aggregate of all gas-fired plants

---

## Column Name Mapping

If your downloaded CSV files use different column names than expected,
update the `EXPECTED_*_COL_CANDIDATES` lists in `src/data_pipeline/data_loader.py`.

Current accepted column names per file:

| File | Date column candidates | Price column candidates |
|------|----------------------|------------------------|
| eu_ets_prices.csv | date, Date, DATE, Day | price_eur_tco2, price, Price, EUA, eua_spot, carbon_price, value |
| gas_prices.csv | date, Date, DATE, Day | price_eur_mwh_th, price, Price, gas_price, THE, TTF, value, close |

---

## Data Freshness

| Dataset | Update trigger | Who fetches |
|---------|---------------|-------------|
| DA prices | Quarterly refresh | `ENTSOEClient` via API |
| aFRR prices | Quarterly refresh | `RegelleistungClient` via HTTP |
| EU-ETS | Annual refresh | Manual download + update CSV |
| Gas prices | Annual refresh | Manual download + update CSV |
| CCGT generation | Quarterly refresh | `ENTSOEClient` via API |
