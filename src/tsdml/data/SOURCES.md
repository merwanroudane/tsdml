# Bundled data — provenance and terms

## `data_macroprudential.csv`

Quarterly Italian macro-financial and banking series, **2005Q2 – 2024Q4**,
73 rows (two metadata rows plus 71 quarters) × 37 variables.

This is the dataset behind Section 5 of

> Ciganovic, M., D'Amario, F. and Tancioni, M. (2026).
> *Double Machine Learning for Time Series.* The Econometrics Journal.
> arXiv:2603.10999.

It is distributed here as it appears in the authors' replication package, so
that the empirical application can be reproduced from a `pip install` with no
extra downloads.

### Structure

| Row | Meaning |
|---|---|
| `speed` | `fast` = enters the control set contemporaneously; `slow` = lags only |
| `Transform:` | stationarity transformation code (see `tsdml.TRANSFORM_CODES`) |
| dates | quarter-end observations |

### Variable groups and underlying sources

| Group | Variables | Source |
|---|---|---|
| Bank capital and risk | `Tier 1 capital to risk-weighted assets_Percent`, `Regulatory capital to risk-weighted assets_Percent`, `Tier 1 capital`, `Total regulatory capital`, `Risk-weighted assets`, `Tier 1 capital_to_total_Assets_Percent` | IMF Financial Soundness Indicators (semi-annual for Italy from 2005Q2, interpolated to quarterly) |
| Bank asset quality and profitability | `NPLs`, `Specific provisions`, `Provisions_to_NPLs`, `ROE`, `Total assets` | IMF FSI, Bank of Italy |
| Credit | `PNFC_Lending_K2020`, `HH_Lending_K2020`, `Volume_PNFC`, `Volume_HH` | Bank of Italy, ECB |
| Lending rates and spreads | `PNFC_Spread`, `HH_Spread` | ECB MIR statistics; spread = rate on new loans less three-month Euribor |
| National accounts | `GDP_K2020`, `C_K2020`, `I_K2020`, `G_K2020`, `X_K2020`, `M_K2020` | ISTAT, chain-linked volumes, 2020 reference |
| Labour | `Ur15_74`, `Emp` | ISTAT |
| Sovereign yields | `GrossYield_BTP3/5/10/30`, `GrossYield_CCT`, `GrossYiled_Bund10`, `Spread_BTP_BUND` | Refinitiv / market data |
| Exchange rates and equities | `GBP_EUR`, `JPY_EUR`, `USD_EUR`, `FTITLMS` | Refinitiv / market data |
| Monetary policy | `Shadow_rate` | shadow short rate for the euro area |

The variable name `GrossYiled_Bund10` carries a typo from the original data
file; it is kept unchanged so that specifications written against the authors'
replication package transfer verbatim.

### Terms

The series are compiled from official statistical publications (IMF, ISTAT,
Bank of Italy, ECB) and licensed market data. The compilation is redistributed
here for research reproducibility, following the authors' own public release of
the replication package.

If you intend to redistribute this dataset further, or to use it commercially,
check the terms of the underlying providers — in particular the market-data
series (Refinitiv) — and cite the paper above as the source of the compilation.

To use the package without the bundled data, simply do not call
`load_macroprudential()`; every estimator accepts plain NumPy arrays.
