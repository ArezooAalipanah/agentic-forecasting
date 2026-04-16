# Food Price CPI Forecasting

Replicates and extends the **Canada's Food Price Report (CFPR)** forecasting
methodology — an annual 18-month-ahead prediction of Canadian food price changes
across 9 CPI sub-categories.

---

## Forecasting task

**Target variable:** Consumer Price Index (CPI) for food products and sub-categories
in Canada (index, 2002=100).  Year-over-year percentage changes are derived from
index forecasts at reporting time.

**Target categories (9):**

| Series ID | Description |
|-----------|-------------|
| `cpi_food_canada` | Overall Food (headline) |
| `cpi_bakery_cereal_canada` | Bakery and cereal products (excl. baby food) |
| `cpi_dairy_eggs_canada` | Dairy products and eggs |
| `cpi_fish_seafood_canada` | Fish, seafood and other marine products |
| `cpi_restaurants_canada` | Food purchased from restaurants |
| `cpi_fruit_preparations_nuts_canada` | Fruit, fruit preparations and nuts |
| `cpi_meat_canada` | Meat |
| `cpi_other_food_nonalcoholic_canada` | Other food products and non-alcoholic beverages |
| `cpi_vegetables_preparations_canada` | Vegetables and vegetable preparations |

**Data source:** Statistics Canada table 18-10-0004-11.  Populated via
`scripts/fetch_cpi.py`.

---

## CFPR methodology

The CFPR is published each November/December.  By that point, the July CPI release
is the most recent data available.  The report projects food prices **18 months
ahead** (to approximately January of the following year).

This experiment replicates that discipline exactly:

- **Horizon:** 18 months.
- **Origins:** July 1 of each year (annual stride).
- **Backtest window:** July 2009 → July 2024 — 16 annual origins spanning three
  distinct macro regimes: low-inflation (2010–19), COVID shock (2020–21), and the
  food-price surge and retreat (2021–24).
- **Protected eval window:** July 2021 → July 2024 — the four most recent origins,
  all resolvable by April 2026.  Budget-limited to 5 `multi_evaluate()` calls.
- **Information cutoff:** at each origin, the predictor can only see data with
  `timestamp ≤ origin`, enforced by `ForecastContext.as_of`.

---

## Exogenous covariates (FRED)

Five monthly FRED series are available for predictors that consume exogenous inputs.
Populated via `scripts/fetch_fred.py`.  All are monthly frequency (MS) — matching
the StatCan target — so no resampling is needed.

| Series ID | FRED ID | Description |
|-----------|---------|-------------|
| `fred_us_cpi_food_at_home` | CPIFABSL | US CPI: Food at Home |
| `fred_us_cpi_meats_poultry_fish_eggs` | CUSR0000SAF112 | US CPI: Meats, Poultry, Fish, Eggs |
| `fred_us_cpi_fruits_vegetables` | CUSR0000SAF113 | US CPI: Fruits and Vegetables |
| `fred_canada_10yr_bond_yield` | IRLTLT01CAM156N | Canada 10-year government bond yield |
| `fred_canada_us_exchange_rate` | EXCAUS | Canada/US exchange rate (CAD per USD) |

**Note:** `FRED_API_KEY` must be set in `.env` before running `scripts/fetch_fred.py`.
The notebook reads from the local parquet cache — no API key needed at run time.

---

## Reference specs

```
reference_specs/food_cpi/
├── food_cpi_18m_backtest.yaml   # MultiTargetBacktestSpec — July 2009–2024, 16 origins
└── food_cpi_18m_eval.yaml       # MultiTargetEvalSpec — July 2021–2024, 4 origins, max_runs=5
```

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `food_data_exploration.ipynb` | Register and inspect all series; date ranges, gaps, seasonality, full history plots and FRED covariate visualisation |
| `food_cpi_experiment.ipynb` | **Main experiment:** CFPR-replica backtest for one selectable category — 16 July origins, 4 predictors, disaggregated error plots, CRPS + MAPE tables, YoY derivation, protected eval |

---

## Prerequisites

```bash
uv run python scripts/fetch_cpi.py
uv run python scripts/fetch_fred.py   # requires FRED_API_KEY in .env
```

---

## Key design decisions

- **One category at a time.** The notebook focuses the analysis on a single `CATEGORY_ID`
  (change one variable to switch categories).  Running all 9 simultaneously is possible
  via `multi_backtest` but runs slowly with the ARIMA and regression models.
- **Forecast target is the raw CPI index.** YoY % change is derived at reporting time:
  `yoy_pct = (forecast − obs_12m_before_forecast_date) / obs_12m_before_forecast_date × 100`.
- **No ensemble model selection.** We compare individual methods on CRPS + MAPE (median).
- **CRPS is the primary metric.** MAPE on the median is reported as a familiar
  secondary metric, not the ranking criterion.
- **Exogenous inputs are predictor-level.** Tasks define the question; predictors decide
  whether and how to use FRED covariates.
