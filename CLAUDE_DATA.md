# Data Layer Guide

## Quote Fetching
- Source: Yahoo Finance via `yfinance` library
- File: `backend/data/quotes.py` → `fetch_option_chain()`
- Target: 30-day maturity, min 7 days to expiry, max 30 strikes centered on ATM
- Market detection: if >30% bids are zero → after-hours mode (use lastPrice, 2% spread assumption)
- IV computation: own Brent inversion from mid prices (more reliable than yfinance IV)
- Previous close: `prev_price = mid_price - change`, IV inverted at prev_spot forward
- Fetches both calls AND puts for put-call parity forward computation

## Forwards & Rates
- **Treasury curve**: `backend/data/rates.py` — FRED public CSV, 11 tenors (1M–30Y), log-interp, 1h cache
- **Dividends**: `backend/data/dividends.py` — Yahoo Finance yield + projected discrete divs
- **Forward formula**: `F = S * exp((r - q - repo) * T) - PV(discrete divs)`
- **Put-call parity**: `F_parity = K* + exp(rT)(C(K*) - P(K*))` where K* minimises |C-P|; shown as reference
- **Model forward** used for all fitting (consistent between prior and current smile)
- **Forward override**: per-ticker, independent for current smile vs prior (stored in `_state`)
- **Repo**: GC rate (default 0%) + per-ticker overrides for hard-to-borrow names

## Data Model
```python
OptionChainData:
    ticker, expiry, T, spot, forward
    strikes, mid_ivs, bid_ask_spread, open_interest, call_mids
    prev_close_ivs, prev_close_atm_iv, prev_spot  # for prior calibration
    forward_parity, forward_model                   # forward validation
    rate_used, div_yield_used, repo_rate_used       # rates breakdown
    put_mids                                        # put mid prices
```

## Persistence
- **Quotes**: SQLite (`db/marks.sqlite`) via `backend/data/store.py`
- **Priors**: JSON files in `priors/` directory via `backend/data/prior_store.py`
  - Format: `{ticker}_prior.json` with SVI params, original strikes/IVs, excluded_indices, added_quotes, rate_used
  - Save: captures chain's prev_close data + frontend exclusions/additions
  - Load: reconstructs full prior dict by refitting SVI from effective quote set (original - excluded + added)

## Backend State (`_state` in routes.py)
```python
_state = {
    "quotes": {},              # ticker → OptionChainData (from latest fetch)
    "W": None,                 # influence matrix (N×N)
    "alphas": None,            # self-trust vector
    "config": EngineConfig(),
    "priors_base": {},         # ticker → immutable base prior
    "priors": {},              # ticker → active prior (base + overrides)
    "solve_result": None,      # last MarkingResult
    "treasury_curve": None,    # TreasuryCurve from FRED
    "dividends": {},           # ticker → DividendInfo
    "forward_overrides": {},   # ticker → absolute forward override (current smile)
    "prev_forward_overrides": {}, # ticker → absolute forward override (prior)
}
```

## Referential
- File: `backend/config.py` (AssetDef, DEFAULT_UNIVERSE, DEFAULT_CORRELATIONS, EngineConfig)
- File: `backend/data/referential.py` (get_universe, get_asset_map)
- Currently: 10 hardcoded assets (SPY + 5 equities + 2 financials + 2 ETFs)
- Correlation fallback: 0.3 for unknown pairs
- `EngineConfig.risk_free_rate`: None = Treasury curve, float = flat override
- `EngineConfig.repo_rate_gc`: GC repo rate (default 0%)
- `EngineConfig.repo_overrides`: per-ticker repo rate overrides

## API Endpoints
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/universe` | GET | List active universe |
| `/api/fetch-quotes` | POST | Yahoo Finance fetch all (incl. puts, divs, rates) |
| `/api/fit` | POST | Lightweight: SVI observed + prior unobserved |
| `/api/solve` | POST | Full: graph propagation |
| `/api/graph` | GET/PUT | Influence matrix W |
| `/api/calibrate-priors` | POST | Fit all priors from prev close |
| `/api/prior/{ticker}` | GET | Distribution view for prior |
| `/api/prior/{ticker}/svi-override` | POST | Override prior SVI params |
| `/api/prior/{ticker}/refit` | POST | Refit with exclusions/additions |
| `/api/smile/{ticker}/svi-override` | POST | Override current smile SVI |
| `/api/node/{ticker}/distribution` | GET | Prior + marked distribution views |
| `/api/priors/saved` | GET | List saved prior files |
| `/api/priors/save/{ticker}` | POST | Save current prior to file |
| `/api/priors/load/{ticker}` | POST | Load prior from file |
| `/api/forward/{ticker}` | PUT | Set/clear forward override (current smile) |
| `/api/forward/{ticker}/prior` | PUT | Set/clear forward override (prior) |
| `/api/rates/treasury` | GET | Current Treasury curve |
| `/api/rates/config` | GET/PUT | Repo rate config |

## Planned
- Multi-source: per-ticker source selection for spot/options/dividends
- Expanded universe: ~100 S&P names with sector-based correlation defaults
