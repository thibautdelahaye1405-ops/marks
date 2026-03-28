# Vol Marking — Development Roadmap

## 1. Forwards & Rates

### 1.1 Forward fitting from put-call parity ⭐ HIGH PRIORITY
- Currently `F = S * exp(rT)` with hardcoded `r=0.045` — wrong for dividend-paying stocks
- Fit forward from put-call parity: `F = K + exp(rT) * (C(K) - P(K))` at liquid ATM strikes
- This fixes all downstream SVI fits (currently shifted due to wrong forward)
- Fallback: use hardcoded r when put data is unavailable

### 1.2 Interest rate curve
- Default: flat rate from user input (current: 0.045)
- Source selection: manual input, Treasury curve (FRED API), OIS
- Term structure: interpolate to each expiry's maturity
- Store in referential config, per-currency

### 1.3 Dividends and repo
- Discrete dividends for single stocks (ex-dates + amounts)
- Continuous dividend yield for indices (SPY, QQQ)
- Sources: manual input, Yahoo Finance (limited), Bloomberg (future)
- Forward: `F = S * exp((r - q)T) - PV(discrete divs)`
- Repo rate: typically zero for equities, but configurable

### 1.4 Forward validation
- Display fitted forward vs model forward on smile charts
- Warning indicator when difference > 0.5%
- Option to lock forward to market-implied value

---

## 2. Referential & Connectors

### 2.1 Selectable universe ⭐ NEXT
- Expand `config.py` to ~100 liquid names (S&P constituents + major ETFs)
- Multi-select UI in Referential panel (currently stubbed)
- Sector-based correlation defaults (intra ~0.6-0.8, cross ~0.3-0.5)
- Text input to add arbitrary tickers to the universe

### 2.2 Multi-source architecture
- Per-ticker source selection for: spot price, options chain, dividends
- Default source: Yahoo Finance
- Future sources: Bloomberg, Refinitiv, CSV upload, custom API
- Source abstraction layer: `DataSource` interface with `fetch_spot()`, `fetch_chain()`, `fetch_dividends()`

### 2.3 Interest rate curve source
- FRED API for Treasury rates
- Manual override per tenor point
- CSV upload for custom curves

### 2.4 Expiry management
- Currently: single expiry (30-day target)
- Multi-expiry: select multiple expiries per asset
- Source: from option chain available dates
- UI: expiry selector in the asset detail panel

---

## 3. Smile Models

### 3.1 SVI refinements
- Current: raw SVI with 5 params, bounded optimisation
- Extended SVI (eSSVI): adds inter-expiry consistency, natural for multi-maturity
- SSVI (Surface SVI): Gatheral-Jacquier parametrisation ensuring no calendar arbitrage
- Better initial guess from data moments

### 3.2 Polynomial of Sigmoids
- Flexible parametric model: `σ(k) = Σ_i w_i * sigmoid(a_i * k + b_i) + c`
- Naturally bounded (no negative variance)
- Needs butterfly arbitrage constraint (density ≥ 0)
- Number of sigmoids as a model order parameter

### 3.3 LQD expansion improvements
- Current: 5 basis functions (constant, 2 tail, 2 Legendre)
- Add 2 more Legendre polynomials (M=7) for better interior resolution
- Better boundary basis: replace -log(u) with smoother tail functions?
- Role is now limited to propagation encoding (display uses SVI)

### 3.4 Model selection UI
- Radio group in Modelling panel (SVI active, others greyed → implement)
- Per-asset model choice (most assets SVI, some could use sigmoid)
- Model-agnostic propagation: encode/decode pattern generalises across models

---

## 4. Propagation

### 4.1 Current: SVI-space propagation ✅ DONE
- Encode: proportional for (a, b, σ), absolute for (ρ, m)
- Propagate: P = (I - W_UU)⁻¹ W_UO applied to 5-vectors
- Decode: apply to each unobserved prior SVI

### 4.2 Butterfly arbitrage safeguard
- After propagation, check density non-negativity: `d²C/dK² ≥ 0`
- Visual indicator on graph nodes (red = arbitrage violation)
- Optional: project onto nearest arbitrage-free SVI (constrained optimisation)

### 4.3 Multi-expiry propagation
- Propagation matrix W shared across expiries (same influence graph)
- Per-expiry SVI delta encoding → propagate independently per expiry
- Convex ordering constraint: total variance τ(k,T) non-decreasing in T
- QP optimisation: minimise distance to unconstrained solution subject to calendar constraints
- Tensor structure: W ⊗ I_T or more nuanced per-expiry W scaling

### 4.4 Cinematics / animation
- Neumann series visualisation: show hop-by-hop influence flow
- "Current flowing through graph" animation after propagation
- Per-hop contribution to each unobserved node's SVI delta

---

## 5. Influence Matrix

### 5.1 New asset defaults
- When adding an asset to observed set, default W weights = 0 (conservative)
- Sector-based template: auto-populate from sector average weights as suggestion

### 5.2 Matrix editing improvements
- Worksheet-like copy-paste (bulk edit rows/columns)
- "Copy weights from similar asset" one-click action
- Import/export W matrix as CSV
- Constraint validation: row sums ≤ 1, diagonal = 0, non-negative

### 5.3 Automatic W estimation
- Estimate W from historical correlation of IV changes (not just spot correlation)
- Liquidity-weighted asymmetric influence (liquid → illiquid)
- Index-constituent boosting

---

## 6. Arbitrage & Validation ⭐ HIGH PRIORITY

### 6.1 Butterfly arbitrage check
- Compute risk-neutral density from marked smile
- Flag negative density regions (red highlight on chart)
- Node-level indicator on the graph (green/yellow/red)

### 6.2 Calendar spread arbitrage (multi-expiry)
- Total variance must be non-decreasing in T
- Flag violations across expiry pairs
- QP projection onto arbitrage-free surface

### 6.3 Put-call parity check
- Verify consistency between call and put markets
- Flag large deviations

---

## 7. UX & Workflow

### 7.1 Undo/history ⭐ HIGH PRIORITY
- Stack of prior states (quote exclusions, SVI overrides, W changes)
- Ctrl+Z to revert, Ctrl+Shift+Z to redo
- History panel showing recent actions

### 7.2 Export
- Marked surfaces as CSV/JSON (strikes × IVs per asset per expiry)
- W matrix export
- Prior save/load (partially done ✅)
- Snapshot comparison: diff two marking sessions

### 7.3 Audit trail
- Log of marking actions: who, when, what changed
- Per-asset modification history
- Compliance-friendly export

### 7.4 Keyboard shortcuts
- Space: propagate
- Escape: deselect node / close panel
- Arrow keys: navigate assets
- +/-: fine-tune selected slider

---

## 8. Dev Workflow & Packaging

### 8.1 Executable package
- Electron or Tauri wrapping Vite frontend + bundled Python backend (PyInstaller)
- Single-click install for Windows/Mac
- Auto-update mechanism

### 8.2 Sub-agent structure
- `CLAUDE_UX.md` — layout, components, interactions, styling
- `CLAUDE_MODELS.md` — SVI, LQD, sigmoid, arbitrage checks, fitting
- `CLAUDE_GRAPH.md` — propagation, influence matrix, graph algorithms, cinematics
- `CLAUDE_DATA.md` — sources, referential, persistence, forwards, rates, dividends
- `CLAUDE_TESTS.md` — test strategy, fixtures, integration tests, CI

### 8.3 Testing
- Unit tests: SVI fit, propagation encode/decode, CDF/LQD computation
- Integration tests: full pipeline (fetch → fit → propagate → display)
- Snapshot tests: regression on known market data
- Frontend: component tests for sliders, quote editing, modal flows

### 8.4 GitHub
- Branch strategy: main (stable) + feature branches
- PR workflow with CI checks
- Release tags for packaged versions
