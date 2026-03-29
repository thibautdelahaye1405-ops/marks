# Vol Marking — Development Roadmap

## 1. Forwards & Rates ✅ DONE

### 1.1 Forward fitting from put-call parity ✅
- Fetches both calls and puts from Yahoo Finance
- Put-call parity: `F = K* + exp(rT)(C(K*) - P(K*))` where K* minimises |C-P|
- Parity forward shown as reference (dashed gray line), model forward used for fitting
- Consistent: both prior (F_prev) and current smile use model forward

### 1.2 Interest rate curve ✅
- FRED Treasury curve: 11 tenors (1M to 30Y), fetched from public CSV endpoint
- Log-maturity interpolation to each expiry's T
- 1-hour cache, fallback to 4.5% flat if FRED unavailable
- `config.risk_free_rate`: None = Treasury curve (default), float = flat override

### 1.3 Dividends and repo ✅
- Continuous dividend yield from Yahoo Finance (`trailingAnnualDividendYield`)
- Discrete dividends for stocks: projected from historical pattern
- PV of discrete dividends: `sum(d_i * exp(-r(T_i)*T_i))`
- Repo: GC rate (default 0%) + per-ticker overrides for hard-to-borrow
- Full forward: `F = S * exp((r - q - repo)*T) - PV(discrete divs)`

### 1.4 Forward validation ✅
- Dual forward lines on charts: model (amber) + parity reference (gray dashed)
- Info bar: F, r, q, repo, parity-vs-model % with red/green colour coding
- Forward slider with +/- buttons in both Smile and Prior tabs (independent)
- SVI sliders track forward changes and revert on reset

---

## 2. Referential & Connectors

### 2.1 Selectable universe ✅ DONE
- 101 tickers across 12 GICS-aligned sectors in `config.py` CATALOG
- Searchable multi-select UI in Referential panel, grouped by sector with collapse/expand
- Sector-based correlation matrix (replaces hardcoded pairwise correlations): intra ~0.60-0.80, cross ~0.20-0.50
- Text input to add arbitrary tickers — validates on Yahoo Finance before accepting
- Persistence: `selections/default.json` for active universe, `catalog_custom.json` for user-added tickers
- Save button with dirty-flag tracking; auto-loads previous selection on restart
- W matrix rebuilt automatically on universe change

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

### 3.1 SVI-JW Normalised ✅ DONE
- Native parameterisation: {v, ψ̂, p̂, ĉ, ṽ/v} — ATM variance, normalised skew, put/call wings, min-var ratio
- Normalisation: z = k/√(vτ) makes shape params stationary across term structure and vol regimes
- Raw SVI {a,b,ρ,m,σ} is internal evaluation detail only — all UI/API uses JW
- Bijective conversion raw↔JW with round-trip error at machine epsilon
- Sliders: ATM Var (%), Skew (ψ̂), Put Wing (p̂), Call Wing (ĉ), Min-Var Ratio
- Reference: `SVI-JW-normalized.tex`, `backend/engine/svi.py`

### 3.2 LQD model ⭐ NEXT
- LQD-native smile model as alternative to SVI
- Basis expansion on quantile derivative: ψ(u) = ψ₀(u) + Σ βₘ φₘ(u)
- Improved basis functions (M=7?), better tail handling
- Model-agnostic propagation: same encode/decode pattern as SVI-JW

### 3.3 Polynomial of Sigmoids ⭐ NEXT
- Flexible parametric model: `σ(k) = Σ_i w_i * sigmoid(a_i * k + b_i) + c`
- Naturally bounded (no negative variance)
- Needs butterfly arbitrage constraint (density ≥ 0)
- Number of sigmoids as a model order parameter

### 3.4 Model selection UI
- Radio group in Modelling panel (SVI-JW active, others greyed → implement)
- Per-asset model choice (most assets SVI-JW, some could use sigmoid or LQD)
- Model-agnostic propagation: encode/decode pattern generalises across models

---

## 4. Propagation

### 4.1 SVI-JW normalised propagation ✅ DONE
- Two channels through same P matrix: level (v only) and shape (ψ̂, p̂, ĉ, ṽ/v)
- Log-ratio encoding for positive params (v, p̂, ĉ, ṽ/v) — geometric averaging via P matrix
- Clamped-reference encoding for skew (ψ̂): `Δψ̂/max(|ψ̂_prior|, ε)` with ε=0.1 — handles sign changes and near-zero skew gracefully
- All encoded values capped at ±2.0
- Decode: exp for log-ratio params, linear for skew, then JW→raw SVI for evaluation
- File: `backend/engine/pipeline.py` (_jw_encode/_jw_decode)

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

### 5.1 New asset defaults ✅ DONE
- Sector-based correlation matrix drives W weights automatically for any universe size
- Self-trust α scales continuously with liquidity: α_min=0.10 (illiquid) to α_max=0.90 (most liquid)
- Liquidity asymmetry: `W[i,j] ∝ ℓ_j/(ℓ_i+ℓ_j)` (no sqrt — stronger differentiation)
- Index→constituent boost (one-directional); same-sector ×1.2 affinity

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
