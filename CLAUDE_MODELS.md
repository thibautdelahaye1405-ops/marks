# Smile Models Guide

## SVI-JW Normalised (current, active — native parameterisation)

### Normalised Parameters
- `v` — ATM implied variance (σ²_ATM), the level parameter
- `ψ̂` — normalised ATM skew: `ψ̂ = ψ · √(τ/v)`, stationary across term structure
- `p̂` — normalised put wing slope: `p̂ = b(1-ρ)/√(vτ)`, always > 0
- `ĉ` — normalised call wing slope: `ĉ = b(1+ρ)/√(vτ)`, always > 0
- `ṽ/v` — min-variance ratio: `min_k v(k) / v(0)`, in [0, 1]

### Conversion to/from Raw SVI
- `raw_svi_to_jw_normalized(a, b, rho, m_svi, sigma, T)` → {v, vt_ratio, psi_hat, p_hat, c_hat}
- `jw_normalized_to_raw_svi(v, vt_ratio, psi_hat, p_hat, c_hat, T)` → {a, b, rho, m, sigma}
- Round-trip exact to machine epsilon (~1e-16)
- Key identities: `ρ = (ĉ-p̂)/(ĉ+p̂)`, `b = ½(ĉ+p̂)√(vτ)`, `β = m/√(m²+σ²)`
- Reference: `SVI-JW-normalized.tex`
- File: `backend/engine/svi.py`

### Raw SVI (internal evaluation only)
- `w(k) = a + b(ρ(k-m) + √((k-m)² + σ²))` where w = σ²T, k = log(K/F)
- Fitting: `scipy.optimize.least_squares` with ATM-weighted residuals, bounded
- Filter: remove |k|>0.10, outliers via local smoothness check (`filter_quotes_for_fit`)
- Forward anchoring: `svi_iv_at_strikes` always uses `svi_params["forward"]`

### Slider Control
- Frontend: `SviSliders.tsx` — JW params with ranges centered on fitted values
- Labels: ATM Var (v) shown as %, Skew (ψ̂), Put Wing (p̂), Call Wing (ĉ), Min-Var Ratio
- Backend: `POST /prior/{ticker}/svi-override` and `POST /smile/{ticker}/svi-override`
- API accepts JW params, converts to raw SVI internally for evaluation

### Propagation Encoding
- Two channels through same P matrix: level (v) and shape (ψ̂, p̂, ĉ, ṽ/v)
- v, p̂, ĉ, ṽ/v: log-ratio encoding → geometric averaging
- ψ̂: clamped-reference encoding `(ψ̂_mkt - ψ̂_prior) / max(|ψ̂_prior|, 0.1)` — handles sign changes
- All capped at ±2.0 in encoded space
- File: `backend/engine/pipeline.py` (_jw_encode/_jw_decode)

## CDF & LQD from SVI
- Analytical Breeden-Litzenberger: `dC/dK = -e^{-rT}N(d2) + vega·dσ/dK` (no numerical diff)
- `dσ/dK` from SVI closed-form: `dw/dk = b(ρ + (k-m)/√((k-m)²+σ²))`
- CDF inversion: PchipInterpolator (monotone), Q'(u) from Pchip derivative
- Boundary handling: blend Pchip Q' with BS normal Q' where CDF coverage is thin
- Normalisation: median-centered `x = (Q-median)/σ√T`, LQD `ψ̃ = log(Q') - log(σ√T)`
- Asymmetric strike grid width: left side wider when ρ < 0 (skew extends left tail)
- File: `backend/engine/prior.py` → `_svi_to_cdf_lqd()`

## Prior Calibration
- Fit SVI to prev-close IVs (`lastPrice - change` from yfinance)
- JW params stored as `_jw_params` on prior dict alongside `_svi_params`
- BS flat base quantile for LQD machinery (legacy, kept for compatibility)
- Save/load: `priors/{TICKER}_prior.json` with strikes, IVs, exclusions, additions, SVI params
- `fit_lqd_prior()` returns: {psi0, Q, Q_tilde, m, s, _bs_base, _svi_params, _jw_params, _fit_strikes}

## LQD Expansion (legacy, used minimally)
- Basis: φ₀=1, φ₁=-log(u), φ₂=-log(1-u), φ₃₊=Legendre on [-1,1]
- M=5 basis functions on Beta(0.7,0.7) quantile grid (200 points)
- Known issue: columns nearly collinear in IV space → large cancelling betas
- Role: only for the normal equations solve (which feeds the propagation matrix P)

## Planned Models
- **LQD model**: LQD-native smile as alternative to SVI, improved basis (M=7?), better tails
- **Polynomial of Sigmoids**: `σ(k) = Σ wᵢ·sigmoid(aᵢk+bᵢ) + c`, flexible, needs butterfly constraint
- **Model selection UI**: per-asset model choice, model-agnostic propagation via encode/decode pattern
- **Arbitrage checks**: density ≥ 0 (butterfly), total variance non-decreasing in T (calendar)
