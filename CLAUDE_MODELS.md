# Smile Models Guide

## SVI (current, active)
- Raw SVI: `w(k) = a + b(ρ(k-m) + √((k-m)² + σ²))` where w = σ²T, k = log(K/F)
- 5 params: {a, b, rho, m, sigma} — level, wings, skew, shift, curvature
- Fitting: `scipy.optimize.least_squares` with ATM-weighted residuals, bounded (a∈[0,0.1], b∈[0,0.5], etc.)
- Filter: remove |k|>0.10, outliers via local smoothness check (`filter_quotes_for_fit`)
- Forward anchoring: `svi_iv_at_strikes` always uses `svi_params["forward"]`, not caller's forward
- Files: `backend/engine/svi.py`

## CDF & LQD from SVI
- Analytical Breeden-Litzenberger: `dC/dK = -e^{-rT}N(d2) + vega·dσ/dK` (no numerical diff)
- `dσ/dK` from SVI closed-form: `dw/dk = b(ρ + (k-m)/√((k-m)²+σ²))`
- CDF inversion: PchipInterpolator (monotone), Q'(u) from Pchip derivative
- Boundary handling: blend Pchip Q' with BS normal Q' where CDF coverage is thin
- Normalisation: median-centered `x = (Q-median)/σ√T`, LQD `ψ̃ = log(Q') - log(σ√T)`
- Asymmetric strike grid width: left side wider when ρ < 0 (skew extends left tail)
- File: `backend/engine/prior.py` → `_svi_to_cdf_lqd()`

## SVI Slider Control
- Frontend: `SviSliders.tsx` with ranges centered on base (fitted) values
- Backend: `POST /prior/{ticker}/svi-override` and `POST /smile/{ticker}/svi-override`
- Half-widths: a±0.01, b±0.1, ρ±0.2, m±0.05, σ±0.1

## Prior Calibration
- Fit SVI to prev-close IVs (`lastPrice - change` from yfinance)
- BS flat base quantile for LQD machinery (legacy, kept for compatibility)
- Save/load: `priors/{TICKER}_prior.json` with strikes, IVs, exclusions, additions, SVI params
- `fit_lqd_prior()` returns: {psi0, Q, Q_tilde, m, s, _bs_base, _svi_params, _fit_strikes}

## LQD Expansion (legacy, used minimally)
- Basis: φ₀=1, φ₁=-log(u), φ₂=-log(1-u), φ₃₊=Legendre on [-1,1]
- M=5 basis functions on Beta(0.7,0.7) quantile grid (200 points)
- Known issue: columns nearly collinear in IV space → large cancelling betas
- Role: only for the normal equations solve (which feeds the propagation matrix P)

## Planned Models
- **Extended SVI (eSSVI/SSVI)**: inter-expiry consistency, calendar arbitrage free
- **Polynomial of Sigmoids**: `σ(k) = Σ wᵢ·sigmoid(aᵢk+bᵢ) + c`, flexible, needs butterfly constraint
- **Arbitrage checks**: density ≥ 0 (butterfly), total variance non-decreasing in T (calendar)
