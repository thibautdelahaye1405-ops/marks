# Graph & Propagation Guide

## Influence Matrix W
- `W[i,j]` = how much node j influences node i (j→i, asymmetric)
- Properties: W[i,i]=0, W≥0, row sums ≤ 1 (sub-stochastic)
- Self-trust: α_i = 1 - Σ_j W[i,j]
- Construction: `build_influence_matrix()` in `backend/engine/graph.py`
  - Correlation: sector-based `get_sector_correlation(s1, s2)` from `config.py` (no hardcoded pairwise)
  - Base: `|ρ_ij| · ℓ_j/(ℓ_i+ℓ_j)` (correlation × liquidity asymmetry, no sqrt for stronger differentiation)
  - Boosts: index→constituent only (+index_weight×10), same sector (×1.2)
  - Self-trust: α_i = α_min + (α_max - α_min) × ℓ_i/ℓ_max (continuous, linear in liquidity)
    - SPY (ℓ=10): α=0.90 (mostly self-trusting, 10% neighbor influence)
    - Small stock (ℓ=2): α=0.26 (74% pulled by neighbors)
  - Row normalisation: `W[i,j] = (1-α_i) · W_raw[i,j] / Σ_w W_raw[i,w]`

## Propagation Matrix P
- `P = (I - W_UU)⁻¹ W_UO` (eq.33 from the paper)
- Partitioned from W into observed (O) / unobserved (U) submatrices
- P[i,j] = total influence of observed node j on unobserved node i (includes multi-hop)
- Neumann series: P = W_UO + W_UU·W_UO + W_UU²·W_UO + ... (5 terms for animation)
- File: `backend/engine/graph.py` → `propagation_matrix()`, `neumann_series_terms()`

## SVI-Space Propagation (current approach)
The propagation operates in a mixed proportional/absolute encoding of SVI parameter changes.

### Encoding: g = encode(market_svi, prior_svi)
| Param | Mode | Formula | Rationale |
|-------|------|---------|-----------|
| a (level) | proportional | market/prior - 1 | 10% vol increase → 10% for all |
| b (wings) | proportional | market/prior - 1 | scales with asset's own wing level |
| ρ (skew) | absolute | market - prior | bounded [-1,1], absolute shift natural |
| m (shift) | absolute | market - prior | log-moneyness units, comparable |
| σ (curv) | proportional | market/prior - 1 | curvature scales with level |

### Propagation
```
g_observed = [encode(market_svi, prior_svi) for each observed node]
g_unobserved = P @ g_observed    # shape: (N_unobs, 5)
```

### Decoding: new_svi = decode(prior_svi, g_propagated)
- Proportional: `new = prior * (1 + g)` (g capped at ±5x)
- Absolute: `new = prior + g`
- Clamp to fit_svi bounds: a∈[0,0.15], b∈[0,0.50], ρ∈[-0.999,0.999], m∈[-0.20,0.20], σ∈[1e-4,1.0]

### Known limitation
Raw SVI params {a,b,ρ,m,σ} are not orthogonal — `a` mixes variance level with shape.
Proportional encoding on `a` conflates level changes with shape. Next step: SVI-JW
(Jump-Wing) reparametrisation {v_t, ψ_t, p_t, c_t, ṽ_t} which cleanly separates
ATM variance level from shape parameters.

### File: `backend/engine/pipeline.py` → `_svi_encode()`, `_svi_decode()`, Phase A/B/C

## Graph Visualisation
- Deterministic layout: preset positions from `computeInfluenceRanks()` (column sums of W)
- Top = most influential, bottom = leaves, horizontal tiers
- Rebuild only when W changes (not on selection/solve)
- Node styles updated separately via lightweight effect (no graph rebuild)
- Hover: highlight connected edges + show W coefficient labels
- Propagation tab: ego-graph (concentric layout, selected node at center, Viridis color)

## Fit vs Propagate
- **`/api/fit`** (lightweight): SVI fit for observed, prior for unobserved. No graph. Called by all interactive actions.
- **`/api/solve`** (full): LQD normal equations + graph propagation + SVI-space encode/propagate/decode. Called ONLY by Propagate button.

## Planned
- **SVI-JW propagation**: replace raw SVI encoding with Jump-Wing params for orthogonal level/shape separation
- Multi-expiry: W shared or per-expiry scaled, convex ordering constraint via QP
- Cinematics: Neumann series animation (hop-by-hop influence flow)
