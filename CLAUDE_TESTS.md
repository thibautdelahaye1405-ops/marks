# Testing Guide

## Current Tests
- `backend/tests/test_pipeline.py` — basic pipeline smoke test
- `backend/tests/test_paper_example.py` — paper's example data

## Test Strategy

### Unit Tests (backend)
- **SVI**: fit_svi correctness (known params → fit → recover), svi_iv_at_strikes round-trip, filter_quotes_for_fit edge cases
- **CDF/LQD**: _svi_to_cdf_lqd monotonicity, CDF(0)=0.5, LQD smoothness, various SVI shapes (steep skew, high convexity, flat)
- **Propagation encoding**: _svi_encode/_svi_decode round-trip, proportional vs absolute correctness
- **Prior save/load**: save → load → compare SVI params, exclusions preserved, additions preserved
- **Graph**: W construction (row sums ≤ 1, diagonal = 0), propagation matrix P, partition correctness

### Integration Tests (backend)
- Full pipeline: fetch mock quotes → calibrate → fit → propagate → verify IVs reasonable
- Quote exclusion: remove a quote → refit → verify SVI changes
- Quote addition: add synthetic point → refit → verify curve passes through it
- Prior load: save with modifications → load → verify modifications reproduced

### Frontend Tests
- Component: SviSliders renders correct values, HoldButton fires repeatedly
- Store: fit() vs propagate() state transitions, _propagateSeq race condition
- Plot: double-click detection timing, customdata mapping for quote exclusion

### Regression / Snapshot Tests
- Known SPY-like SVI params → compute CDF/LQD → compare to saved reference
- Known W matrix + observed deltas → propagate → compare to saved reference
- Guard against accidental changes to numerical code

## Running Tests
```bash
cd marks
python -m pytest backend/tests/ -v
cd frontend && npx vitest run  # (when frontend tests are added)
```

## CI (planned)
- GitHub Actions: Python tests + TypeScript type check on every PR
- Lint: ruff (Python), eslint (TypeScript)
- No deployment (desktop app, not hosted)
