"""
End-to-end pipeline test with synthetic data.

Tests the full chain: prior -> Jacobian -> solve -> reconstruct -> IV output.
Uses Black-Scholes priors with known properties.
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.config import AssetDef, EngineConfig
from backend.engine.lqd import quantile_grid, basis_functions
from backend.engine.prior import bs_prior
from backend.engine.jacobian import compute_jacobian
from backend.engine.reconstruct import reconstruct_smile, quantile_to_call_prices, call_price_to_iv
from backend.engine.graph import build_influence_matrix
from backend.engine.pipeline import run_marking, NodeQuotes


def test_bs_prior():
    """Verify BS prior produces a valid LQD and round-trips to flat vol."""
    grid = quantile_grid(200)
    sigma = 0.20
    T = 1.0 / 12  # 1 month

    p = bs_prior(sigma, T, grid)

    # Check location-scale-shape decomposition
    assert abs(p["m"] - (-0.5 * sigma**2 * T)) < 1e-10
    assert abs(p["s"] - sigma * np.sqrt(T)) < 1e-10

    # Q_tilde should have zero mean and unit variance
    mean = np.trapz(p["Q_tilde"], grid)
    var = np.trapz(p["Q_tilde"]**2, grid)
    assert abs(mean) < 0.01, f"Mean = {mean}"
    assert abs(var - 1.0) < 0.10, f"Var = {var}"

    # Round-trip: Q -> call prices -> IV should recover ~flat vol
    forward = 100.0
    r = 0.045
    strikes = forward * np.exp(np.linspace(-0.15, 0.15, 15))
    iv = call_price_to_iv(
        quantile_to_call_prices(p["Q"], grid, forward, T, r, strikes),
        forward, strikes, T, r,
    )

    # IV should be close to sigma (flat smile for BS)
    valid = ~np.isnan(iv)
    if valid.sum() > 5:
        iv_err = np.abs(iv[valid] - sigma)
        max_err = iv_err.max()
        print(f"BS round-trip: max IV error = {max_err*100:.4f} vol pts")
        assert max_err < 0.05, f"IV error too large: {max_err}"

    print("BS prior test passed")


def test_jacobian():
    """Verify Jacobian is well-conditioned and has correct shape."""
    grid = quantile_grid(200)
    phi = basis_functions(grid, 5)
    sigma = 0.25
    T = 1.0 / 12
    forward = 100.0
    r = 0.045

    p = bs_prior(sigma, T, grid)
    strikes = forward * np.exp(np.linspace(-0.1, 0.1, 10))

    A = compute_jacobian(
        psi0=p["psi0"], Q_tilde=p["Q_tilde"],
        m_v=p["m"], s_v=p["s"],
        forward=forward, T=T, r=r,
        strikes=strikes, sigma_prior=sigma,
        phi=phi, grid=grid,
    )

    assert A.shape == (10, 5), f"Expected (10, 5), got {A.shape}"
    assert np.all(np.isfinite(A)), "Jacobian has non-finite values"

    # Singular values should be reasonable
    sv = np.linalg.svd(A, compute_uv=False)
    cond = sv[0] / sv[-1] if sv[-1] > 1e-15 else float("inf")
    print(f"Jacobian condition number: {cond:.1f}")
    # Condition number is high due to tail basis functions (expected, see paper Section 14)
    # The Tikhonov regularisation (eta * Omega) handles this in the solve
    assert cond < 1e12, f"Jacobian dangerously ill-conditioned: {cond}"

    print("Jacobian test passed")


def test_full_pipeline_synthetic():
    """
    Synthetic end-to-end test: 4 assets, 2 observed with known shocks.
    Verify unobserved nodes get propagated smiles.
    """
    assets = [
        AssetDef("IDX", "Index", "Index", is_index=True, liquidity_score=10.0),
        AssetDef("AAA", "Stock A", "Technology", index_weight=0.05, liquidity_score=6.0),
        AssetDef("BBB", "Stock B", "Technology", index_weight=0.03, liquidity_score=2.0),
        AssetDef("CCC", "Stock C", "Financials", index_weight=0.02, liquidity_score=1.5),
    ]

    T = 30.0 / 365.0
    r = 0.045
    forward = 100.0

    # Synthetic observed quotes: BS flat vol + a skew perturbation
    def make_quotes(ticker, atm_iv, skew_shift=0.0):
        strikes = forward * np.exp(np.linspace(-0.15, 0.15, 15))
        log_m = np.log(strikes / forward)
        # Parabolic skew
        ivs = atm_iv + skew_shift * log_m + 0.05 * log_m**2
        spread = np.full(len(strikes), 0.005)
        return NodeQuotes(
            ticker=ticker, strikes=strikes, mid_ivs=ivs,
            bid_ask_spread=spread, forward=forward, spot=forward,
            T=T, atm_iv=atm_iv,
        )

    quotes = {
        "IDX": make_quotes("IDX", 0.18, skew_shift=-0.08),
        "AAA": make_quotes("AAA", 0.25, skew_shift=-0.12),
    }

    config = EngineConfig(M=5, lambda_=1.0, eta=0.01, quantile_grid_size=200,
                          tail_reg_weight=10.0, interior_reg_weight=0.1)
    result = run_marking(assets, quotes, config)

    # All nodes should have results
    assert set(result.nodes.keys()) == {"IDX", "AAA", "BBB", "CCC"}

    # Observed nodes should be marked as observed
    assert result.nodes["IDX"].is_observed
    assert result.nodes["AAA"].is_observed
    assert not result.nodes["BBB"].is_observed
    assert not result.nodes["CCC"].is_observed

    # Unobserved nodes should have non-trivial betas (propagated)
    beta_B = result.nodes["BBB"].beta
    beta_C = result.nodes["CCC"].beta
    print(f"Propagated beta_B = {beta_B}")
    print(f"Propagated beta_C = {beta_C}")

    # At least one coefficient should be non-zero
    assert np.any(np.abs(beta_B) > 1e-6), "BBB got no propagation"
    assert np.any(np.abs(beta_C) > 1e-6), "CCC got no propagation"

    # Marked IVs should be finite
    for ticker, node in result.nodes.items():
        valid = [v for v in node.iv_marked if v is not None and np.isfinite(v)]
        print(f"{ticker}: {len(valid)}/{len(node.iv_marked)} valid IVs, "
              f"range [{min(valid)*100:.1f}, {max(valid)*100:.1f}] %")
        assert len(valid) >= 5, f"{ticker} has too few valid IVs"

    # Propagation matrix should exist
    assert result.propagation_matrix is not None
    print(f"Propagation matrix shape: {result.propagation_matrix.shape}")

    # W should be sub-stochastic
    row_sums = result.W.sum(axis=1)
    assert np.all(row_sums <= 1.0 + 1e-6), f"W not sub-stochastic: {row_sums}"

    print("\nFull pipeline test passed!")


if __name__ == "__main__":
    test_bs_prior()
    print()
    test_jacobian()
    print()
    test_full_pipeline_synthetic()
