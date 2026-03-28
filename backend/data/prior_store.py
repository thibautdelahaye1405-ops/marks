"""
JSON file-based persistence for prior calibrations.

Saves/loads prior state to {PROJECT_ROOT}/priors/{TICKER}_prior.json so that
a user can resume marking from a previously fitted prior.
"""
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

PRIORS_DIR = Path(__file__).parent.parent.parent / "priors"


def _ensure_dir():
    PRIORS_DIR.mkdir(parents=True, exist_ok=True)


def _to_python(obj):
    """Recursively convert numpy types to plain Python for JSON serialisation."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    return obj


def list_saved_priors() -> Dict[str, dict]:
    """Scan the priors directory for saved prior JSON files.

    Returns:
        dict of ticker -> {filename, timestamp (ISO), ticker}
    """
    _ensure_dir()
    result = {}
    for p in sorted(PRIORS_DIR.glob("*_prior.json")):
        ticker = p.stem.replace("_prior", "")
        try:
            with open(p, "r") as f:
                data = json.load(f)
            result[ticker] = {
                "filename": p.name,
                "timestamp": data.get("timestamp", ""),
                "ticker": ticker,
            }
        except (json.JSONDecodeError, OSError):
            continue
    return result


def save_prior(
    ticker: str,
    prior_dict: dict,
    excluded_indices: Optional[List[int]] = None,
    added_quotes: Optional[List[List[float]]] = None,
    chain=None,
) -> str:
    """Serialise the current prior state for *ticker* to a JSON file.

    Saves the original prev-close strikes/IVs (from the chain), SVI params,
    excluded indices, added quotes, and key scalars.  On reload the SVI is
    refitted from the effective quote set (original minus exclusions plus
    additions) so the saved modifications are faithfully reproduced.

    Returns:
        The filename written (e.g. ``SPY_prior.json``).
    """
    _ensure_dir()

    svi_params = prior_dict.get("_svi_params", {})

    # Original prev-close data from the chain (before any exclusions/additions)
    if chain is not None and hasattr(chain, 'strikes'):
        orig_strikes = chain.strikes
        orig_ivs = chain.prev_close_ivs if chain.prev_close_ivs is not None else chain.mid_ivs
        prev_spot = chain.prev_spot or chain.spot
        T = chain.T
        rate = getattr(chain, 'rate_used', 0.045)
        forward = prev_spot * np.exp(rate * T)
    else:
        orig_strikes = prior_dict.get("_fit_strikes", np.array([]))
        orig_ivs = np.array(svi_params.get("iv_fitted", [])) if svi_params else np.array([])
        forward = float(svi_params.get("forward", 100.0)) if svi_params else 100.0
        T = float(svi_params.get("T", 30 / 365)) if svi_params else 30 / 365

    atm_iv = float(prior_dict.get("s", 0.25)) / max(np.sqrt(T), 1e-10)

    payload = {
        "ticker": ticker,
        "timestamp": datetime.now().isoformat(),
        "svi_params": _to_python({
            "a": svi_params.get("a"),
            "b": svi_params.get("b"),
            "rho": svi_params.get("rho"),
            "m": svi_params.get("m"),
            "sigma": svi_params.get("sigma"),
            "forward": svi_params.get("forward"),
            "T": svi_params.get("T"),
        }) if svi_params else None,
        "strikes": _to_python(orig_strikes),
        "ivs": _to_python(orig_ivs),
        "excluded_indices": excluded_indices or [],
        "added_quotes": _to_python(added_quotes) if added_quotes else [],
        "atm_iv": atm_iv,
        "forward": forward,
        "T": T,
        "rate_used": float(getattr(chain, 'rate_used', 0.045)) if chain is not None else 0.045,
        "bs_m": float(prior_dict.get("m", 0.0)),
        "bs_s": float(prior_dict.get("s", 0.0)),
    }

    filename = f"{ticker}_prior.json"
    filepath = PRIORS_DIR / filename
    with open(filepath, "w") as f:
        json.dump(payload, f, indent=2)

    return filename


def load_prior(ticker: str) -> dict:
    """Load a saved prior from JSON and reconstruct the full prior dict.

    If the saved file contains excluded_indices or added_quotes, the SVI is
    refitted from the effective quote set (original strikes/IVs minus
    exclusions plus additions).  Otherwise the saved SVI params are used
    directly.

    The returned dict has the same structure as ``fit_lqd_prior`` returns.

    Raises:
        FileNotFoundError: if no saved prior exists for *ticker*.
    """
    filepath = PRIORS_DIR / f"{ticker}_prior.json"
    if not filepath.exists():
        raise FileNotFoundError(f"No saved prior for {ticker}")

    with open(filepath, "r") as f:
        data = json.load(f)

    from ..engine.prior import bs_prior, fit_lqd_prior
    from ..engine.lqd import quantile_grid, basis_functions
    from ..config import EngineConfig

    cfg = EngineConfig()
    grid = quantile_grid(cfg.quantile_grid_size)
    phi = basis_functions(grid, cfg.M)

    T = data.get("T", 30.0 / 365.0)
    forward = data.get("forward", 100.0)
    rate_used = data.get("rate_used", 0.045)
    strikes = np.array(data.get("strikes", []), dtype=float)
    ivs = np.array(data.get("ivs", []), dtype=float)
    excluded = data.get("excluded_indices", [])
    added = data.get("added_quotes", [])

    # Reconstruct the effective quote set used for the saved SVI fit
    if len(strikes) > 0 and len(ivs) == len(strikes):
        # Apply exclusions
        if excluded:
            keep = np.ones(len(strikes), dtype=bool)
            for idx in excluded:
                if 0 <= idx < len(keep):
                    keep[idx] = False
            eff_strikes = strikes[keep]
            eff_ivs = ivs[keep]
        else:
            eff_strikes = strikes.copy()
            eff_ivs = ivs.copy()

        # Append additions
        if added:
            for pt in added:
                if len(pt) >= 2:
                    eff_strikes = np.append(eff_strikes, pt[0])
                    eff_ivs = np.append(eff_ivs, pt[1])
            order = np.argsort(eff_strikes)
            eff_strikes = eff_strikes[order]
            eff_ivs = eff_ivs[order]

        # Refit SVI from the effective quote set
        if len(eff_strikes) >= 5:
            try:
                prior = fit_lqd_prior(
                    eff_strikes, eff_ivs, forward, T,
                    rate_used, grid, phi,
                )
                prior["_fit_strikes"] = strikes  # keep original full set for reference
                return prior
            except Exception:
                pass

    # Fallback: reconstruct from saved SVI params directly
    svi_saved = data.get("svi_params")
    svi_params = None
    if svi_saved and svi_saved.get("a") is not None:
        svi_params = {
            "a": float(svi_saved["a"]),
            "b": float(svi_saved["b"]),
            "rho": float(svi_saved["rho"]),
            "m": float(svi_saved["m"]),
            "sigma": float(svi_saved["sigma"]),
            "forward": float(svi_saved["forward"]),
            "T": float(svi_saved["T"]),
            "iv_fitted": np.array(data.get("ivs", []), dtype=float),
            "residuals": np.zeros(len(data.get("ivs", []))),
        }

    atm_iv = data.get("atm_iv", 0.25)
    if atm_iv < 0.01:
        atm_iv = 0.25

    base = bs_prior(atm_iv, T, grid)
    return {
        **base,
        "beta_fit": np.zeros(phi.shape[0]),
        "_bs_base": base,
        "_svi_params": svi_params,
        "_fit_strikes": strikes,
    }
