"""
Market data fetcher using yfinance.

Fetches option chains and computes the data needed by the engine:
strikes, mid IVs, bid-ask spreads, forward price, ATM vol.

Handles after-hours / weekend scenarios where bid/ask are zero by
falling back to lastPrice and yfinance's impliedVolatility.
"""
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from dataclasses import dataclass
from scipy.stats import norm as norm_dist
from scipy.optimize import brentq


@dataclass
class OptionChainData:
    ticker: str
    expiry: str               # ISO date string
    T: float                  # time to maturity in years
    spot: float
    forward: float
    strikes: np.ndarray       # shape (n,)
    mid_ivs: np.ndarray       # implied vols (decimal), shape (n,)
    bid_ask_spread: np.ndarray  # half spread in vol points, shape (n,)
    atm_iv: float             # ATM implied vol
    open_interest: np.ndarray  # shape (n,)
    call_mids: np.ndarray     # mid call prices, shape (n,)
    # Previous close data (for prior calibration)
    prev_close_ivs: Optional[np.ndarray] = None   # IVs from prev close prices
    prev_close_atm_iv: Optional[float] = None      # ATM IV from prev close
    prev_spot: Optional[float] = None               # previous day's spot price


def _bs_call_price(F, K, T, r, sigma):
    """Black-Scholes call price from forward."""
    if sigma < 1e-10 or T < 1e-10:
        return max(np.exp(-r * T) * (F - K), 0.0)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return np.exp(-r * T) * (F * norm_dist.cdf(d1) - K * norm_dist.cdf(d2))


def _price_to_iv(price, F, K, T, r):
    """Invert a single call price to implied vol."""
    intrinsic = max(np.exp(-r * T) * (F - K), 0.0)
    if price <= intrinsic + 1e-8:
        return None
    try:
        return brentq(lambda s: _bs_call_price(F, K, T, r, s) - price, 0.01, 5.0, xtol=1e-6)
    except (ValueError, RuntimeError):
        return None


def fetch_option_chain(
    ticker: str,
    target_maturity_days: int = 30,
    min_oi: int = 10,
    max_strikes: int = 30,
    r: float = 0.045,
) -> Optional[OptionChainData]:
    """
    Fetch option chain from Yahoo Finance for the expiry closest to target maturity.
    Robust to after-hours/weekend when bid/ask are zero.
    """
    try:
        tk = yf.Ticker(ticker)
        spot = tk.info.get("regularMarketPrice") or tk.info.get("previousClose")
        if spot is None:
            hist = tk.history(period="5d")
            if hist.empty:
                return None
            spot = float(hist["Close"].iloc[-1])

        expirations = tk.options
        if not expirations:
            return None

        # Filter out expiries less than 7 days away
        min_date = datetime.now() + timedelta(days=7)
        target_date = datetime.now() + timedelta(days=target_maturity_days)
        valid_exps = [e for e in expirations
                      if datetime.strptime(e, "%Y-%m-%d") >= min_date]
        if not valid_exps:
            valid_exps = list(expirations)

        best_exp = min(valid_exps, key=lambda e: abs(
            (datetime.strptime(e, "%Y-%m-%d") - target_date).days
        ))

        exp_date = datetime.strptime(best_exp, "%Y-%m-%d")
        T = max((exp_date - datetime.now()).days / 365.0, 1 / 365.0)
        forward = spot * np.exp(r * T)

        chain = tk.option_chain(best_exp)
        calls = chain.calls

        if calls.empty:
            return None

        # Determine market state: if most bids are zero, we're after-hours
        bid_col = calls["bid"].fillna(0).values.astype(float)
        market_open = (bid_col > 0).sum() > len(calls) * 0.3

        if market_open:
            # Live market: filter by bid > 0, OI, and IV
            mask = (
                (calls["openInterest"].fillna(0) >= min_oi)
                & (calls["impliedVolatility"] > 0.01)
                & (calls["bid"].fillna(0) > 0)
            )
            calls = calls[mask].copy()
            if len(calls) < 5:
                calls = chain.calls[
                    (chain.calls["impliedVolatility"] > 0.01)
                    & (chain.calls["bid"].fillna(0) > 0)
                ].copy()
        else:
            # After-hours: use lastPrice and OI, compute our own IV
            mask = (
                (calls["openInterest"].fillna(0) >= min_oi)
                & (calls["lastPrice"].fillna(0) > 0.01)
            )
            calls = calls[mask].copy()
            if len(calls) < 5:
                calls = chain.calls[
                    calls["lastPrice"].fillna(0) > 0.01
                ].copy()

        if len(calls) < 3:
            return None

        # Centre around ATM and limit strikes
        calls = calls.copy()
        calls["moneyness"] = np.abs(np.log(calls["strike"].values / forward))
        calls = calls.sort_values("moneyness")
        calls = calls.head(max_strikes).sort_values("strike")

        strikes = calls["strike"].values.astype(float)
        bid = calls["bid"].fillna(0).values.astype(float)
        ask = calls["ask"].fillna(0).values.astype(float)
        last = calls["lastPrice"].fillna(0).values.astype(float)
        change = calls["change"].fillna(0).values.astype(float)
        yf_iv = calls["impliedVolatility"].values.astype(float)
        oi = calls["openInterest"].fillna(0).values.astype(float)

        # Compute prices: use mid if available, else lastPrice
        if market_open:
            prices = np.where((bid > 0) & (ask > 0), (bid + ask) / 2.0, last)
        else:
            prices = last

        # Compute implied vols from prices (more reliable than yfinance's IV)
        mid_ivs = np.zeros(len(strikes))
        for i in range(len(strikes)):
            iv = _price_to_iv(prices[i], forward, strikes[i], T, r)
            if iv is not None and 0.01 < iv < 5.0:
                mid_ivs[i] = iv
            elif 0.01 < yf_iv[i] < 5.0:
                mid_ivs[i] = yf_iv[i]  # fallback to yfinance
            else:
                mid_ivs[i] = 0.0

        # Remove strikes with zero IV
        valid = mid_ivs > 0.01
        if valid.sum() < 3:
            return None
        strikes = strikes[valid]
        mid_ivs = mid_ivs[valid]
        prices = prices[valid]
        bid = bid[valid]
        ask = ask[valid]
        change = change[valid]
        oi = oi[valid]

        # Previous close prices and IVs: prev_price = lastPrice - change
        # Use previous day's spot for the forward
        hist = tk.history(period="5d")
        if len(hist) >= 2:
            prev_spot = float(hist["Close"].iloc[-2])
        else:
            prev_spot = spot
        prev_forward = prev_spot * np.exp(r * T)

        prev_prices = last[valid] if not np.all(valid) else last
        # Recompute: prev_prices from the filtered arrays
        prev_prices = prices - change  # prev close = current price - change
        prev_prices = np.maximum(prev_prices, 0.0)

        prev_close_ivs = np.zeros(len(strikes))
        for i in range(len(strikes)):
            if prev_prices[i] > 0.01:
                iv = _price_to_iv(prev_prices[i], prev_forward, strikes[i], T, r)
                if iv is not None and 0.01 < iv < 5.0:
                    prev_close_ivs[i] = iv
        # Fill zeros with current IV as fallback
        prev_close_ivs = np.where(prev_close_ivs > 0.01, prev_close_ivs, mid_ivs)

        prev_atm_idx = np.argmin(np.abs(strikes - prev_forward))
        prev_atm_iv = prev_close_ivs[prev_atm_idx]

        # Bid-ask spread in vol terms
        if market_open:
            price_spread = np.maximum(ask - bid, 0.01)
            d1 = (np.log(forward / strikes) + 0.5 * mid_ivs ** 2 * T) / (mid_ivs * np.sqrt(T) + 1e-10)
            vega = forward * np.exp(-r * T) * norm_dist.pdf(d1) * np.sqrt(T)
            vol_spread = price_spread / (2.0 * np.maximum(vega, 0.01))
            vol_spread = np.clip(vol_spread, 0.001, 0.10)
        else:
            # After-hours: assume ~2 vol point uncertainty
            vol_spread = np.full(len(strikes), 0.02)

        # ATM vol
        atm_idx = np.argmin(np.abs(strikes - forward))
        atm_iv = mid_ivs[atm_idx]

        return OptionChainData(
            ticker=ticker,
            expiry=best_exp,
            T=T,
            spot=float(spot),
            forward=float(forward),
            strikes=strikes,
            mid_ivs=mid_ivs,
            bid_ask_spread=vol_spread,
            atm_iv=float(atm_iv),
            open_interest=oi,
            call_mids=prices,
            prev_close_ivs=prev_close_ivs,
            prev_close_atm_iv=float(prev_atm_iv),
            prev_spot=float(prev_spot),
        )

    except Exception as e:
        print(f"Failed to fetch {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return None


def fetch_universe_quotes(
    tickers: List[str],
    target_maturity_days: int = 30,
    r: float = 0.045,
) -> Dict[str, OptionChainData]:
    """Fetch option chains for all tickers in the universe."""
    results = {}
    for t in tickers:
        data = fetch_option_chain(t, target_maturity_days, r=r)
        if data is not None:
            results[t] = data
    return results
