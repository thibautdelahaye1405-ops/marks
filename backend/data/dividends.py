"""
Dividend data fetcher using yfinance.

Fetches dividend yield and discrete dividend schedules for stocks and indices.
Provides present-value computation for discrete dividend streams.
"""
import logging
import math
import time
import yfinance as yf
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Module-level cache: ticker -> (timestamp, DividendInfo)
_cache: Dict[str, Tuple[float, "DividendInfo"]] = {}
_CACHE_TTL = 3600.0  # 1 hour


@dataclass
class DividendInfo:
    ticker: str
    continuous_yield: float  # annualized continuous dividend yield
    discrete_dividends: List[Tuple[datetime, float]]  # (ex_date, amount)
    is_index: bool


def fetch_dividend_info(ticker: str, is_index: bool = False) -> DividendInfo:
    """
    Fetch dividend information from Yahoo Finance.

    For indices: uses info-level yield fields, no discrete dividends.
    For stocks: fetches dividend history and extracts upcoming ex-dates.
    Results are cached for 1 hour per ticker.
    """
    now = time.time()
    if ticker in _cache:
        ts, cached = _cache[ticker]
        if now - ts < _CACHE_TTL:
            return cached

    try:
        result = _fetch_impl(ticker, is_index)
    except Exception as e:
        logger.warning("Failed to fetch dividends for %s: %s", ticker, e)
        result = DividendInfo(
            ticker=ticker,
            continuous_yield=0.0,
            discrete_dividends=[],
            is_index=is_index,
        )

    _cache[ticker] = (now, result)
    return result


def _safe_yield(info: dict) -> float:
    """Extract dividend yield as a decimal, handling yfinance inconsistencies.

    yfinance's 'dividendYield' field is unreliable (often in percentage or
    nonsensical).  Prefer 'trailingAnnualDividendYield' which is a proper
    decimal.  Clamp to [0, 0.15] as a sanity check.
    """
    # trailingAnnualDividendYield is the most reliable (already a decimal)
    tady = info.get("trailingAnnualDividendYield")
    if tady is not None and tady > 0:
        return min(float(tady), 0.15)
    # Fallback: dividendYield — if > 0.20 assume it's in percent
    dy = info.get("dividendYield")
    if dy is not None and dy > 0:
        val = float(dy)
        if val > 0.20:
            val = val / 100.0
        return min(val, 0.15)
    return 0.0


def _fetch_impl(ticker: str, is_index: bool) -> DividendInfo:
    """Internal fetch logic."""
    tk = yf.Ticker(ticker)
    info = tk.info or {}

    if is_index:
        yield_val = _safe_yield(info)
        return DividendInfo(
            ticker=ticker,
            continuous_yield=yield_val,
            discrete_dividends=[],
            is_index=True,
        )

    # Stock: get continuous yield as fallback
    continuous_yield = _safe_yield(info)

    # Fetch dividend history (last 2 years to estimate forward schedule)
    discrete_dividends: List[Tuple[datetime, float]] = []
    try:
        divs = tk.dividends
        if divs is not None and not divs.empty:
            # Use historical pattern to project upcoming ex-dates
            now_dt = datetime.now()
            cutoff = now_dt + timedelta(days=365)

            # Get recent dividends to estimate frequency and amount
            recent = divs.tail(8)
            if len(recent) >= 2:
                # Estimate interval between dividends
                dates = [d.to_pydatetime().replace(tzinfo=None) if hasattr(d, "to_pydatetime") else d for d in recent.index]
                intervals = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
                avg_interval = sum(intervals) / len(intervals)
                last_amount = float(recent.iloc[-1])
                last_date = dates[-1]

                # Project forward from the last known ex-date
                next_date = last_date + timedelta(days=avg_interval)
                while next_date <= cutoff:
                    if next_date > now_dt:
                        discrete_dividends.append((next_date, last_amount))
                    next_date = next_date + timedelta(days=avg_interval)
            elif len(recent) == 1:
                # Single dividend: assume annual
                last_date = recent.index[0]
                if hasattr(last_date, "to_pydatetime"):
                    last_date = last_date.to_pydatetime().replace(tzinfo=None)
                last_amount = float(recent.iloc[0])
                next_date = last_date + timedelta(days=365)
                if now_dt < next_date <= now_dt + timedelta(days=365):
                    discrete_dividends.append((next_date, last_amount))
    except Exception as e:
        logger.warning("Failed to fetch dividend history for %s: %s", ticker, e)

    return DividendInfo(
        ticker=ticker,
        continuous_yield=continuous_yield,
        discrete_dividends=discrete_dividends,
        is_index=False,
    )


def pv_discrete_dividends(
    dividends: List[Tuple[datetime, float]],
    rate_func: Callable[[float], float],
    valuation_date: datetime,
) -> float:
    """
    Present value of discrete dividend stream.

    Computes sum of d_i * exp(-r(T_i) * T_i) for each dividend
    with ex_date > valuation_date.

    Args:
        dividends: list of (ex_date, amount) pairs.
        rate_func: maps time-to-maturity T (years) to continuously compounded rate.
        valuation_date: reference date for discounting.

    Returns:
        Total present value of future dividends.
    """
    pv = 0.0
    for ex_date, amount in dividends:
        if ex_date <= valuation_date:
            continue
        T = (ex_date - valuation_date).days / 365.0
        if T <= 0:
            continue
        r = rate_func(T)
        pv += amount * math.exp(-r * T)
    return pv
