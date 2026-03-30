"""
Node key utilities for multi-expiry support.

A node in the influence graph is identified by a compound key "TICKER:EXPIRY"
(e.g. "SPY:2025-04-17"). In single-maturity mode, a plain ticker string is
also accepted and treated as having no expiry suffix.
"""
from datetime import datetime, date
from typing import Dict, List, Tuple
from collections import defaultdict

SEPARATOR = ":"


def make_node_key(ticker: str, expiry: str) -> str:
    """Create a compound node key from ticker and expiry date string."""
    return f"{ticker}{SEPARATOR}{expiry}"


def split_node_key(key: str) -> Tuple[str, str]:
    """Split a node key into (ticker, expiry).

    If the key has no separator (plain ticker), returns (ticker, "").
    """
    idx = key.rfind(SEPARATOR)
    if idx == -1:
        return key, ""
    return key[:idx], key[idx + 1:]


def is_compound_key(key: str) -> bool:
    """Check whether a key contains an expiry suffix."""
    return SEPARATOR in key


def ticker_of(key: str) -> str:
    """Extract the ticker from a node key (compound or plain)."""
    return split_node_key(key)[0]


def expiry_of(key: str) -> str:
    """Extract the expiry from a node key. Returns '' for plain tickers."""
    return split_node_key(key)[1]


def expiry_label(expiry: str, T: float = None) -> str:
    """Human-friendly expiry label: 'Apr 17' or 'Apr 17 (30d)'.

    Args:
        expiry: ISO date string (e.g. "2025-04-17")
        T: time to maturity in years (optional, used for days label)
    """
    try:
        dt = datetime.strptime(expiry, "%Y-%m-%d")
        label = dt.strftime("%b %d")
    except (ValueError, TypeError):
        label = expiry
    if T is not None:
        days = int(round(T * 365))
        label += f" ({days}d)"
    return label


def group_by_ticker(node_keys: List[str]) -> Dict[str, List[str]]:
    """Group node keys by ticker, preserving order.

    Returns: {"SPY": ["SPY:2025-04-17", "SPY:2025-05-16"], "AAPL": [...]}
    """
    groups: Dict[str, List[str]] = defaultdict(list)
    for key in node_keys:
        tk = ticker_of(key)
        groups[tk].append(key)
    return dict(groups)


def unique_tickers(node_keys: List[str]) -> List[str]:
    """Extract unique tickers from node keys, preserving first-seen order."""
    seen = set()
    result = []
    for key in node_keys:
        tk = ticker_of(key)
        if tk not in seen:
            seen.add(tk)
            result.append(tk)
    return result
