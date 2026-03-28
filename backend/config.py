"""
Configuration for the Graph-Regularised Vol Marking engine.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class AssetDef:
    ticker: str
    name: str
    sector: str
    is_index: bool = False
    index_weight: float = 0.0  # weight in SPX (or parent index)
    liquidity_score: float = 1.0  # relative, higher = more liquid


DEFAULT_UNIVERSE: List[AssetDef] = [
    AssetDef("SPY",  "S&P 500 ETF",    "Index",      is_index=True, liquidity_score=10.0),
    AssetDef("AAPL", "Apple",           "Technology", index_weight=0.07, liquidity_score=8.0),
    AssetDef("MSFT", "Microsoft",       "Technology", index_weight=0.07, liquidity_score=7.0),
    AssetDef("GOOGL","Alphabet",        "Technology", index_weight=0.04, liquidity_score=5.0),
    AssetDef("AMZN", "Amazon",          "Consumer",   index_weight=0.03, liquidity_score=5.0),
    AssetDef("TSLA", "Tesla",           "Consumer",   index_weight=0.02, liquidity_score=6.0),
    AssetDef("JPM",  "JPMorgan",        "Financials", index_weight=0.01, liquidity_score=4.0),
    AssetDef("GS",   "Goldman Sachs",   "Financials", index_weight=0.01, liquidity_score=3.0),
    AssetDef("XLK",  "Tech Select ETF", "Technology", is_index=True, liquidity_score=6.0),
    AssetDef("XLF",  "Fin Select ETF",  "Financials", is_index=True, liquidity_score=4.0),
]


@dataclass
class EngineConfig:
    M: int = 5                          # number of LQD basis functions
    quantile_grid_size: int = 200       # discretisation of u in (0,1)
    lambda_: float = 1.0               # graph coupling strength
    eta: float = 0.01                   # Tikhonov / smoothness damping
    alpha_liquid: float = 0.1           # self-trust for liquid nodes
    alpha_illiquid: float = 0.3         # self-trust for illiquid nodes
    target_maturity_days: int = 30      # target expiry for Phase 1
    risk_free_rate: Optional[float] = None  # None = use Treasury curve, float = flat override
    repo_rate_gc: float = 0.0           # GC (General Collateral) repo rate applied to all names by default
    repo_overrides: Dict[str, float] = field(default_factory=dict)  # per-ticker repo rate overrides for hard-to-borrow names
    tail_reg_weight: float = 1.0        # omega_tail for tail basis coefficients
    interior_reg_weight: float = 0.01   # omega_m for interior Legendre terms
    epsilon_tail: float = 0.01          # lower bound for tail coefficients β₁, β₂

    def repo_rate_for(self, ticker: str) -> float:
        return self.repo_overrides.get(ticker, self.repo_rate_gc)


DEFAULT_CORRELATIONS = {
    ("SPY", "AAPL"): 0.75, ("SPY", "MSFT"): 0.78, ("SPY", "GOOGL"): 0.72,
    ("SPY", "AMZN"): 0.70, ("SPY", "TSLA"): 0.55, ("SPY", "JPM"): 0.65,
    ("SPY", "GS"): 0.62,   ("SPY", "XLK"): 0.92,  ("SPY", "XLF"): 0.75,
    ("AAPL", "MSFT"): 0.72, ("AAPL", "GOOGL"): 0.68, ("AAPL", "AMZN"): 0.60,
    ("AAPL", "TSLA"): 0.45, ("AAPL", "XLK"): 0.85,
    ("MSFT", "GOOGL"): 0.74, ("MSFT", "AMZN"): 0.62, ("MSFT", "XLK"): 0.88,
    ("GOOGL", "AMZN"): 0.65, ("GOOGL", "XLK"): 0.80,
    ("AMZN", "XLK"): 0.65,
    ("TSLA", "XLK"): 0.50,
    ("JPM", "GS"): 0.82, ("JPM", "XLF"): 0.90, ("GS", "XLF"): 0.85,
}


def get_correlation(t1: str, t2: str) -> float:
    if t1 == t2:
        return 1.0
    key = (t1, t2) if (t1, t2) in DEFAULT_CORRELATIONS else (t2, t1)
    return DEFAULT_CORRELATIONS.get(key, 0.3)
