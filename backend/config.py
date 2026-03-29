"""
Configuration for the Graph-Regularised Vol Marking engine.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class AssetDef:
    ticker: str
    name: str
    sector: str
    is_index: bool = False
    index_weight: float = 0.0  # weight in SPX (or parent index)
    liquidity_score: float = 1.0  # relative, higher = more liquid


# ---------------------------------------------------------------------------
# Full catalog: ~100 liquid S&P constituents + major sector/thematic ETFs
# ---------------------------------------------------------------------------

CATALOG: List[AssetDef] = [
    # ── Indices & Broad ETFs ──────────────────────────────────────────────
    AssetDef("SPY",  "S&P 500 ETF",         "Index", is_index=True, liquidity_score=10.0),
    AssetDef("QQQ",  "Nasdaq 100 ETF",      "Index", is_index=True, liquidity_score=9.0),
    AssetDef("IWM",  "Russell 2000 ETF",    "Index", is_index=True, liquidity_score=7.0),
    AssetDef("DIA",  "Dow Jones ETF",        "Index", is_index=True, liquidity_score=6.0),

    # ── Sector ETFs ───────────────────────────────────────────────────────
    AssetDef("XLK",  "Tech Select ETF",      "Technology",    is_index=True, liquidity_score=6.0),
    AssetDef("XLF",  "Fin Select ETF",       "Financials",    is_index=True, liquidity_score=4.0),
    AssetDef("XLV",  "Health Select ETF",    "Healthcare",    is_index=True, liquidity_score=4.0),
    AssetDef("XLE",  "Energy Select ETF",    "Energy",        is_index=True, liquidity_score=4.0),
    AssetDef("XLI",  "Industrial Select ETF","Industrials",   is_index=True, liquidity_score=3.5),
    AssetDef("XLY",  "Cons Disc Select ETF", "Consumer Disc", is_index=True, liquidity_score=3.5),
    AssetDef("XLP",  "Cons Staples ETF",     "Consumer Staples", is_index=True, liquidity_score=3.0),
    AssetDef("XLU",  "Utilities Select ETF", "Utilities",     is_index=True, liquidity_score=2.5),
    AssetDef("XLRE", "Real Estate ETF",      "Real Estate",   is_index=True, liquidity_score=2.5),
    AssetDef("XLB",  "Materials Select ETF", "Materials",     is_index=True, liquidity_score=2.5),
    AssetDef("XLC",  "Comm Services ETF",    "Communication", is_index=True, liquidity_score=3.0),

    # ── Technology ────────────────────────────────────────────────────────
    AssetDef("AAPL",  "Apple",              "Technology", index_weight=0.07, liquidity_score=8.0),
    AssetDef("MSFT",  "Microsoft",          "Technology", index_weight=0.07, liquidity_score=7.0),
    AssetDef("NVDA",  "NVIDIA",             "Technology", index_weight=0.06, liquidity_score=8.0),
    AssetDef("GOOGL", "Alphabet",           "Technology", index_weight=0.04, liquidity_score=5.0),
    AssetDef("META",  "Meta Platforms",     "Technology", index_weight=0.02, liquidity_score=5.0),
    AssetDef("AVGO",  "Broadcom",           "Technology", index_weight=0.02, liquidity_score=4.0),
    AssetDef("ORCL",  "Oracle",             "Technology", index_weight=0.01, liquidity_score=3.5),
    AssetDef("CRM",   "Salesforce",         "Technology", index_weight=0.01, liquidity_score=3.5),
    AssetDef("AMD",   "AMD",                "Technology", index_weight=0.01, liquidity_score=5.0),
    AssetDef("INTC",  "Intel",              "Technology", index_weight=0.005, liquidity_score=4.0),
    AssetDef("CSCO",  "Cisco",              "Technology", index_weight=0.01, liquidity_score=3.0),
    AssetDef("ADBE",  "Adobe",              "Technology", index_weight=0.01, liquidity_score=3.5),
    AssetDef("NOW",   "ServiceNow",         "Technology", index_weight=0.01, liquidity_score=3.0),
    AssetDef("IBM",   "IBM",                "Technology", index_weight=0.005, liquidity_score=2.5),
    AssetDef("QCOM",  "Qualcomm",           "Technology", index_weight=0.005, liquidity_score=3.0),
    AssetDef("TXN",   "Texas Instruments",  "Technology", index_weight=0.005, liquidity_score=2.5),
    AssetDef("AMAT",  "Applied Materials",  "Technology", index_weight=0.005, liquidity_score=2.5),
    AssetDef("MU",    "Micron",             "Technology", index_weight=0.005, liquidity_score=3.0),
    AssetDef("PANW",  "Palo Alto Networks", "Technology", index_weight=0.005, liquidity_score=3.0),
    AssetDef("SHOP",  "Shopify",            "Technology", index_weight=0.003, liquidity_score=3.0),

    # ── Consumer Discretionary ────────────────────────────────────────────
    AssetDef("AMZN",  "Amazon",             "Consumer Disc", index_weight=0.03, liquidity_score=5.0),
    AssetDef("TSLA",  "Tesla",              "Consumer Disc", index_weight=0.02, liquidity_score=6.0),
    AssetDef("HD",    "Home Depot",         "Consumer Disc", index_weight=0.01, liquidity_score=3.0),
    AssetDef("NKE",   "Nike",               "Consumer Disc", index_weight=0.005, liquidity_score=3.0),
    AssetDef("MCD",   "McDonald's",         "Consumer Disc", index_weight=0.005, liquidity_score=2.5),
    AssetDef("SBUX",  "Starbucks",          "Consumer Disc", index_weight=0.005, liquidity_score=2.5),
    AssetDef("LOW",   "Lowe's",             "Consumer Disc", index_weight=0.005, liquidity_score=2.5),
    AssetDef("TJX",   "TJX Companies",      "Consumer Disc", index_weight=0.005, liquidity_score=2.0),
    AssetDef("BKNG",  "Booking Holdings",   "Consumer Disc", index_weight=0.005, liquidity_score=2.5),
    AssetDef("CMG",   "Chipotle",           "Consumer Disc", index_weight=0.003, liquidity_score=2.0),

    # ── Consumer Staples ──────────────────────────────────────────────────
    AssetDef("PG",    "Procter & Gamble",   "Consumer Staples", index_weight=0.01, liquidity_score=3.0),
    AssetDef("KO",    "Coca-Cola",          "Consumer Staples", index_weight=0.01, liquidity_score=3.0),
    AssetDef("PEP",   "PepsiCo",            "Consumer Staples", index_weight=0.01, liquidity_score=2.5),
    AssetDef("COST",  "Costco",             "Consumer Staples", index_weight=0.01, liquidity_score=3.0),
    AssetDef("WMT",   "Walmart",            "Consumer Staples", index_weight=0.01, liquidity_score=3.0),
    AssetDef("PM",    "Philip Morris",      "Consumer Staples", index_weight=0.005, liquidity_score=2.0),
    AssetDef("CL",    "Colgate-Palmolive",  "Consumer Staples", index_weight=0.005, liquidity_score=2.0),

    # ── Financials ────────────────────────────────────────────────────────
    AssetDef("JPM",   "JPMorgan",           "Financials", index_weight=0.01, liquidity_score=4.0),
    AssetDef("GS",    "Goldman Sachs",      "Financials", index_weight=0.01, liquidity_score=3.0),
    AssetDef("BAC",   "Bank of America",    "Financials", index_weight=0.01, liquidity_score=4.0),
    AssetDef("MS",    "Morgan Stanley",     "Financials", index_weight=0.005, liquidity_score=3.0),
    AssetDef("C",     "Citigroup",          "Financials", index_weight=0.005, liquidity_score=3.0),
    AssetDef("WFC",   "Wells Fargo",        "Financials", index_weight=0.005, liquidity_score=3.0),
    AssetDef("BLK",   "BlackRock",          "Financials", index_weight=0.005, liquidity_score=2.5),
    AssetDef("SCHW",  "Charles Schwab",     "Financials", index_weight=0.005, liquidity_score=2.5),
    AssetDef("AXP",   "American Express",   "Financials", index_weight=0.005, liquidity_score=2.5),
    AssetDef("BRK-B", "Berkshire Hathaway", "Financials", index_weight=0.02, liquidity_score=3.5),

    # ── Healthcare ────────────────────────────────────────────────────────
    AssetDef("UNH",   "UnitedHealth",       "Healthcare", index_weight=0.01, liquidity_score=4.0),
    AssetDef("JNJ",   "Johnson & Johnson",  "Healthcare", index_weight=0.01, liquidity_score=3.5),
    AssetDef("LLY",   "Eli Lilly",          "Healthcare", index_weight=0.02, liquidity_score=4.0),
    AssetDef("PFE",   "Pfizer",             "Healthcare", index_weight=0.005, liquidity_score=4.0),
    AssetDef("ABBV",  "AbbVie",             "Healthcare", index_weight=0.01, liquidity_score=3.0),
    AssetDef("MRK",   "Merck",              "Healthcare", index_weight=0.01, liquidity_score=3.0),
    AssetDef("TMO",   "Thermo Fisher",      "Healthcare", index_weight=0.005, liquidity_score=2.5),
    AssetDef("ABT",   "Abbott Labs",        "Healthcare", index_weight=0.005, liquidity_score=2.5),
    AssetDef("BMY",   "Bristol-Myers",      "Healthcare", index_weight=0.005, liquidity_score=2.5),
    AssetDef("AMGN",  "Amgen",              "Healthcare", index_weight=0.005, liquidity_score=2.5),

    # ── Energy ────────────────────────────────────────────────────────────
    AssetDef("XOM",   "Exxon Mobil",        "Energy", index_weight=0.01, liquidity_score=4.0),
    AssetDef("CVX",   "Chevron",            "Energy", index_weight=0.01, liquidity_score=3.5),
    AssetDef("COP",   "ConocoPhillips",     "Energy", index_weight=0.005, liquidity_score=2.5),
    AssetDef("SLB",   "Schlumberger",       "Energy", index_weight=0.005, liquidity_score=2.5),
    AssetDef("EOG",   "EOG Resources",      "Energy", index_weight=0.005, liquidity_score=2.0),
    AssetDef("OXY",   "Occidental",         "Energy", index_weight=0.003, liquidity_score=2.5),

    # ── Industrials ───────────────────────────────────────────────────────
    AssetDef("CAT",   "Caterpillar",        "Industrials", index_weight=0.005, liquidity_score=3.0),
    AssetDef("BA",    "Boeing",             "Industrials", index_weight=0.005, liquidity_score=4.0),
    AssetDef("HON",   "Honeywell",          "Industrials", index_weight=0.005, liquidity_score=2.5),
    AssetDef("UPS",   "UPS",                "Industrials", index_weight=0.005, liquidity_score=2.5),
    AssetDef("RTX",   "RTX Corp",           "Industrials", index_weight=0.005, liquidity_score=2.5),
    AssetDef("DE",    "Deere & Co",         "Industrials", index_weight=0.005, liquidity_score=2.5),
    AssetDef("GE",    "GE Aerospace",       "Industrials", index_weight=0.005, liquidity_score=3.0),
    AssetDef("LMT",   "Lockheed Martin",    "Industrials", index_weight=0.005, liquidity_score=2.5),

    # ── Communication Services ────────────────────────────────────────────
    AssetDef("GOOG",  "Alphabet C",         "Communication", index_weight=0.02, liquidity_score=4.0),
    AssetDef("NFLX",  "Netflix",            "Communication", index_weight=0.01, liquidity_score=4.0),
    AssetDef("DIS",   "Walt Disney",        "Communication", index_weight=0.005, liquidity_score=3.5),
    AssetDef("CMCSA", "Comcast",            "Communication", index_weight=0.005, liquidity_score=2.5),
    AssetDef("T",     "AT&T",               "Communication", index_weight=0.005, liquidity_score=3.0),
    AssetDef("VZ",    "Verizon",            "Communication", index_weight=0.005, liquidity_score=2.5),

    # ── Utilities ─────────────────────────────────────────────────────────
    AssetDef("NEE",   "NextEra Energy",     "Utilities", index_weight=0.005, liquidity_score=2.5),
    AssetDef("SO",    "Southern Company",   "Utilities", index_weight=0.005, liquidity_score=2.0),
    AssetDef("DUK",   "Duke Energy",        "Utilities", index_weight=0.005, liquidity_score=2.0),

    # ── Real Estate ───────────────────────────────────────────────────────
    AssetDef("AMT",   "American Tower",     "Real Estate", index_weight=0.005, liquidity_score=2.0),
    AssetDef("PLD",   "Prologis",           "Real Estate", index_weight=0.005, liquidity_score=2.0),
    AssetDef("CCI",   "Crown Castle",       "Real Estate", index_weight=0.003, liquidity_score=2.0),

    # ── Materials ─────────────────────────────────────────────────────────
    AssetDef("LIN",   "Linde",              "Materials", index_weight=0.005, liquidity_score=2.0),
    AssetDef("APD",   "Air Products",       "Materials", index_weight=0.005, liquidity_score=2.0),
    AssetDef("FCX",   "Freeport-McMoRan",   "Materials", index_weight=0.003, liquidity_score=2.5),
]

# Build lookup for quick access
CATALOG_MAP: Dict[str, AssetDef] = {a.ticker: a for a in CATALOG}

# Default active universe (backward-compatible with original 10)
DEFAULT_ACTIVE_TICKERS: List[str] = [
    "SPY", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "JPM", "GS", "XLK", "XLF",
]

# Legacy alias
DEFAULT_UNIVERSE: List[AssetDef] = [CATALOG_MAP[t] for t in DEFAULT_ACTIVE_TICKERS]


# ---------------------------------------------------------------------------
# Sector-based correlation defaults
# ---------------------------------------------------------------------------
# Replaces hardcoded pairwise correlations. Scales to any universe size.
# Values are typical IV-change correlations (not spot).

SECTORS = [
    "Index", "Technology", "Consumer Disc", "Consumer Staples",
    "Financials", "Healthcare", "Energy", "Industrials",
    "Communication", "Utilities", "Real Estate", "Materials",
]

# Symmetric matrix: SECTOR_CORRELATION[i][j] = default ρ between sectors i and j
_SECTOR_CORR_VALUES = {
    # Intra-sector (diagonal) — used when both assets share a sector
    ("Index",           "Index"):           0.90,
    ("Technology",      "Technology"):      0.75,
    ("Consumer Disc",   "Consumer Disc"):   0.65,
    ("Consumer Staples","Consumer Staples"):0.60,
    ("Financials",      "Financials"):      0.75,
    ("Healthcare",      "Healthcare"):      0.60,
    ("Energy",          "Energy"):          0.80,
    ("Industrials",     "Industrials"):     0.65,
    ("Communication",   "Communication"):   0.65,
    ("Utilities",       "Utilities"):       0.70,
    ("Real Estate",     "Real Estate"):     0.65,
    ("Materials",       "Materials"):       0.65,

    # Cross-sector — representative pairs (symmetric lookup below)
    ("Index",      "Technology"):      0.80,
    ("Index",      "Consumer Disc"):   0.75,
    ("Index",      "Consumer Staples"):0.55,
    ("Index",      "Financials"):      0.70,
    ("Index",      "Healthcare"):      0.55,
    ("Index",      "Energy"):          0.50,
    ("Index",      "Industrials"):     0.70,
    ("Index",      "Communication"):   0.70,
    ("Index",      "Utilities"):       0.35,
    ("Index",      "Real Estate"):     0.40,
    ("Index",      "Materials"):       0.55,

    ("Technology",      "Consumer Disc"):   0.60,
    ("Technology",      "Consumer Staples"):0.30,
    ("Technology",      "Financials"):      0.50,
    ("Technology",      "Healthcare"):      0.35,
    ("Technology",      "Energy"):          0.25,
    ("Technology",      "Industrials"):     0.45,
    ("Technology",      "Communication"):   0.70,
    ("Technology",      "Utilities"):       0.20,
    ("Technology",      "Real Estate"):     0.25,
    ("Technology",      "Materials"):       0.30,

    ("Consumer Disc",   "Consumer Staples"):0.45,
    ("Consumer Disc",   "Financials"):      0.50,
    ("Consumer Disc",   "Healthcare"):      0.35,
    ("Consumer Disc",   "Energy"):          0.30,
    ("Consumer Disc",   "Industrials"):     0.55,
    ("Consumer Disc",   "Communication"):   0.55,
    ("Consumer Disc",   "Utilities"):       0.25,
    ("Consumer Disc",   "Real Estate"):     0.35,
    ("Consumer Disc",   "Materials"):       0.40,

    ("Consumer Staples","Financials"):      0.40,
    ("Consumer Staples","Healthcare"):      0.45,
    ("Consumer Staples","Energy"):          0.25,
    ("Consumer Staples","Industrials"):     0.40,
    ("Consumer Staples","Communication"):   0.30,
    ("Consumer Staples","Utilities"):       0.50,
    ("Consumer Staples","Real Estate"):     0.40,
    ("Consumer Staples","Materials"):       0.35,

    ("Financials",      "Healthcare"):      0.35,
    ("Financials",      "Energy"):          0.40,
    ("Financials",      "Industrials"):     0.55,
    ("Financials",      "Communication"):   0.45,
    ("Financials",      "Utilities"):       0.35,
    ("Financials",      "Real Estate"):     0.50,
    ("Financials",      "Materials"):       0.40,

    ("Healthcare",      "Energy"):          0.20,
    ("Healthcare",      "Industrials"):     0.35,
    ("Healthcare",      "Communication"):   0.30,
    ("Healthcare",      "Utilities"):       0.40,
    ("Healthcare",      "Real Estate"):     0.30,
    ("Healthcare",      "Materials"):       0.25,

    ("Energy",          "Industrials"):     0.45,
    ("Energy",          "Communication"):   0.25,
    ("Energy",          "Utilities"):       0.30,
    ("Energy",          "Real Estate"):     0.20,
    ("Energy",          "Materials"):       0.50,

    ("Industrials",     "Communication"):   0.40,
    ("Industrials",     "Utilities"):       0.35,
    ("Industrials",     "Real Estate"):     0.35,
    ("Industrials",     "Materials"):       0.55,

    ("Communication",   "Utilities"):       0.25,
    ("Communication",   "Real Estate"):     0.30,
    ("Communication",   "Materials"):       0.30,

    ("Utilities",       "Real Estate"):     0.45,
    ("Utilities",       "Materials"):       0.35,

    ("Real Estate",     "Materials"):       0.30,
}


def get_sector_correlation(s1: str, s2: str) -> float:
    """Look up default correlation between two sectors."""
    if s1 == s2:
        return _SECTOR_CORR_VALUES.get((s1, s2), 0.60)
    key = (s1, s2) if (s1, s2) in _SECTOR_CORR_VALUES else (s2, s1)
    return _SECTOR_CORR_VALUES.get(key, 0.30)


def get_correlation(t1: str, t2: str) -> float:
    """
    Get correlation between two tickers using sector-based defaults.
    Index ETFs get a boost when paired with constituents of their sector.
    """
    if t1 == t2:
        return 1.0
    a1 = CATALOG_MAP.get(t1)
    a2 = CATALOG_MAP.get(t2)
    if a1 is None or a2 is None:
        return 0.30  # unknown ticker fallback
    return get_sector_correlation(a1.sector, a2.sector)


@dataclass
class EngineConfig:
    M: int = 5                          # number of LQD basis functions
    quantile_grid_size: int = 200       # discretisation of u in (0,1)
    lambda_: float = 1.0               # graph coupling strength
    eta: float = 0.01                   # Tikhonov / smoothness damping
    alpha_min: float = 0.10             # self-trust floor (least liquid nodes)
    alpha_max: float = 0.90             # self-trust cap   (most liquid nodes)
    target_maturity_days: int = 30      # target expiry for Phase 1
    risk_free_rate: Optional[float] = None  # None = use Treasury curve, float = flat override
    repo_rate_gc: float = 0.0           # GC (General Collateral) repo rate applied to all names by default
    repo_overrides: Dict[str, float] = field(default_factory=dict)  # per-ticker repo rate overrides for hard-to-borrow names
    tail_reg_weight: float = 1.0        # omega_tail for tail basis coefficients
    interior_reg_weight: float = 0.01   # omega_m for interior Legendre terms
    epsilon_tail: float = 0.01          # lower bound for tail coefficients β₁, β₂

    def repo_rate_for(self, ticker: str) -> float:
        return self.repo_overrides.get(ticker, self.repo_rate_gc)
