"""
Pydantic schemas for API request/response models.
"""
from pydantic import BaseModel
from typing import List, Optional, Dict


class Asset(BaseModel):
    ticker: str
    name: str
    sector: str
    is_index: bool
    index_weight: float
    liquidity_score: float


class QuoteSnapshot(BaseModel):
    node_key: Optional[str] = None  # compound key "TICKER:EXPIRY" (multi-expiry)
    ticker: str
    expiry: str
    T: float
    spot: float
    forward: float
    atm_iv: float
    strikes: List[float]
    mid_ivs: List[float]
    bid_ask_spread: List[float]
    open_interest: List[float]
    prev_close_ivs: Optional[List[float]] = None
    prev_close_atm_iv: Optional[float] = None
    prev_spot: Optional[float] = None
    forward_parity: Optional[float] = None
    forward_model: Optional[float] = None
    rate_used: Optional[float] = None
    div_yield_used: Optional[float] = None
    repo_rate_used: Optional[float] = None


class SmileData(BaseModel):
    node_key: Optional[str] = None  # compound key (multi-expiry)
    ticker: str
    strikes: List[float]
    iv_prior: List[float]
    iv_marked: List[float]
    beta: List[float]
    is_observed: bool


class SolveRequest(BaseModel):
    W_override: Optional[List[List[float]]] = None
    lambda_: float = 1.0
    eta: float = 0.01
    alpha_overrides: Optional[Dict[str, float]] = None
    shock_nudges: Optional[Dict[str, float]] = None
    observed_tickers: Optional[List[str]] = None
    excluded_quotes: Optional[Dict[str, List[int]]] = None
    added_quotes: Optional[Dict[str, List[List[float]]]] = None  # ticker -> [[strike, iv], ...]
    lambda_prior: float = 0.10         # prior-anchoring strength for SVI fit
    use_bid_ask_fit: bool = True       # use bid-ask dead-zone loss in SVI fit
    smile_model: str = "svi"           # smile model: "svi", "lqd", or "sigmoid"
    lambda_T: float = 2.0              # time kernel decay for cross-maturity influence


class SolveResponse(BaseModel):
    nodes: Dict[str, SmileData]
    W: List[List[float]]
    alphas: List[float]
    tickers: List[str]
    propagation_matrix: Optional[List[List[float]]] = None
    neumann_terms: Optional[List[List[List[float]]]] = None
    influence_scores: Optional[List[float]] = None
    wasserstein_distances: Optional[Dict[str, float]] = None


class GraphData(BaseModel):
    tickers: List[str]
    W: List[List[float]]
    alphas: List[float]
    assets: List[Asset]


class WMatrixUpdate(BaseModel):
    W: List[List[float]]
    alphas: Optional[List[float]] = None


class DistributionView(BaseModel):
    moneyness: List[float]
    iv_curve: List[Optional[float]]
    cdf_x: List[float]
    cdf_y: List[float]
    lqd_u: List[float]
    lqd_psi: List[Optional[float]]
    beta: Optional[List[float]] = None
    basis_labels: List[str] = ["ATM Var (v)", "Skew (\u03c8\u0302)", "Put Wing (p\u0302)", "Call Wing (c\u0302)", "Min-Var Ratio"]
    fit_forward: Optional[float] = None  # the forward the SVI was fitted with


class PriorOverrideRequest(BaseModel):
    beta: List[float]


class BetaOverrideRequest(BaseModel):
    beta_adjustment: List[float]


class PriorRefitRequest(BaseModel):
    excluded_indices: List[int] = []
    added_quotes: Optional[List[List[float]]] = None  # [[strike, iv], ...]


class FitRequest(BaseModel):
    """Lightweight fit: SVI for observed, prior for unobserved. No propagation."""
    observed_tickers: Optional[List[str]] = None
    excluded_quotes: Optional[Dict[str, List[int]]] = None
    added_quotes: Optional[Dict[str, List[List[float]]]] = None
    lambda_prior: float = 0.10         # prior-anchoring strength for SVI fit
    use_bid_ask_fit: bool = True       # use bid-ask dead-zone loss in SVI fit
    smile_model: str = "svi"           # smile model: "svi" or "lqd"


class SviOverrideRequest(BaseModel):
    v: float          # ATM implied variance
    psi_hat: float    # normalized ATM skew
    p_hat: float      # normalized put wing slope
    c_hat: float      # normalized call wing slope
    vt_ratio: float   # min-variance / ATM-variance ratio


class SavePriorRequest(BaseModel):
    excluded_indices: List[int] = []
    added_quotes: Optional[List[List[float]]] = None

class SavedPriorInfo(BaseModel):
    ticker: str
    filename: str
    timestamp: str


class NodeDistributionResponse(BaseModel):
    prior: DistributionView
    marked: Optional[DistributionView] = None
    ticker: str
    is_observed: bool
    wasserstein_dist: float = 0.0


class RatesConfig(BaseModel):
    repo_rate_gc: float = 0.0
    repo_overrides: Dict[str, float] = {}

class TreasuryCurveResponse(BaseModel):
    date: str
    tenors: List[float]
    rates: List[float]


class UniverseSelectRequest(BaseModel):
    tickers: List[str]


class AddTickerRequest(BaseModel):
    ticker: str
    name: str = ""
    sector: str = "Other"
    is_index: bool = False
    liquidity_score: float = 2.0


class CatalogResponse(BaseModel):
    assets: List[Asset]
    active_tickers: List[str]


class AvailableExpiriesResponse(BaseModel):
    ticker: str
    expiries: List[str]
    T_values: List[float]


class ExpirySelectionRequest(BaseModel):
    selections: Dict[str, List[str]]  # ticker -> list of expiry dates
