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
    basis_labels: List[str] = ["Level (a)", "Wings (b)", "Skew (rho)", "Shift (m)", "Curvature (sig)"]
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


class SviOverrideRequest(BaseModel):
    a: float
    b: float
    rho: float
    m: float
    sigma: float


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
