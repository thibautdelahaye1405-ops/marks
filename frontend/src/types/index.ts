export interface Asset {
  ticker: string;
  name: string;
  sector: string;
  is_index: boolean;
  index_weight: number;
  liquidity_score: number;
}

export interface QuoteSnapshot {
  node_key?: string;  // compound key "TICKER:EXPIRY" (multi-expiry)
  ticker: string;
  expiry: string;
  T: number;
  spot: number;
  forward: number;
  atm_iv: number;
  strikes: number[];
  mid_ivs: number[];
  bid_ask_spread: number[];
  open_interest: number[];
  prev_close_ivs: number[] | null;
  prev_close_atm_iv: number | null;
  prev_spot: number | null;
  forward_parity: number | null;
  forward_model: number | null;
  rate_used: number | null;
  div_yield_used: number | null;
  repo_rate_used: number | null;
}

export interface SmileData {
  node_key?: string;  // compound key (multi-expiry)
  ticker: string;
  strikes: number[];
  iv_prior: (number | null)[];
  iv_marked: (number | null)[];
  beta: number[];
  is_observed: boolean;
}

export interface SolveRequest {
  W_override?: number[][] | null;
  lambda_: number;
  eta: number;
  alpha_overrides?: Record<string, number> | null;
  shock_nudges?: Record<string, number> | null;
  observed_tickers?: string[] | null;
  excluded_quotes?: Record<string, number[]> | null;
  added_quotes?: Record<string, number[][]> | null; // ticker -> [[strike, iv], ...]
  lambda_prior: number;
  use_bid_ask_fit: boolean;
  smile_model: string;
  lambda_T?: number;  // time kernel decay for cross-maturity influence
}

export interface SolveResponse {
  nodes: Record<string, SmileData>;
  W: number[][];
  alphas: number[];
  tickers: string[];
  propagation_matrix: number[][] | null;
  neumann_terms: number[][][] | null;
  influence_scores: number[] | null;
  wasserstein_distances: Record<string, number> | null;
}

export interface GraphData {
  tickers: string[];
  W: number[][];
  alphas: number[];
  assets: Asset[];
}

export interface DistributionView {
  moneyness: number[];
  iv_curve: (number | null)[];
  cdf_x: number[];
  cdf_y: number[];
  lqd_u: number[];
  lqd_psi: (number | null)[];
  beta: number[] | null;
  basis_labels: string[];
  fit_forward: number | null;
}

export interface NodeDistributionResponse {
  prior: DistributionView;
  marked: DistributionView | null;
  ticker: string;
  is_observed: boolean;
  wasserstein_dist: number;
}

export interface RatesConfig {
  repo_rate_gc: number;
  repo_overrides: Record<string, number>;
}

export interface TreasuryCurveData {
  date: string;
  tenors: number[];
  rates: number[];
}

export interface CatalogResponse {
  assets: Asset[];
  active_tickers: string[];
}

export interface UniverseSelectResponse {
  status: string;
  tickers: string[];
  graph: GraphData;
}

export interface AddTickerResponse {
  status: string;
  asset: Asset;
  tickers: string[];
}

export interface AvailableExpiries {
  ticker: string;
  expiries: string[];
  T_values: number[];
}

export interface ExpirySelection {
  selections: Record<string, string[]>;
}
