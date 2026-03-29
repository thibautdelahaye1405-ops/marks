const BASE_URL = "/api";

import type {
  Asset,
  QuoteSnapshot,
  SolveRequest,
  SolveResponse,
  GraphData,
  DistributionView,
  NodeDistributionResponse,
  SmileData,
  CatalogResponse,
  UniverseSelectResponse,
  AddTickerResponse,
} from "../types";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json();
}

export const api = {
  getUniverse: () => request<Asset[]>("/universe"),

  getCatalog: () => request<CatalogResponse>("/catalog"),

  selectUniverse: (tickers: string[]) =>
    request<UniverseSelectResponse>("/universe/select", {
      method: "POST",
      body: JSON.stringify({ tickers }),
    }),

  addTicker: (ticker: string, name?: string, sector?: string) =>
    request<AddTickerResponse>("/universe/add", {
      method: "POST",
      body: JSON.stringify({ ticker, name: name || "", sector: sector || "Other" }),
    }),

  saveUniverseSelection: () =>
    request<{ status: string; tickers: string[] }>("/universe/save", {
      method: "POST",
    }),

  fetchQuotes: () =>
    request<Record<string, QuoteSnapshot>>("/fetch-quotes", { method: "POST" }),

  getLatestQuotes: () =>
    request<Record<string, QuoteSnapshot>>("/quotes/latest"),

  fit: (req: { observed_tickers?: string[] | null; excluded_quotes?: Record<string, number[]> | null; added_quotes?: Record<string, number[][]> | null }) =>
    request<{ nodes: Record<string, any>; tickers: string[] }>("/fit", {
      method: "POST",
      body: JSON.stringify(req),
    }),

  solve: (req: SolveRequest) =>
    request<SolveResponse>("/solve", {
      method: "POST",
      body: JSON.stringify(req),
    }),

  getGraph: () => request<GraphData>("/graph"),

  updateGraph: (W: number[][], alphas?: number[]) =>
    request<{ status: string }>("/graph", {
      method: "PUT",
      body: JSON.stringify({ W, alphas }),
    }),

  calibratePriors: () =>
    request<{ status: string; calibrated: string[] }>("/calibrate-priors", {
      method: "POST",
    }),

  getPrior: (ticker: string) =>
    request<DistributionView>(`/prior/${ticker}`),

  overridePrior: (ticker: string, beta: number[]) =>
    request<DistributionView>(`/prior/${ticker}/override`, {
      method: "POST",
      body: JSON.stringify({ beta }),
    }),

  refitPrior: (ticker: string, excludedIndices: number[], addedQuotes?: [number, number][]) =>
    request<DistributionView>(`/prior/${ticker}/refit`, {
      method: "POST",
      body: JSON.stringify({
        excluded_indices: excludedIndices,
        added_quotes: addedQuotes && addedQuotes.length > 0 ? addedQuotes : null,
      }),
    }),

  getNodeDistribution: (ticker: string) =>
    request<NodeDistributionResponse>(`/node/${ticker}/distribution`),

  sviOverridePrior: (ticker: string, params: { a: number; b: number; rho: number; m: number; sigma: number }) =>
    request<DistributionView>(`/prior/${ticker}/svi-override`, {
      method: "POST",
      body: JSON.stringify(params),
    }),

  sviOverrideSmile: (ticker: string, params: { a: number; b: number; rho: number; m: number; sigma: number }) =>
    request<SmileData>(`/smile/${ticker}/svi-override`, {
      method: "POST",
      body: JSON.stringify(params),
    }),

  // Prior save/load
  listSavedPriors: () =>
    request<Record<string, { ticker: string; filename: string; timestamp: string }>>(
      "/priors/saved"
    ),

  savePrior: (ticker: string, excludedIndices?: number[], addedQuotes?: [number, number][]) =>
    request<{ status: string; filename: string; ticker: string }>(
      `/priors/save/${ticker}`,
      {
        method: "POST",
        body: JSON.stringify({
          excluded_indices: excludedIndices ?? [],
          added_quotes: addedQuotes ?? [],
        }),
      }
    ),

  loadPriorFromFile: (ticker: string) =>
    request<{
      status: string;
      ticker: string;
      excluded_indices: number[];
      added_quotes: number[][];
      chain_snapshot: QuoteSnapshot | null;
    }>(`/priors/load/${ticker}`, {
      method: "POST",
    }),

  // Rates & forwards
  getTreasuryCurve: () =>
    request<{ date: string; tenors: number[]; rates: number[] }>("/rates/treasury"),

  getRatesConfig: () =>
    request<{ repo_rate_gc: number; repo_overrides: Record<string, number> }>("/rates/config"),

  updateRatesConfig: (config: { repo_rate_gc: number; repo_overrides: Record<string, number> }) =>
    request<{ status: string }>("/rates/config", {
      method: "PUT",
      body: JSON.stringify(config),
    }),

  setForwardOverride: (ticker: string, forward: number | null) =>
    request<{ status: string; ticker: string; forward: number; forward_model: number; forward_parity: number | null }>(
      `/forward/${ticker}`,
      { method: "PUT", body: JSON.stringify({ forward }) },
    ),

  setPriorForwardOverride: (ticker: string, forward: number | null) =>
    request<{ status: string }>(`/forward/${ticker}/prior`, {
      method: "PUT",
      body: JSON.stringify({ forward }),
    }),
};
