import { create } from "zustand";
import type {
  Asset,
  QuoteSnapshot,
  SolveResponse,
  GraphData,
  CatalogResponse,
} from "../types";
import { api } from "../api/client";

interface EngineState {
  // Data
  catalog: Asset[];
  activeTickers: string[];
  universe: Asset[];
  quotes: Record<string, QuoteSnapshot>;
  solveResult: SolveResponse | null;
  graphData: GraphData | null;

  // UI state
  selectedNode: string | null;
  loading: boolean;
  error: string | null;

  // Hyperparameters
  lambda: number;
  eta: number;
  lambdaPrior: number;
  useBidAskFit: boolean;
  shockNudges: Record<string, number>;
  alphaOverrides: Record<string, number>;

  // Observed/unobserved toggle and quote exclusion
  observedTickers: string[];
  excludedQuotes: Record<string, number[]>;
  excludedPriorQuotes: Record<string, number[]>;
  addedQuotes: Record<string, [number, number][]>;      // ticker -> [[strike, iv], ...]
  addedPriorQuotes: Record<string, [number, number][]>; // ticker -> [[strike, iv], ...]

  // Smile model
  smileModel: string;  // "svi" | "lqd" | "sigmoid"
  setSmileModel: (model: string) => void;

  // Computation tracking
  computing: number;  // >0 means background work in progress

  // Prior state
  priorsCalibrated: boolean;
  priorsVersion: number;  // incremented when priors change, forces Prior tab refresh

  // Universe dirty flag (unsaved selection changes)
  universeUnsaved: boolean;

  // Actions
  fetchCatalog: () => Promise<void>;
  fetchUniverse: () => Promise<void>;
  selectUniverse: (tickers: string[]) => Promise<void>;
  saveSelection: () => Promise<void>;
  addTicker: (ticker: string, name?: string, sector?: string) => Promise<void>;
  fetchPriors: () => Promise<void>;
  fetchSnapshot: () => Promise<void>;
  restoreQuotes: () => Promise<void>;
  fit: () => Promise<void>;       // lightweight: SVI for observed, prior for unobserved
  propagate: () => Promise<void>;  // full: graph propagation (only via Propagate button)
  fetchGraph: () => Promise<void>;
  setSelectedNode: (ticker: string | null) => void;
  setLambda: (v: number) => void;
  setEta: (v: number) => void;
  setLambdaPrior: (v: number) => void;
  setUseBidAskFit: (v: boolean) => void;
  setShockNudge: (ticker: string, nudge: number) => void;
  setAlphaOverride: (ticker: string, alpha: number) => void;
  updateW: (W: number[][]) => Promise<void>;
  toggleObserved: (ticker: string) => void;
  toggleQuotePoint: (ticker: string, index: number) => void;
  resetExclusions: (ticker: string) => void;
  addQuotePoint: (ticker: string, strike: number, iv: number) => void;
  removeAddedQuote: (ticker: string, index: number) => void;
  resetAdditions: (ticker: string) => void;
  calibrateAllPriors: () => Promise<void>;
  fitAllSnapshots: () => Promise<void>;
  fitSingleAsset: (ticker: string) => Promise<void>;
  calibrateSinglePrior: (ticker: string) => Promise<void>;
  togglePriorQuotePoint: (ticker: string, index: number) => void;
  resetPriorExclusions: (ticker: string) => void;
  addPriorQuotePoint: (ticker: string, strike: number, iv: number) => void;
  removeAddedPriorQuote: (ticker: string, index: number) => void;
  resetPriorAdditions: (ticker: string) => void;
}

let _propagateSeq = 0;

/** Recalculate the currently selected node (prior + snapshot if observed). */
function _recalcSelected(get: () => EngineState) {
  const { selectedNode, quotes, observedTickers } = get();
  if (!selectedNode || Object.keys(quotes).length === 0) return;
  get().calibrateSinglePrior(selectedNode);
  if (observedTickers.includes(selectedNode)) {
    get().fitSingleAsset(selectedNode);
  }
}

export const useEngine = create<EngineState>((set, get) => ({
  catalog: [],
  activeTickers: [],
  universe: [],
  quotes: {},
  solveResult: null,
  graphData: null,
  selectedNode: null,
  loading: false,
  error: null,
  lambda: 1.0,
  eta: 0.01,
  lambdaPrior: 0.10,
  useBidAskFit: true,
  shockNudges: {},
  alphaOverrides: {},
  observedTickers: [],
  excludedQuotes: {},
  excludedPriorQuotes: {},
  addedQuotes: {},
  addedPriorQuotes: {},
  smileModel: "svi",
  computing: 0,
  priorsCalibrated: false,
  priorsVersion: 0,
  universeUnsaved: false,

  fetchCatalog: async () => {
    try {
      const resp = await api.getCatalog();
      set({ catalog: resp.assets, activeTickers: resp.active_tickers });
    } catch (e: any) {
      set({ error: e.message });
    }
  },

  fetchUniverse: async () => {
    try {
      const universe = await api.getUniverse();
      set({ universe });
    } catch (e: any) {
      set({ error: e.message });
    }
  },

  selectUniverse: async (tickers: string[]) => {
    set({ loading: true, error: null });
    try {
      const resp = await api.selectUniverse(tickers);
      set({
        activeTickers: resp.tickers,
        universe: resp.graph.assets,
        graphData: resp.graph,
        universeUnsaved: true,
        // Clear stale state
        quotes: {},
        solveResult: null,
        observedTickers: [],
        excludedQuotes: {},
        addedQuotes: {},
        excludedPriorQuotes: {},
        addedPriorQuotes: {},
        priorsCalibrated: false,
        selectedNode: null,
        loading: false,
      });
    } catch (e: any) {
      set({ error: e.message, loading: false });
    }
  },

  saveSelection: async () => {
    try {
      await api.saveUniverseSelection();
      set({ universeUnsaved: false });
    } catch (e: any) {
      set({ error: e.message });
    }
  },

  addTicker: async (ticker: string, name?: string, sector?: string) => {
    set({ loading: true, error: null });
    try {
      const resp = await api.addTicker(ticker, name, sector);
      // Add to catalog if not present, mark as active
      set((s) => {
        const inCatalog = s.catalog.some((a) => a.ticker === resp.asset.ticker);
        return {
          catalog: inCatalog ? s.catalog : [...s.catalog, resp.asset],
          activeTickers: resp.tickers,
          universeUnsaved: true,
          loading: false,
        };
      });
      // Refresh universe + graph
      await get().fetchUniverse();
      await get().fetchGraph();
    } catch (e: any) {
      set({ error: e.message, loading: false });
      throw e;  // re-throw so the UI can catch and display
    }
  },

  fetchPriors: async () => {
    set({ loading: true, error: null });
    try {
      const quotes = await api.fetchQuotes();
      set({
        quotes,
        observedTickers: Object.keys(quotes),
        excludedQuotes: {},
        excludedPriorQuotes: {},
        solveResult: null,
        loading: false,
      });
    } catch (e: any) {
      set({ error: e.message, loading: false });
    }
  },

  restoreQuotes: async () => {
    // Restore cached quotes from backend on page load / hot-reload
    try {
      const quotes = await api.getLatestQuotes();
      if (Object.keys(quotes).length > 0) {
        const { observedTickers } = get();
        const tickers = Object.keys(quotes);
        const kept = observedTickers.length > 0
          ? observedTickers.filter((t) => tickers.includes(t))
          : tickers;
        set({ quotes, observedTickers: kept });
      }
    } catch {
      // Silently ignore — quotes will be empty until user fetches
    }
  },

  fetchSnapshot: async () => {
    set({ loading: true, error: null });
    try {
      const quotes = await api.fetchQuotes();
      // Keep existing observedTickers if already set, otherwise default to all
      const { observedTickers } = get();
      const tickers = Object.keys(quotes);
      const kept = observedTickers.length > 0
        ? observedTickers.filter((t) => tickers.includes(t))
        : tickers;
      set({
        quotes,
        observedTickers: kept,
        loading: false,
      });
    } catch (e: any) {
      set({ error: e.message, loading: false });
    }
  },

  fit: async () => {
    const { observedTickers, excludedQuotes, addedQuotes, lambdaPrior, useBidAskFit, smileModel } = get();
    const seq = ++_propagateSeq;
    set({ loading: true, error: null });
    try {
      const excl = Object.keys(excludedQuotes).length > 0 ? excludedQuotes : null;
      const added = Object.keys(addedQuotes).length > 0 ? addedQuotes : null;
      const result = await api.fit({
        observed_tickers: observedTickers.length > 0 ? observedTickers : null,
        excluded_quotes: excl,
        added_quotes: added,
        lambda_prior: lambdaPrior,
        use_bid_ask_fit: useBidAskFit,
        smile_model: smileModel,
      });
      if (seq === _propagateSeq) {
        // Build a minimal solveResult-like object for display
        set({
          solveResult: {
            nodes: result.nodes,
            W: get().solveResult?.W ?? [],
            alphas: get().solveResult?.alphas ?? [],
            tickers: result.tickers,
            propagation_matrix: null,
            neumann_terms: null,
            influence_scores: null,
            wasserstein_distances: null,
          },
          loading: false,
        });
      }
    } catch (e: any) {
      if (seq === _propagateSeq) {
        set({ error: e.message, loading: false });
      }
    }
  },

  calibrateAllPriors: async () => {
    const { smileModel } = get();
    set((s) => ({ loading: true, error: null, computing: s.computing + 1 }));
    try {
      await api.calibratePriors(smileModel);
      set((s) => ({ priorsCalibrated: true, loading: false, priorsVersion: s.priorsVersion + 1, computing: s.computing - 1 }));
    } catch (e: any) {
      set((s) => ({ error: e.message, loading: false, computing: s.computing - 1 }));
    }
  },

  fitAllSnapshots: async () => {
    set({ loading: true, error: null });
    try {
      await get().fit();
      set({ loading: false });
    } catch (e: any) {
      set({ error: e.message, loading: false });
    }
  },

  fitSingleAsset: async (ticker: string) => {
    const { excludedQuotes, addedQuotes, lambdaPrior, useBidAskFit, smileModel } = get();
    set((s) => ({ computing: s.computing + 1 }));
    try {
      const excl = excludedQuotes[ticker] ? { [ticker]: excludedQuotes[ticker] } : null;
      const added = addedQuotes[ticker] ? { [ticker]: addedQuotes[ticker] } : null;
      const result = await api.fitSingle(ticker, {
        excluded_quotes: excl,
        added_quotes: added,
        lambda_prior: lambdaPrior,
        use_bid_ask_fit: useBidAskFit,
        smile_model: smileModel,
      });
      // Update just this ticker in solveResult
      set((s) => {
        const nodes = { ...(s.solveResult?.nodes ?? {}), [ticker]: result };
        return {
          solveResult: {
            ...(s.solveResult ?? { W: [], alphas: [], tickers: Object.keys(nodes), propagation_matrix: null, neumann_terms: null, influence_scores: null, wasserstein_distances: null }),
            nodes,
          },
          computing: s.computing - 1,
        };
      });
    } catch (e: any) {
      set((s) => ({ error: e.message, computing: s.computing - 1 }));
    }
  },

  calibrateSinglePrior: async (ticker: string) => {
    const { smileModel } = get();
    set((s) => ({ computing: s.computing + 1 }));
    try {
      await api.calibrateSinglePrior(ticker, smileModel);
      set((s) => ({ priorsVersion: s.priorsVersion + 1, computing: s.computing - 1 }));
    } catch (e: any) {
      set((s) => ({ error: e.message, computing: s.computing - 1 }));
    }
  },

  propagate: async () => {
    const {
      lambda, eta, lambdaPrior, useBidAskFit, smileModel, shockNudges, alphaOverrides, graphData,
      observedTickers, excludedQuotes, addedQuotes,
    } = get();
    const seq = ++_propagateSeq;
    set({ loading: true, error: null });
    try {
      const excl = Object.keys(excludedQuotes).length > 0 ? excludedQuotes : null;
      const added = Object.keys(addedQuotes).length > 0 ? addedQuotes : null;
      const result = await api.solve({
        lambda_: lambda,
        eta: eta,
        lambda_prior: lambdaPrior,
        use_bid_ask_fit: useBidAskFit,
        smile_model: smileModel,
        shock_nudges: Object.keys(shockNudges).length > 0 ? shockNudges : null,
        alpha_overrides:
          Object.keys(alphaOverrides).length > 0 ? alphaOverrides : null,
        W_override: graphData?.W ?? null,
        observed_tickers: observedTickers.length > 0 ? observedTickers : null,
        excluded_quotes: excl,
        added_quotes: added,
      });
      // Only apply if this is still the latest request
      if (seq === _propagateSeq) {
        set({ solveResult: result, loading: false });
      }
    } catch (e: any) {
      if (seq === _propagateSeq) {
        set({ error: e.message, loading: false });
      }
    }
  },

  fetchGraph: async () => {
    try {
      const graphData = await api.getGraph();
      set({ graphData });
    } catch (e: any) {
      set({ error: e.message });
    }
  },

  setSelectedNode: (ticker) => {
    set({ selectedNode: ticker });
    if (ticker && Object.keys(get().quotes).length > 0) {
      get().calibrateSinglePrior(ticker);
      if (get().observedTickers.includes(ticker)) {
        get().fitSingleAsset(ticker);
      }
    }
  },
  setLambda: (v) => {
    set({ lambda: v });
    _recalcSelected(get);
  },
  setEta: (v) => {
    set({ eta: v });
    _recalcSelected(get);
  },
  setLambdaPrior: (v) => {
    set({ lambdaPrior: v });
    _recalcSelected(get);
  },
  setUseBidAskFit: (v) => {
    set({ useBidAskFit: v });
    _recalcSelected(get);
  },
  setSmileModel: (model) => {
    set({ smileModel: model });
    _recalcSelected(get);
  },

  setShockNudge: (ticker, nudge) =>
    set((s) => ({
      shockNudges: { ...s.shockNudges, [ticker]: nudge },
    })),

  setAlphaOverride: (ticker, alpha) =>
    set((s) => ({
      alphaOverrides: { ...s.alphaOverrides, [ticker]: alpha },
    })),

  updateW: async (W) => {
    try {
      await api.updateGraph(W);
      set((s) => ({
        graphData: s.graphData ? { ...s.graphData, W } : null,
      }));
    } catch (e: any) {
      set({ error: e.message });
    }
  },

  toggleObserved: (ticker) => {
    set((s) => {
      const isObs = s.observedTickers.includes(ticker);
      return {
        observedTickers: isObs
          ? s.observedTickers.filter((t) => t !== ticker)
          : [...s.observedTickers, ticker],
      };
    });
    if (Object.keys(get().quotes).length > 0) {
      get().fit();
    }
  },

  toggleQuotePoint: (ticker, index) => {
    set((s) => {
      const current = s.excludedQuotes[ticker] ?? [];
      const isExcluded = current.includes(index);
      const next = isExcluded
        ? current.filter((i) => i !== index)
        : [...current, index];
      return {
        excludedQuotes: {
          ...s.excludedQuotes,
          [ticker]: next,
        },
      };
    });
    get().fitSingleAsset(ticker);
  },

  resetExclusions: (ticker) => {
    set((s) => {
      const { [ticker]: _, ...rest } = s.excludedQuotes;
      return { excludedQuotes: rest };
    });
    get().fitSingleAsset(ticker);
  },

  togglePriorQuotePoint: (ticker, index) =>
    set((s) => {
      const current = s.excludedPriorQuotes[ticker] ?? [];
      const isExcluded = current.includes(index);
      return {
        excludedPriorQuotes: {
          ...s.excludedPriorQuotes,
          [ticker]: isExcluded
            ? current.filter((i) => i !== index)
            : [...current, index],
        },
      };
    }),

  addQuotePoint: (ticker, strike, iv) => {
    set((s) => ({
      addedQuotes: {
        ...s.addedQuotes,
        [ticker]: [...(s.addedQuotes[ticker] ?? []), [strike, iv]],
      },
    }));
    get().fitSingleAsset(ticker);
  },

  removeAddedQuote: (ticker, index) => {
    set((s) => {
      const current = s.addedQuotes[ticker] ?? [];
      return {
        addedQuotes: {
          ...s.addedQuotes,
          [ticker]: current.filter((_, i) => i !== index),
        },
      };
    });
    get().fitSingleAsset(ticker);
  },

  resetAdditions: (ticker) => {
    set((s) => {
      const { [ticker]: _, ...rest } = s.addedQuotes;
      return { addedQuotes: rest };
    });
    get().fitSingleAsset(ticker);
  },

  resetPriorExclusions: (ticker) =>
    set((s) => {
      const { [ticker]: _, ...rest } = s.excludedPriorQuotes;
      return { excludedPriorQuotes: rest };
    }),

  addPriorQuotePoint: (ticker, strike, iv) => {
    set((s) => ({
      addedPriorQuotes: {
        ...s.addedPriorQuotes,
        [ticker]: [...(s.addedPriorQuotes[ticker] ?? []), [strike, iv]],
      },
    }));
  },

  removeAddedPriorQuote: (ticker, index) => {
    set((s) => {
      const current = s.addedPriorQuotes[ticker] ?? [];
      return {
        addedPriorQuotes: {
          ...s.addedPriorQuotes,
          [ticker]: current.filter((_, i) => i !== index),
        },
      };
    });
  },

  resetPriorAdditions: (ticker) => {
    set((s) => {
      const { [ticker]: _, ...rest } = s.addedPriorQuotes;
      return { addedPriorQuotes: rest };
    });
  },
}));
