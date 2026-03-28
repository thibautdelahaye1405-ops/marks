import { useState, useEffect, useCallback } from "react";
import { useEngine } from "../hooks/useEngine";
import { api } from "../api/client";
import DistributionTripleView from "./DistributionTripleView";
import SviSliders from "./SviSliders";
import Plot from "./Plot";
import type { DistributionView } from "../types";

type ViewMode = "smile" | "distributions";

export default function PriorCalibrationView() {
  const {
    selectedNode, quotes, observedTickers, priorsVersion,
    excludedPriorQuotes, togglePriorQuotePoint, resetPriorExclusions,
    addedPriorQuotes, addPriorQuotePoint, removeAddedPriorQuote, resetPriorAdditions,
  } = useEngine();
  const [priorView, setPriorView] = useState<DistributionView | null>(null);
  const [currentView, setCurrentView] = useState<DistributionView | null>(null);
  const [sviParams, setSviParams] = useState<{ a: number; b: number; rho: number; m: number; sigma: number } | null>(null);
  const [baseSviParams, setBaseSviParams] = useState<{ a: number; b: number; rho: number; m: number; sigma: number } | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>("smile");
  const [saveStatus, setSaveStatus] = useState<string | null>(null);

  const ticker = selectedNode;
  const isObserved = ticker ? observedTickers.includes(ticker) : false;
  const excluded = new Set(ticker ? excludedPriorQuotes[ticker] ?? [] : []);

  // Fetch prior when ticker changes or priors are reloaded
  useEffect(() => {
    if (!ticker) return;
    api.getPrior(ticker).then((view) => {
      setPriorView(view);
      setCurrentView(view);
      // Extract SVI params from beta field (a, b, rho, m, sigma)
      const b = view.beta;
      if (b && b.length >= 5) {
        const p = { a: b[0], b: b[1], rho: b[2], m: b[3], sigma: b[4] };
        setSviParams(p);
        setBaseSviParams(p);
      }
    }).catch(() => {
      setPriorView(null);
      setCurrentView(null);
    });
  }, [ticker, priorsVersion]);

  const addedPrior = ticker ? addedPriorQuotes[ticker] ?? [] : [];

  // Refit prior when exclusions or additions change
  useEffect(() => {
    if (!ticker || !isObserved) return;
    const excl = excludedPriorQuotes[ticker] ?? [];
    const added = addedPriorQuotes[ticker] ?? [];
    if (excl.length === 0 && added.length === 0) return;
    api.refitPrior(ticker, excl, added).then((view) => {
      setCurrentView(view);
      const b = view.beta;
      if (b && b.length >= 5) {
        const p = { a: b[0], b: b[1], rho: b[2], m: b[3], sigma: b[4] };
        setSviParams(p);
        setBaseSviParams(p);
      }
    }).catch(() => {});
  }, [ticker, isObserved, excludedPriorQuotes, addedPriorQuotes]);

  const handleSviChange = useCallback(
    (params: { a: number; b: number; rho: number; m: number; sigma: number }) => {
      if (!ticker) return;
      setSviParams(params);
      api.sviOverridePrior(ticker, params).then(setCurrentView).catch(() => {});
    },
    [ticker]
  );

  const handleSviReset = useCallback(() => {
    if (!ticker || !baseSviParams) return;
    setSviParams(baseSviParams);
    api.sviOverridePrior(ticker, baseSviParams).then(setCurrentView).catch(() => {});
  }, [ticker, baseSviParams]);

  const handleResetExclusions = useCallback(() => {
    if (!ticker) return;
    resetPriorExclusions(ticker);
    const added = addedPriorQuotes[ticker] ?? [];
    api.refitPrior(ticker, [], added).then((view) => {
      setCurrentView(view);
      setBeta(view.beta ?? [0, 0, 0, 0, 0]);
    }).catch(() => {});
  }, [ticker, resetPriorExclusions, addedPriorQuotes]);

  const handleResetAdditions = useCallback(() => {
    if (!ticker) return;
    resetPriorAdditions(ticker);
    const excl = excludedPriorQuotes[ticker] ?? [];
    api.refitPrior(ticker, excl, []).then((view) => {
      setCurrentView(view);
      setBeta(view.beta ?? [0, 0, 0, 0, 0]);
    }).catch(() => {});
  }, [ticker, resetPriorAdditions, excludedPriorQuotes]);

  const handlePriorQuoteClick = useCallback(
    (event: any) => {
      if (!ticker || !isObserved) return;
      const pt = event?.points?.[0];
      if (!pt) return;
      const idx = pt.customdata;
      if (typeof idx === "string" && idx.startsWith("added:")) {
        removeAddedPriorQuote(ticker, parseInt(idx.slice(6), 10));
      } else if (typeof idx === "number") {
        togglePriorQuotePoint(ticker, idx);
      }
    },
    [ticker, isObserved, togglePriorQuotePoint, removeAddedPriorQuote]
  );

  const handlePriorDoubleClick = useCallback(
    (coords: { x: number; y: number }) => {
      if (!ticker || !isObserved) return;
      // x = strike, y = IV in %
      const iv = coords.y / 100;
      if (iv > 0.005 && iv < 5.0 && coords.x > 0) {
        addPriorQuotePoint(ticker, coords.x, iv);
      }
    },
    [ticker, isObserved, addPriorQuotePoint]
  );

  const handleSavePrior = useCallback(async () => {
    if (!ticker) return;
    setSaveStatus("saving...");
    try {
      const excl = excludedPriorQuotes[ticker] ?? [];
      const added = addedPriorQuotes[ticker] ?? [];
      await api.savePrior(ticker, excl, added);
      setSaveStatus("saved");
      setTimeout(() => setSaveStatus(null), 2000);
    } catch (e: any) {
      setSaveStatus("error");
      setTimeout(() => setSaveStatus(null), 3000);
    }
  }, [ticker, excludedPriorQuotes, addedPriorQuotes]);

  if (!ticker) {
    return (
      <div style={{ padding: 20, color: "#94a3b8", textAlign: "center" }}>
        Select a node to calibrate its prior
      </div>
    );
  }

  if (!priorView) {
    return (
      <div style={{ padding: 20, color: "#94a3b8", textAlign: "center" }}>
        No prior data for {ticker}. Fetch quotes and calibrate first.
      </div>
    );
  }

  const quoteData = quotes[ticker];
  const clean = (arr: (number | null)[]) => arr.map((v) => (v != null ? v * 100 : NaN));

  // Build prev close quote traces (split into included/excluded)
  const prevCloseTraces: Plotly.Data[] = [];
  if (quoteData?.prev_close_ivs) {
    const incStrikes: number[] = [];
    const incIvs: number[] = [];
    const incCustom: number[] = [];
    const excStrikes: number[] = [];
    const excIvs: number[] = [];
    const excCustom: number[] = [];

    quoteData.strikes.forEach((k, i) => {
      const iv = quoteData.prev_close_ivs?.[i];
      if (iv == null) return;
      if (excluded.has(i)) {
        excStrikes.push(k);
        excIvs.push(iv * 100);
        excCustom.push(i);
      } else {
        incStrikes.push(k);
        incIvs.push(iv * 100);
        incCustom.push(i);
      }
    });

    prevCloseTraces.push({
      x: incStrikes,
      y: incIvs,
      mode: "markers",
      name: "Prev close",
      marker: {
        color: "#f59e0b",
        size: isObserved ? 7 : 5,
        symbol: "diamond",
      },
      customdata: incCustom,
      hovertemplate: isObserved
        ? "K=%{x:.0f} IV=%{y:.1f}%<br><i>Click to exclude</i><extra></extra>"
        : "K=%{x:.0f} IV=%{y:.1f}%<extra></extra>",
    } as Plotly.Data);

    if (excStrikes.length > 0) {
      prevCloseTraces.push({
        x: excStrikes,
        y: excIvs,
        mode: "markers",
        name: "Excluded prev",
        marker: { color: "#64748b", size: 6, symbol: "x", opacity: 0.5 },
        customdata: excCustom,
        hovertemplate: "K=%{x:.0f} IV=%{y:.1f}%<br><i>Click to restore</i><extra></extra>",
      } as Plotly.Data);
    }
  }

  // User-added synthetic prior quotes (clickable to remove)
  if (addedPrior.length > 0) {
    prevCloseTraces.push({
      x: addedPrior.map((p) => p[0]),
      y: addedPrior.map((p) => p[1] * 100),
      mode: "markers",
      name: "Added",
      marker: { color: "#f59e0b", size: 9, symbol: "star", line: { width: 1, color: "#fff" } },
      customdata: addedPrior.map((_, i) => `added:${i}`),
      hovertemplate: "K=%{x:.0f} IV=%{y:.1f}%<br><i>Click to remove</i><extra></extra>",
    } as Plotly.Data);
  }

  return (
    <div style={{ padding: "8px 12px", overflow: "auto" }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
        <h3 style={{ margin: 0, fontSize: 14, color: "#e2e8f0" }}>
          Prior: {ticker}
          {!isObserved && <span style={{ fontSize: 11, color: "#f97316", marginLeft: 8 }}>(inferred)</span>}
        </h3>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          {saveStatus && (
            <span style={{ fontSize: 10, color: saveStatus === "saved" ? "#22c55e" : saveStatus === "error" ? "#f87171" : "#94a3b8" }}>
              {saveStatus === "saved" ? "Saved" : saveStatus === "error" ? "Failed" : "Saving..."}
            </span>
          )}
          <button onClick={handleSavePrior} style={saveBtnStyle}>
            Save Prior
          </button>
        </div>
      </div>

      {/* View toggle */}
      <div style={{ display: "flex", gap: 4, marginBottom: 8 }}>
        {(["smile", "distributions"] as ViewMode[]).map((m) => (
          <button
            key={m}
            onClick={() => setViewMode(m)}
            style={{
              fontSize: 11,
              padding: "3px 10px",
              background: viewMode === m ? "#334155" : "transparent",
              color: viewMode === m ? "#e2e8f0" : "#64748b",
              border: "1px solid #334155",
              borderRadius: 4,
              cursor: "pointer",
            }}
          >
            {m === "smile" ? "Smile (IV vs Strike)" : "IV / CDF / LQD"}
          </button>
        ))}
      </div>

      {/* Smile view */}
      {viewMode === "smile" && (
        <>
          <Plot
            data={[
              // Prev close quotes (clickable for observed, display-only for inferred)
              ...prevCloseTraces,
              // Fitted prior curve
              {
                x: (currentView ?? priorView).moneyness.map((m) =>
                  quoteData ? quoteData.forward * Math.exp(m) : Math.exp(m) * 100
                ),
                y: clean((currentView ?? priorView).iv_curve),
                mode: "lines" as const,
                name: "Fitted prior",
                line: { color: "#22c55e", width: 2.5 },
              },
              // Current market quotes
              ...(quoteData
                ? [
                    {
                      x: quoteData.strikes,
                      y: quoteData.mid_ivs.map((v) => v * 100),
                      mode: "markers" as const,
                      name: "Current quotes",
                      marker: { color: "#3b82f6", size: 5, symbol: "circle" as const },
                    },
                  ]
                : []),
            ]}
            layout={{
              title: { text: `${ticker} -- Prior Smile`, font: { color: "#e2e8f0", size: 13 } },
              paper_bgcolor: "#1e293b",
              plot_bgcolor: "#1e293b",
              font: { color: "#94a3b8" },
              shapes: [
                // Prior's forward (prev close)
                ...((currentView ?? priorView).fit_forward ? [{
                  type: "line" as const, x0: (currentView ?? priorView).fit_forward!, x1: (currentView ?? priorView).fit_forward!,
                  y0: 0, y1: 1, yref: "paper" as const,
                  line: { color: "#22c55e", width: 1, dash: "dot" as const },
                }] : []),
                // Current forward
                ...(quoteData ? [{
                  type: "line" as const, x0: quoteData.forward, x1: quoteData.forward,
                  y0: 0, y1: 1, yref: "paper" as const,
                  line: { color: "#3b82f6", width: 1, dash: "dot" as const },
                }] : []),
              ],
              annotations: [
                ...((currentView ?? priorView).fit_forward ? [{
                  x: (currentView ?? priorView).fit_forward!, y: 1.05, yref: "paper" as const,
                  text: `F_prev=${(currentView ?? priorView).fit_forward!.toFixed(0)}`,
                  showarrow: false, font: { color: "#22c55e", size: 9 },
                }] : []),
                ...(quoteData ? [{
                  x: quoteData.forward, y: 1.02, yref: "paper" as const,
                  text: `F=${quoteData.forward.toFixed(0)}`,
                  showarrow: false, font: { color: "#3b82f6", size: 9 },
                }] : []),
              ],
              xaxis: { title: "Strike", gridcolor: "#334155", zerolinecolor: "#334155" },
              yaxis: { title: "Implied Vol (%)", gridcolor: "#334155", zerolinecolor: "#334155" },
              legend: { x: 1, y: 1, xanchor: "right", bgcolor: "rgba(0,0,0,0)", font: { size: 9 } },
              margin: { t: 35, r: 15, b: 45, l: 55 },
              autosize: true,
              height: 350,
            }}
            style={{ width: "100%", height: 350 }}
            config={{ displayModeBar: false, doubleClick: false }}
            onClick={isObserved ? handlePriorQuoteClick : undefined}
            onDoubleClick={isObserved ? handlePriorDoubleClick : undefined}
          />
          {isObserved && (excluded.size > 0 || addedPrior.length > 0) && (
            <div style={{ padding: "4px 0", display: "flex", justifyContent: "flex-end", gap: 8 }}>
              {excluded.size > 0 && (
                <button onClick={handleResetExclusions} style={actionBtnStyle}>
                  Reset exclusions ({excluded.size})
                </button>
              )}
              {addedPrior.length > 0 && (
                <button onClick={handleResetAdditions} style={actionBtnStyle}>
                  Forget additions ({addedPrior.length})
                </button>
              )}
            </div>
          )}
        </>
      )}

      {/* Distribution triple view */}
      {viewMode === "distributions" && (
        <DistributionTripleView
          prior={priorView}
          current={currentView}
          priorLabel="Initial"
          currentLabel="Calibrated"
          currentColor="#22c55e"
          height={165}
        />
      )}

      {/* SVI parameter sliders */}
      {sviParams && (
        <div style={{ borderTop: "1px solid #334155", marginTop: 8, paddingTop: 8 }}>
          <div style={{ fontSize: 11, color: "#94a3b8", marginBottom: 4 }}>
            SVI Parameters
          </div>
          <SviSliders
            values={sviParams}
            baseValues={baseSviParams ?? undefined}
            onChange={handleSviChange}
            onReset={handleSviReset}
          />
        </div>
      )}
    </div>
  );
}

const actionBtnStyle: React.CSSProperties = {
  fontSize: 11,
  color: "#94a3b8",
  background: "#1e293b",
  border: "1px solid #334155",
  borderRadius: 4,
  padding: "3px 10px",
  cursor: "pointer",
};

const saveBtnStyle: React.CSSProperties = {
  fontSize: 11,
  padding: "4px 12px",
  background: "#6366f1",
  color: "#fff",
  border: "none",
  borderRadius: 4,
  cursor: "pointer",
  fontWeight: 600,
};
