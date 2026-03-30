import { useState, useEffect, useCallback, useRef } from "react";
import { useEngine } from "../hooks/useEngine";
import { api } from "../api/client";
import DistributionTripleView from "./DistributionTripleView";
import SviSliders from "./SviSliders";
import LqdSliders from "./LqdSliders";
import type { LqdTraderParams } from "./LqdSliders";
import SigmoidSliders from "./SigmoidSliders";
import type { SigmoidTraderParams } from "./SigmoidSliders";
import Plot from "./Plot";
import type { DistributionView } from "../types";

type ViewMode = "smile" | "distributions";

export default function PriorCalibrationView() {
  const {
    selectedNode, quotes, observedTickers, priorsVersion, smileModel,
    excludedPriorQuotes, togglePriorQuotePoint, resetPriorExclusions,
    addedPriorQuotes, addPriorQuotePoint, removeAddedPriorQuote, resetPriorAdditions,
  } = useEngine();
  const [priorView, setPriorView] = useState<DistributionView | null>(null);
  const [currentView, setCurrentView] = useState<DistributionView | null>(null);
  const [sviParams, setSviParams] = useState<{ v: number; psi_hat: number; p_hat: number; c_hat: number; vt_ratio: number } | null>(null);
  const [baseSviParams, setBaseSviParams] = useState<{ v: number; psi_hat: number; p_hat: number; c_hat: number; vt_ratio: number } | null>(null);
  const [lqdParams, setLqdParams] = useState<LqdTraderParams | null>(null);
  const [baseLqdParams, setBaseLqdParams] = useState<LqdTraderParams | null>(null);
  const [sigmoidParams, setSigmoidParams] = useState<SigmoidTraderParams | null>(null);
  const [baseSigmoidParams, setBaseSigmoidParams] = useState<SigmoidTraderParams | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>("smile");
  const [saveStatus, setSaveStatus] = useState<string | null>(null);

  const ticker = selectedNode;
  const isObserved = ticker ? observedTickers.includes(ticker) : false;
  const excluded = new Set(ticker ? excludedPriorQuotes[ticker] ?? [] : []);

  // Prior forward adjustment (F_prev, separate from current smile's forward)
  const [fwdOverride, setFwdOverride] = useState<number | null>(null);
  const fwdTimerRef = useRef<ReturnType<typeof setTimeout>>();

  useEffect(() => { setFwdOverride(null); }, [ticker]);

  const quoteDataForFwd = ticker ? quotes[ticker] : null;
  // Prior uses prev_spot-based forward, not current forward
  const prevSpot = quoteDataForFwd?.prev_spot ?? quoteDataForFwd?.spot ?? 100;
  const modelPrevFwd = quoteDataForFwd?.forward_model != null
    ? prevSpot / (quoteDataForFwd?.spot ?? 1) * quoteDataForFwd.forward_model
    : prevSpot;
  const effectivePrevFwd = fwdOverride ?? (currentView?.fit_forward ?? modelPrevFwd);

  // Update view + model params (but NOT base) — used by forward slider
  const _updateViewOnly = useCallback((view: DistributionView) => {
    setCurrentView(view);
    const b = view.beta;
    if (smileModel === "lqd" && b && b.length >= 6) {
      setLqdParams({ min_iv: b[0], atm_skew: b[1], atm_curv: b[2], put_slope: b[3], call_slope: b[4], shoulder: b[5] });
    } else if (smileModel === "sigmoid" && b && b.length >= 6) {
      setSigmoidParams({ sigma_atm: b[0], s_atm: b[1], k_atm: b[2], w_p: b[3], w_c: b[4], sigma_min: b[5] });
    } else if (b && b.length >= 5) {
      setSviParams({ v: b[0], psi_hat: b[1], p_hat: b[2], c_hat: b[3], vt_ratio: b[4] });
    }
  }, [smileModel]);

  // Update view + model params AND base — used by exclusion/addition changes
  const _updateViewAndBase = useCallback((view: DistributionView) => {
    setCurrentView(view);
    const b = view.beta;
    if (smileModel === "lqd" && b && b.length >= 6) {
      const lp: LqdTraderParams = { min_iv: b[0], atm_skew: b[1], atm_curv: b[2], put_slope: b[3], call_slope: b[4], shoulder: b[5] };
      setLqdParams(lp);
      setBaseLqdParams(lp);
    } else if (smileModel === "sigmoid" && b && b.length >= 6) {
      const sp: SigmoidTraderParams = { sigma_atm: b[0], s_atm: b[1], k_atm: b[2], w_p: b[3], w_c: b[4], sigma_min: b[5] };
      setSigmoidParams(sp);
      setBaseSigmoidParams(sp);
    } else if (b && b.length >= 5) {
      const p = { v: b[0], psi_hat: b[1], p_hat: b[2], c_hat: b[3], vt_ratio: b[4] };
      setSviParams(p);
      setBaseSviParams(p);
    }
  }, [smileModel]);

  const handleFwdChange = useCallback((val: number) => {
    if (!ticker) return;
    setFwdOverride(val);
    clearTimeout(fwdTimerRef.current);
    fwdTimerRef.current = setTimeout(() => {
      api.setPriorForwardOverride(ticker, val).then(() => {
        const excl = excludedPriorQuotes[ticker] ?? [];
        const added = addedPriorQuotes[ticker] ?? [];
        api.refitPrior(ticker, excl, added).then(_updateViewOnly).catch(() => {});
      }).catch(() => {});
    }, 150);
  }, [ticker, excludedPriorQuotes, addedPriorQuotes, _updateViewOnly]);

  const handleFwdReset = useCallback(() => {
    if (!ticker) return;
    setFwdOverride(null);
    api.setPriorForwardOverride(ticker, null).then(() => {
      const excl = excludedPriorQuotes[ticker] ?? [];
      const added = addedPriorQuotes[ticker] ?? [];
      // Reset restores base, so update both
      api.refitPrior(ticker, excl, added).then(_updateViewAndBase).catch(() => {});
    }).catch(() => {});
  }, [ticker, excludedPriorQuotes, addedPriorQuotes, _updateViewAndBase]);

  const fwdRef = useRef(effectivePrevFwd);
  fwdRef.current = effectivePrevFwd;
  const nudgeFwd = useCallback((dir: number) => {
    const step = modelPrevFwd * 0.001;
    handleFwdChange(fwdRef.current + dir * step);
  }, [modelPrevFwd, handleFwdChange]);

  // Fetch prior when ticker changes or priors are reloaded
  useEffect(() => {
    if (!ticker) return;
    api.getPrior(ticker, smileModel).then((view) => {
      setPriorView(view);
      setCurrentView(view);
      const b = view.beta;
      if (smileModel === "lqd" && b && b.length >= 6) {
        const lp: LqdTraderParams = { min_iv: b[0], atm_skew: b[1], atm_curv: b[2], put_slope: b[3], call_slope: b[4], shoulder: b[5] };
        setLqdParams(lp);
        setBaseLqdParams(lp);
      } else if (smileModel === "sigmoid" && b && b.length >= 6) {
        const sp: SigmoidTraderParams = { sigma_atm: b[0], s_atm: b[1], k_atm: b[2], w_p: b[3], w_c: b[4], sigma_min: b[5] };
        setSigmoidParams(sp);
        setBaseSigmoidParams(sp);
      } else if (b && b.length >= 5) {
        const p = { v: b[0], psi_hat: b[1], p_hat: b[2], c_hat: b[3], vt_ratio: b[4] };
        setSviParams(p);
        setBaseSviParams(p);
      }
    }).catch(() => {
      setPriorView(null);
      setCurrentView(null);
    });
  }, [ticker, priorsVersion, smileModel]);

  const addedPrior = ticker ? addedPriorQuotes[ticker] ?? [] : [];

  // Refit prior when exclusions or additions change
  useEffect(() => {
    if (!ticker || !isObserved) return;
    const excl = excludedPriorQuotes[ticker] ?? [];
    const added = addedPriorQuotes[ticker] ?? [];
    if (excl.length === 0 && added.length === 0) return;
    api.refitPrior(ticker, excl, added).then(_updateViewAndBase).catch(() => {});
  }, [ticker, isObserved, excludedPriorQuotes, addedPriorQuotes, _updateViewAndBase]);

  const handleSviChange = useCallback(
    (params: { v: number; psi_hat: number; p_hat: number; c_hat: number; vt_ratio: number }) => {
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

  // LQD parameter override handlers
  const handleLqdChange = useCallback(
    (params: LqdTraderParams) => {
      if (!ticker) return;
      setLqdParams(params);
      const theta = [params.min_iv, params.atm_skew, params.atm_curv,
                     params.put_slope, params.call_slope, params.shoulder];
      api.lqdOverridePrior(ticker, theta).then((view) => {
        setCurrentView(view);
      }).catch(() => {});
    },
    [ticker]
  );

  const handleLqdReset = useCallback(() => {
    if (!ticker || !baseLqdParams) return;
    setLqdParams(baseLqdParams);
    const theta = [baseLqdParams.min_iv, baseLqdParams.atm_skew, baseLqdParams.atm_curv,
                   baseLqdParams.put_slope, baseLqdParams.call_slope, baseLqdParams.shoulder];
    api.lqdOverridePrior(ticker, theta).then((view) => {
      setCurrentView(view);
    }).catch(() => {});
  }, [ticker, baseLqdParams]);

  // Sigmoid parameter override handlers
  const handleSigmoidChange = useCallback(
    (params: SigmoidTraderParams) => {
      if (!ticker) return;
      setSigmoidParams(params);
      const p = [params.sigma_atm, params.s_atm, params.k_atm,
                 params.w_p, params.w_c, params.sigma_min];
      api.sigmoidOverridePrior(ticker, p).then((view) => {
        setCurrentView(view);
      }).catch(() => {});
    },
    [ticker]
  );

  const handleSigmoidReset = useCallback(() => {
    if (!ticker || !baseSigmoidParams) return;
    setSigmoidParams(baseSigmoidParams);
    const p = [baseSigmoidParams.sigma_atm, baseSigmoidParams.s_atm, baseSigmoidParams.k_atm,
               baseSigmoidParams.w_p, baseSigmoidParams.w_c, baseSigmoidParams.sigma_min];
    api.sigmoidOverridePrior(ticker, p).then((view) => {
      setCurrentView(view);
    }).catch(() => {});
  }, [ticker, baseSigmoidParams]);

  const handleResetExclusions = useCallback(() => {
    if (!ticker) return;
    resetPriorExclusions(ticker);
    const added = addedPriorQuotes[ticker] ?? [];
    api.refitPrior(ticker, [], added).then(_updateViewAndBase).catch(() => {});
  }, [ticker, resetPriorExclusions, addedPriorQuotes, _updateViewAndBase]);

  const handleResetAdditions = useCallback(() => {
    if (!ticker) return;
    resetPriorAdditions(ticker);
    const excl = excludedPriorQuotes[ticker] ?? [];
    api.refitPrior(ticker, excl, []).then(_updateViewAndBase).catch(() => {});
  }, [ticker, resetPriorAdditions, excludedPriorQuotes, _updateViewAndBase]);

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
    <div style={{ overflow: "auto" }}>
      {/* View toggle + save */}
      <div style={{ display: "flex", gap: 4, padding: "8px 12px 4px", alignItems: "center" }}>
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
        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 6 }}>
          {isObserved && excluded.size > 0 && (
            <button onClick={handleResetExclusions} style={actionBtnStyle}>
              Reset exclusions ({excluded.size})
            </button>
          )}
          {isObserved && addedPrior.length > 0 && (
            <button onClick={handleResetAdditions} style={actionBtnStyle}>
              Forget additions ({addedPrior.length})
            </button>
          )}
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
                name: `Fitted prior (${smileModel.toUpperCase()})`,
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
                // Prior's forward (follows slider)
                {
                  type: "line" as const, x0: effectivePrevFwd, x1: effectivePrevFwd,
                  y0: 0, y1: 1, yref: "paper" as const,
                  line: { color: "#22c55e", width: 1, dash: "dot" as const },
                },
                // Current forward (reference)
                ...(quoteData ? [{
                  type: "line" as const, x0: quoteData.forward, x1: quoteData.forward,
                  y0: 0, y1: 1, yref: "paper" as const,
                  line: { color: "#3b82f6", width: 1, dash: "dot" as const },
                }] : []),
              ],
              annotations: [
                {
                  x: effectivePrevFwd, y: 1.05, yref: "paper" as const,
                  text: `F_prev=${effectivePrevFwd.toFixed(1)}`,
                  showarrow: false, font: { color: "#22c55e", size: 9 },
                },
                ...(quoteData ? [{
                  x: quoteData.forward, y: 1.02, yref: "paper" as const,
                  text: `F=${quoteData.forward.toFixed(0)}`,
                  showarrow: false, font: { color: "#3b82f6", size: 9 },
                }] : []),
              ],
              xaxis: { title: "Strike", gridcolor: "#334155", zerolinecolor: "#334155" },
              yaxis: { title: "Implied Vol (%)", gridcolor: "#334155", zerolinecolor: "#334155" },
              legend: { x: 1, y: 1, xanchor: "right", bgcolor: "rgba(0,0,0,0)", font: { size: 9 } },
              margin: { t: 35, r: 20, b: 45, l: 55 },
              autosize: true,
              height: 440,
            }}
            style={{ width: "100%", height: 440 }}
            config={{ displayModeBar: false, doubleClick: false }}
            onClick={isObserved ? handlePriorQuoteClick : undefined}
            onDoubleClick={isObserved ? handlePriorDoubleClick : undefined}
          />
          <div style={{ padding: "2px 16px", fontSize: 10, color: "#94a3b8", display: "flex", alignItems: "center", gap: 6 }}>
            <span style={{ fontWeight: 600, color: "#22c55e" }}>F_prev={effectivePrevFwd.toFixed(1)}</span>
            <button onClick={() => nudgeFwd(-1)} style={fwdNudgeBtn}>{"\u2212"}</button>
            <input
              type="range"
              min={modelPrevFwd * 0.95}
              max={modelPrevFwd * 1.05}
              step={modelPrevFwd * 0.0005}
              value={effectivePrevFwd}
              onChange={(e) => handleFwdChange(parseFloat(e.target.value))}
              style={{ width: 80, accentColor: "#22c55e", height: 3, margin: 0 }}
            />
            <button onClick={() => nudgeFwd(1)} style={fwdNudgeBtn}>+</button>
            {fwdOverride != null && (
              <button onClick={handleFwdReset} style={{ ...fwdNudgeBtn, fontSize: 8, padding: "0 4px" }}>rst</button>
            )}
          </div>
        </>
      )}

      {/* Distribution triple view */}
      {viewMode === "distributions" && (
        <DistributionTripleView
          prior={priorView}
          current={currentView}
          priorLabel={`Initial (${smileModel.toUpperCase()})`}
          currentLabel={`Calibrated (${smileModel.toUpperCase()})`}
          currentColor="#22c55e"
          height={165}
        />
      )}

      {/* Model parameter sliders */}
      {lqdParams && smileModel === "lqd" && (
        <div style={{ padding: "0 16px", borderTop: "1px solid #334155", marginTop: 4, paddingTop: 8 }}>
          <div style={{ fontSize: 11, color: "#94a3b8", marginBottom: 4 }}>
            LQD Parameters ({"\u03B8"})
          </div>
          <LqdSliders
            values={lqdParams}
            baseValues={baseLqdParams ?? undefined}
            onChange={handleLqdChange}
            onReset={handleLqdReset}
          />
        </div>
      )}
      {sviParams && smileModel === "svi" && (
        <div style={{ padding: "0 16px", borderTop: "1px solid #334155", marginTop: 4, paddingTop: 8 }}>
          <div style={{ fontSize: 11, color: "#94a3b8", marginBottom: 4 }}>
            SVI-JW Parameters
          </div>
          <SviSliders
            values={sviParams}
            baseValues={baseSviParams ?? undefined}
            onChange={handleSviChange}
            onReset={handleSviReset}
          />
        </div>
      )}
      {sigmoidParams && smileModel === "sigmoid" && (
        <div style={{ padding: "0 16px", borderTop: "1px solid #334155", marginTop: 4, paddingTop: 8 }}>
          <div style={{ fontSize: 11, color: "#94a3b8", marginBottom: 4 }}>
            Sigmoid Parameters
          </div>
          <SigmoidSliders
            values={sigmoidParams}
            baseValues={baseSigmoidParams ?? undefined}
            onChange={handleSigmoidChange}
            onReset={handleSigmoidReset}
          />
        </div>
      )}
    </div>
  );
}

const fwdNudgeBtn: React.CSSProperties = {
  width: 16,
  height: 16,
  padding: 0,
  fontSize: 11,
  lineHeight: "14px",
  textAlign: "center",
  background: "#1e293b",
  color: "#94a3b8",
  border: "1px solid #334155",
  borderRadius: 3,
  cursor: "pointer",
  flexShrink: 0,
};

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
