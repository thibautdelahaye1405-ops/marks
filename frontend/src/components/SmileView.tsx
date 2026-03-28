import { useState, useCallback, useEffect, useRef } from "react";
import Plot from "./Plot";
import SviSliders from "./SviSliders";
import { api } from "../api/client";
import type { SmileData, QuoteSnapshot } from "../types";
import { useEngine } from "../hooks/useEngine";

interface Props {
  smileData: SmileData | null;
  quoteData: QuoteSnapshot | null;
  ticker: string;
}

export default function SmileView({ smileData, quoteData, ticker }: Props) {
  const {
    excludedQuotes, toggleQuotePoint, resetExclusions,
    addedQuotes, addQuotePoint, removeAddedQuote, resetAdditions,
  } = useEngine();

  const added = addedQuotes[ticker] ?? [];

  // Forward adjustment state
  const [fwdOverride, setFwdOverride] = useState<number | null>(null);
  const modelFwd = quoteData?.forward_model ?? quoteData?.forward ?? 100;
  const effectiveFwd = fwdOverride ?? quoteData?.forward ?? modelFwd;
  const fwdTimerRef = useRef<ReturnType<typeof setTimeout>>();

  // Reset forward override when ticker changes
  useEffect(() => {
    setFwdOverride(null);
  }, [ticker]);

  const handleFwdChange = useCallback((val: number) => {
    setFwdOverride(val);
    // Debounce API call
    clearTimeout(fwdTimerRef.current);
    fwdTimerRef.current = setTimeout(() => {
      api.setForwardOverride(ticker, val).then(() => {
        useEngine.getState().fit();
      }).catch(() => {});
    }, 150);
  }, [ticker]);

  const handleFwdReset = useCallback(() => {
    setFwdOverride(null);
    api.setForwardOverride(ticker, null).then(() => {
      useEngine.getState().fit();
    }).catch(() => {});
  }, [ticker]);

  const fwdRef = useRef(effectiveFwd);
  fwdRef.current = effectiveFwd;
  const nudgeFwd = useCallback((dir: number) => {
    const step = modelFwd * 0.001; // 0.1% of forward
    handleFwdChange(fwdRef.current + dir * step);
  }, [modelFwd, handleFwdChange]);

  // SVI parameter override state
  const [sviParams, setSviParams] = useState<{ a: number; b: number; rho: number; m: number; sigma: number } | null>(null);
  const [baseSviParams, setBaseSviParams] = useState<{ a: number; b: number; rho: number; m: number; sigma: number } | null>(null);
  const [overriddenSmile, setOverriddenSmile] = useState<SmileData | null>(null);

  // Extract SVI params from solve result
  useEffect(() => {
    if (smileData?.beta && smileData.beta.length >= 5 && smileData.is_observed) {
      const p = { a: smileData.beta[0], b: smileData.beta[1], rho: smileData.beta[2], m: smileData.beta[3], sigma: smileData.beta[4] };
      setSviParams(p);
      // Only update base when forward is NOT overridden — so slider shows delta
      if (fwdOverride == null) {
        setBaseSviParams(p);
      }
      setOverriddenSmile(null);
    }
  }, [smileData, fwdOverride]);

  const handleSviChange = useCallback(
    (params: { a: number; b: number; rho: number; m: number; sigma: number }) => {
      if (!ticker) return;
      setSviParams(params);
      api.sviOverrideSmile(ticker, params).then((result) => {
        setOverriddenSmile(result);
      }).catch(() => {});
    },
    [ticker]
  );

  const handleSviReset = useCallback(() => {
    if (!baseSviParams) return;
    setSviParams(baseSviParams);
    setOverriddenSmile(null);
  }, [baseSviParams]);

  // Use overridden smile data if available
  const displaySmile = overriddenSmile ?? smileData;

  const handleDoubleClick = useCallback(
    (coords: { x: number; y: number }) => {
      if (!ticker) return;
      // x = strike, y = IV in % → convert to decimal
      const iv = coords.y / 100;
      if (iv > 0.005 && iv < 5.0 && coords.x > 0) {
        addQuotePoint(ticker, coords.x, iv);
      }
    },
    [ticker, addQuotePoint]
  );

  if (!ticker) {
    return (
      <div style={{ padding: 20, color: "#94a3b8", textAlign: "center" }}>
        Select a node in the graph to view its volatility smile
      </div>
    );
  }

  if (!displaySmile && !quoteData) {
    return (
      <div style={{ padding: 20, color: "#94a3b8", textAlign: "center" }}>
        <div style={{ fontSize: 16, marginBottom: 8 }}>{ticker}</div>
        No data yet. Click <b>Fetch Priors</b> then <b>Propagate</b>.
      </div>
    );
  }

  const excluded = new Set(excludedQuotes[ticker] ?? []);
  const traces: Plotly.Data[] = [];

  // Market quotes — split into included and excluded
  if (quoteData) {
    const incStrikes: number[] = [];
    const incIvs: number[] = [];
    const incSpreads: number[] = [];
    const excStrikes: number[] = [];
    const excIvs: number[] = [];

    quoteData.strikes.forEach((k, i) => {
      if (excluded.has(i)) {
        excStrikes.push(k);
        excIvs.push(quoteData.mid_ivs[i] * 100);
      } else {
        incStrikes.push(k);
        incIvs.push(quoteData.mid_ivs[i] * 100);
        incSpreads.push(quoteData.bid_ask_spread[i] * 100);
      }
    });

    // Included quotes — clickable to exclude
    traces.push({
      x: incStrikes,
      y: incIvs,
      mode: "markers",
      name: "Market",
      marker: { color: "#22c55e", size: 7, symbol: "circle" },
      error_y: {
        type: "data",
        array: incSpreads,
        visible: true,
        color: "#22c55e",
        thickness: 1,
      },
      customdata: quoteData.strikes.map((_, i) => i).filter((i) => !excluded.has(i)),
    } as Plotly.Data);

    // Excluded quotes
    if (excStrikes.length > 0) {
      traces.push({
        x: excStrikes,
        y: excIvs,
        mode: "markers",
        name: "Excluded",
        marker: { color: "#64748b", size: 6, symbol: "x", opacity: 0.5 },
        customdata: quoteData.strikes.map((_, i) => i).filter((i) => excluded.has(i)),
      } as Plotly.Data);
    }
  }

  // User-added synthetic quotes (clickable to remove)
  if (added.length > 0) {
    traces.push({
      x: added.map((p) => p[0]),
      y: added.map((p) => p[1] * 100),
      mode: "markers",
      name: "Added",
      marker: { color: "#f59e0b", size: 9, symbol: "star", line: { width: 1, color: "#fff" } },
      customdata: added.map((_, i) => `added:${i}`),
      hovertemplate: "K=%{x:.0f} IV=%{y:.1f}%<br><i>Click to remove</i><extra></extra>",
    } as Plotly.Data);
  }

  // Prior + marked curves
  if (displaySmile) {
    traces.push({
      x: displaySmile.strikes,
      y: displaySmile.iv_prior.map((v) => (v != null ? v * 100 : null)) as number[],
      mode: "lines",
      name: "Prior",
      line: { color: "#64748b", dash: "dash", width: 2 },
    });

    traces.push({
      x: displaySmile.strikes,
      y: displaySmile.iv_marked.map((v) => (v != null ? v * 100 : null)) as number[],
      mode: "lines",
      name: "Marked",
      line: { color: displaySmile.is_observed ? "#3b82f6" : "#f97316", width: 3 },
    });
  }

  const subtitle = displaySmile
    ? displaySmile.is_observed
      ? "observed"
      : "propagated"
    : "pre-solve";

  const handleClick = (event: any) => {
    if (!event?.points?.[0]) return;
    const pt = event.points[0];
    const idx = pt.customdata;
    if (typeof idx === "string" && idx.startsWith("added:")) {
      removeAddedQuote(ticker, parseInt(idx.slice(6), 10));
    } else if (typeof idx === "number") {
      toggleQuotePoint(ticker, idx);
    }
  };

  return (
    <div>
      <Plot
        data={traces}
        layout={{
          title: {
            text: `${ticker} -- IV Smile (${subtitle})`,
            font: { color: "#e2e8f0", size: 13 },
          },
          paper_bgcolor: "#1e293b",
          plot_bgcolor: "#1e293b",
          font: { color: "#94a3b8" },
          shapes: quoteData ? [
            // Active forward line (follows slider)
            {
              type: "line", x0: effectiveFwd, x1: effectiveFwd,
              y0: 0, y1: 1, yref: "paper",
              line: { color: "#f59e0b", width: 1, dash: "dot" },
            },
            // Parity forward (reference, if available)
            ...(quoteData.forward_parity != null ? [{
              type: "line" as const, x0: quoteData.forward_parity, x1: quoteData.forward_parity,
              y0: 0, y1: 1, yref: "paper" as const,
              line: { color: "#94a3b8", width: 1, dash: "dash" as const },
            }] : []),
          ] : [],
          annotations: quoteData ? [
            {
              x: effectiveFwd, y: 1.02, yref: "paper",
              text: `F=${effectiveFwd.toFixed(1)}`,
              showarrow: false, font: { color: "#f59e0b", size: 9 },
            },
            ...(quoteData.forward_parity != null ? [{
              x: quoteData.forward_parity, y: 0.96, yref: "paper" as const,
              text: `F(parity)=${quoteData.forward_parity.toFixed(1)}`,
              showarrow: false, font: { color: "#94a3b8", size: 8 },
            }] : []),
          ] : [],
          xaxis: { title: "Strike", gridcolor: "#334155", zerolinecolor: "#334155" },
          yaxis: { title: "Implied Vol (%)", gridcolor: "#334155", zerolinecolor: "#334155" },
          legend: { x: 1, y: 1, xanchor: "right", bgcolor: "rgba(0,0,0,0)", font: { size: 9 } },
          margin: { t: 35, r: 20, b: 45, l: 55 },
          autosize: true,
          height: 440,
        }}
        style={{ width: "100%", height: 440 }}
        config={{ displayModeBar: false, doubleClick: false }}
        onClick={handleClick}
        onDoubleClick={handleDoubleClick}
      />
      {quoteData && (
        <div style={{ padding: "2px 16px", fontSize: 10, color: "#94a3b8", display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
          <span style={{ fontWeight: 600, color: "#f59e0b" }}>F={effectiveFwd.toFixed(1)}</span>
          <button onClick={() => nudgeFwd(-1)} style={fwdNudgeBtn}>{"\u2212"}</button>
          <input
            type="range"
            min={modelFwd * 0.95}
            max={modelFwd * 1.05}
            step={modelFwd * 0.0005}
            value={effectiveFwd}
            onChange={(e) => handleFwdChange(parseFloat(e.target.value))}
            style={{ width: 80, accentColor: "#f59e0b", height: 3, margin: 0 }}
          />
          <button onClick={() => nudgeFwd(1)} style={fwdNudgeBtn}>+</button>
          {fwdOverride != null && (
            <button onClick={handleFwdReset} style={{ ...fwdNudgeBtn, fontSize: 8, padding: "0 4px" }}>rst</button>
          )}
          <span style={{ color: "#334155" }}>|</span>
          {quoteData.rate_used != null && <span>r={(quoteData.rate_used * 100).toFixed(2)}%</span>}
          {quoteData.div_yield_used != null && quoteData.div_yield_used > 0 && <span>q={(quoteData.div_yield_used * 100).toFixed(2)}%</span>}
          {quoteData.repo_rate_used != null && quoteData.repo_rate_used > 0 && <span>repo={(quoteData.repo_rate_used * 100).toFixed(2)}%</span>}
          {quoteData.forward_parity != null && quoteData.forward_model != null && (() => {
            const diff = Math.abs(quoteData.forward_parity! - quoteData.forward_model!) / quoteData.forward_model! * 100;
            const color = diff > 0.5 ? "#ef4444" : "#22c55e";
            return <><span style={{ color: "#334155" }}>|</span><span style={{ color }}>parity vs model: {diff.toFixed(2)}%</span></>;
          })()}
        </div>
      )}
      {/* SVI parameter sliders for observed nodes */}
      {sviParams && displaySmile?.is_observed && (
        <div style={{ padding: "0 16px", borderTop: "1px solid #334155", marginTop: 4, paddingTop: 8 }}>
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

const actionBtn: React.CSSProperties = {
  fontSize: 11,
  color: "#94a3b8",
  background: "#1e293b",
  border: "1px solid #334155",
  borderRadius: 4,
  padding: "3px 10px",
  cursor: "pointer",
};
