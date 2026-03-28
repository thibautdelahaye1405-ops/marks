import { useState, useEffect } from "react";
import { api } from "../api/client";
import { useEngine } from "../hooks/useEngine";
import DistributionTripleView from "./DistributionTripleView";
import Plot from "./Plot";
import type { NodeDistributionResponse } from "../types";

type ViewMode = "smile" | "distributions";

interface Props {
  ticker: string;
}

export default function InferredNodeView({ ticker }: Props) {
  const { quotes, solveResult } = useEngine();
  const [data, setData] = useState<NodeDistributionResponse | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>("smile");

  useEffect(() => {
    if (!ticker) return;
    api.getNodeDistribution(ticker).then(setData).catch(() => setData(null));
  }, [ticker]);

  if (!data) {
    return (
      <div style={{ padding: 20, color: "#94a3b8", textAlign: "center" }}>
        Loading distribution for {ticker}...
      </div>
    );
  }

  const quoteData = quotes[ticker];
  const smileData = solveResult?.nodes[ticker];

  return (
    <div style={{ overflow: "auto" }}>
      {/* View toggle */}
      <div style={{ display: "flex", gap: 4, padding: "8px 12px 4px" }}>
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
              fontWeight: viewMode === m ? 700 : 500,
            }}
          >
            {m === "smile" ? "Smile (IV vs Strike)" : "IV / CDF / LQD"}
          </button>
        ))}
        <span style={{ fontSize: 10, color: "#64748b", marginLeft: "auto", alignSelf: "center" }}>
          {ticker} (inferred) | W2: {data.wasserstein_dist.toFixed(4)}
        </span>
      </div>

      {/* Smile view with non-clickable market quotes as indicators */}
      {viewMode === "smile" && (
        <Plot
          data={[
            // Prior smile (dashed)
            ...(smileData
              ? [
                  {
                    x: smileData.strikes,
                    y: smileData.iv_prior
                      .map((v) => (v != null ? v * 100 : NaN)),
                    mode: "lines" as const,
                    name: "Prior",
                    line: { color: "#64748b", dash: "dash" as const, width: 2 },
                  },
                ]
              : []),
            // Propagated marked smile
            ...(smileData
              ? [
                  {
                    x: smileData.strikes,
                    y: smileData.iv_marked
                      .map((v) => (v != null ? v * 100 : NaN)),
                    mode: "lines" as const,
                    name: "Propagated",
                    line: { color: "#f97316", width: 2.5 },
                  },
                ]
              : []),
            // Current market quotes as read-only indicators
            ...(quoteData
              ? [
                  {
                    x: quoteData.strikes,
                    y: quoteData.mid_ivs.map((v) => v * 100),
                    mode: "markers" as const,
                    name: "Market (indicative)",
                    marker: {
                      color: "rgba(148,163,184,0.5)",
                      size: 5,
                      symbol: "circle-open" as const,
                      line: { width: 1, color: "#94a3b8" },
                    },
                  },
                ]
              : []),
          ]}
          layout={{
            title: {
              text: `${ticker} -- Inferred Smile`,
              font: { color: "#e2e8f0", size: 13 },
            },
            paper_bgcolor: "#1e293b",
            plot_bgcolor: "#1e293b",
            font: { color: "#94a3b8" },
            shapes: quoteData ? [
              // Model forward line
              {
                type: "line", x0: quoteData.forward, x1: quoteData.forward,
                y0: 0, y1: 1, yref: "paper",
                line: { color: "#f59e0b", width: 1, dash: "dot" },
              },
              // Parity forward (reference)
              ...(quoteData.forward_parity != null ? [{
                type: "line" as const, x0: quoteData.forward_parity, x1: quoteData.forward_parity,
                y0: 0, y1: 1, yref: "paper" as const,
                line: { color: "#94a3b8", width: 1, dash: "dash" as const },
              }] : []),
            ] : [],
            annotations: quoteData ? [
              {
                x: quoteData.forward, y: 1.02, yref: "paper",
                text: `F=${quoteData.forward.toFixed(1)}`,
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
            margin: { t: 35, r: 15, b: 45, l: 55 },
            autosize: true,
            height: 380,
          }}
          style={{ width: "100%", height: 380 }}
          config={{ displayModeBar: false }}
        />
      )}

      {/* Distribution triple view */}
      {viewMode === "distributions" && (
        <DistributionTripleView
          prior={data.prior}
          current={data.marked}
          priorLabel="Prior"
          currentLabel="Propagated"
          currentColor="#f97316"
          height={165}
        />
      )}

      <div style={{ fontSize: 10, color: "#475569", marginTop: 8, textAlign: "center" }}>
        Inferred from graph neighbours. Market quotes shown as indicative only.
      </div>
    </div>
  );
}
