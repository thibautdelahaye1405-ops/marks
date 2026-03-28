import { useState, useEffect } from "react";
import Plot from "./Plot";
import { api } from "../api/client";
import type { SolveResponse, NodeDistributionResponse } from "../types";

type ViewMode = "iv" | "cdf" | "lqd";

interface Props {
  solveResult: SolveResponse | null;
}

const COLORS = [
  "#3b82f6", "#22c55e", "#f59e0b", "#ef4444", "#8b5cf6",
  "#06b6d4", "#ec4899", "#14b8a6", "#f97316", "#6366f1",
];

export default function SurfaceView({ solveResult }: Props) {
  const [viewMode, setViewMode] = useState<ViewMode>("iv");
  const [distData, setDistData] = useState<Record<string, NodeDistributionResponse>>({});

  // Fetch distribution data for CDF/LQD views
  useEffect(() => {
    if (!solveResult) return;
    const tickers = solveResult.tickers;
    const missing = tickers.filter((t) => !distData[t]);
    if (missing.length === 0) return;
    Promise.all(
      missing.map((t) =>
        api.getNodeDistribution(t).then((d) => [t, d] as const).catch(() => null)
      )
    ).then((results) => {
      const update: Record<string, NodeDistributionResponse> = {};
      for (const r of results) {
        if (r) update[r[0]] = r[1];
      }
      setDistData((prev) => ({ ...prev, ...update }));
    });
  }, [solveResult]);

  if (!solveResult) {
    return (
      <div style={{ padding: 20, color: "#94a3b8", textAlign: "center" }}>
        Run Propagate to see all smiles
      </div>
    );
  }

  const tickers = solveResult.tickers;
  const priorTraces: Plotly.Data[] = [];
  const posteriorTraces: Plotly.Data[] = [];

  if (viewMode === "iv") {
    // Normalized IV: show as moneyness (log(K/F)) so all assets are comparable
    tickers.forEach((ticker, idx) => {
      const node = solveResult.nodes[ticker];
      if (!node) return;
      const color = COLORS[idx % COLORS.length];

      // Compute moneyness for x-axis: use strike index as proxy for comparable axis
      // Better: normalize strikes around ATM
      const strikes = node.strikes;
      const mid = strikes[Math.floor(strikes.length / 2)];
      const moneyness = strikes.map((k) => ((k - mid) / mid) * 100); // % moneyness

      // Prior (dotted)
      const ivPrior = node.iv_prior.map((v) => (v != null ? v * 100 : NaN));
      priorTraces.push({
        x: moneyness,
        y: ivPrior,
        mode: "lines",
        name: `${ticker} prior`,
        line: { color, width: 1.5, dash: "dot" },
        legendgroup: ticker,
        showlegend: false,
      });

      // Posterior (solid)
      const ivMarked = node.iv_marked.map((v) => (v != null ? v * 100 : NaN));
      posteriorTraces.push({
        x: moneyness,
        y: ivMarked,
        mode: "lines",
        name: `${ticker}${node.is_observed ? "" : " (inf)"}`,
        line: { color, width: 2, dash: node.is_observed ? "solid" : "dashdot" },
        legendgroup: ticker,
      });
    });
  } else {
    // CDF or LQD views from distribution data
    tickers.forEach((ticker, idx) => {
      const dd = distData[ticker];
      if (!dd) return;
      const color = COLORS[idx % COLORS.length];
      const isObs = solveResult.nodes[ticker]?.is_observed ?? true;

      if (viewMode === "cdf") {
        // Prior CDF (dotted)
        if (dd.prior.cdf_x && dd.prior.cdf_y) {
          // Normalize CDF x-axis: center on mean, scale by std
          const xs = dd.prior.cdf_x;
          const ys = dd.prior.cdf_y;
          const mean = xs[Math.floor(xs.length / 2)] || 0;
          const normX = xs.map((x) => ((x - mean) / Math.max(Math.abs(mean), 1)) * 100);

          priorTraces.push({
            x: normX,
            y: ys,
            mode: "lines",
            name: `${ticker} prior`,
            line: { color, width: 1.5, dash: "dot" },
            legendgroup: ticker,
            showlegend: false,
          });
        }

        // Posterior CDF (solid)
        if (dd.marked?.cdf_x && dd.marked?.cdf_y) {
          const xs = dd.marked.cdf_x;
          const ys = dd.marked.cdf_y;
          const mean = xs[Math.floor(xs.length / 2)] || 0;
          const normX = xs.map((x) => ((x - mean) / Math.max(Math.abs(mean), 1)) * 100);

          posteriorTraces.push({
            x: normX,
            y: ys,
            mode: "lines",
            name: `${ticker}${isObs ? "" : " (inf)"}`,
            line: { color, width: 2, dash: isObs ? "solid" : "dashdot" },
            legendgroup: ticker,
          });
        }
      } else if (viewMode === "lqd") {
        // Prior LQD (dotted)
        if (dd.prior.lqd_u && dd.prior.lqd_psi) {
          const us = dd.prior.lqd_u;
          const psis = dd.prior.lqd_psi.map((v) => (v != null ? v : NaN));

          priorTraces.push({
            x: us,
            y: psis,
            mode: "lines",
            name: `${ticker} prior`,
            line: { color, width: 1.5, dash: "dot" },
            legendgroup: ticker,
            showlegend: false,
          });
        }

        // Posterior LQD (solid)
        if (dd.marked?.lqd_u && dd.marked?.lqd_psi) {
          const us = dd.marked.lqd_u;
          const psis = dd.marked.lqd_psi.map((v) => (v != null ? v : NaN));

          posteriorTraces.push({
            x: us,
            y: psis,
            mode: "lines",
            name: `${ticker}${isObs ? "" : " (inf)"}`,
            line: { color, width: 2, dash: isObs ? "solid" : "dashdot" },
            legendgroup: ticker,
          });
        }
      }
    });
  }

  const xTitles: Record<ViewMode, string> = {
    iv: "Moneyness (%)",
    cdf: "Normalised Price (%)",
    lqd: "Quantile u",
  };
  const yTitles: Record<ViewMode, string> = {
    iv: "Implied Vol (%)",
    cdf: "CDF",
    lqd: "\u03C8(u)",
  };

  return (
    <div style={{ height: "100%", display: "flex", flexDirection: "column" }}>
      {/* Mode toggle */}
      <div style={{ display: "flex", gap: 4, padding: "8px 12px 4px" }}>
        {(["iv", "cdf", "lqd"] as ViewMode[]).map((m) => (
          <button
            key={m}
            onClick={() => setViewMode(m)}
            style={{
              fontSize: 11,
              padding: "3px 12px",
              background: viewMode === m ? "#334155" : "transparent",
              color: viewMode === m ? "#e2e8f0" : "#64748b",
              border: "1px solid #334155",
              borderRadius: 4,
              cursor: "pointer",
              fontWeight: viewMode === m ? 700 : 500,
            }}
          >
            {m.toUpperCase()}
          </button>
        ))}
        <span style={{ fontSize: 10, color: "#475569", marginLeft: 8, alignSelf: "center" }}>
          dotted = prior | solid = posterior
        </span>
      </div>

      {/* Plot */}
      <div style={{ flex: 1, minHeight: 0 }}>
        <Plot
          data={[...priorTraces, ...posteriorTraces]}
          layout={{
            paper_bgcolor: "#0f172a",
            plot_bgcolor: "#0f172a",
            font: { color: "#94a3b8", size: 11 },
            xaxis: {
              title: xTitles[viewMode],
              gridcolor: "#1e293b",
              zerolinecolor: "#334155",
            },
            yaxis: {
              title: yTitles[viewMode],
              gridcolor: "#1e293b",
              zerolinecolor: "#334155",
            },
            legend: {
              x: viewMode === "iv" ? 1 : viewMode === "lqd" ? 0.5 : 0,
              xanchor: viewMode === "iv" ? "right" : viewMode === "lqd" ? "center" : "left",
              y: 1,
              bgcolor: "rgba(0,0,0,0)",
              font: { size: 9 },
            },
            margin: { t: 10, r: 20, b: 50, l: 60 },
            autosize: true,
          }}
          useResizeHandler
          style={{ width: "100%", height: "100%" }}
          config={{ displayModeBar: false }}
        />
      </div>
    </div>
  );
}
