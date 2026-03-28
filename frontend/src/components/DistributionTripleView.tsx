import Plot from "./Plot";
import type { DistributionView } from "../types";

interface Props {
  prior: DistributionView;
  current?: DistributionView | null;
  priorLabel?: string;
  currentLabel?: string;
  priorColor?: string;
  currentColor?: string;
  height?: number;
}

export default function DistributionTripleView({
  prior,
  current,
  priorLabel = "Prior",
  currentLabel = "Marked",
  priorColor = "#64748b",
  currentColor = "#3b82f6",
  height = 200,
}: Props) {
  const chartLayout = (
    title: string, xTitle: string, yTitle: string, h: number,
    legendPos?: { x: number; xanchor?: string },
    xRange?: [number, number],
  ) => ({
    title: { text: title, font: { color: "#e2e8f0", size: 11 } },
    paper_bgcolor: "#1e293b",
    plot_bgcolor: "#1e293b",
    font: { color: "#94a3b8", size: 9 },
    xaxis: {
      title: xTitle, gridcolor: "#334155", zerolinecolor: "#475569",
      ...(xRange ? { range: xRange } : {}),
    },
    yaxis: { title: yTitle, gridcolor: "#334155", zerolinecolor: "#475569" },
    legend: { x: legendPos?.x ?? 0, y: 1, xanchor: legendPos?.xanchor ?? "left", bgcolor: "rgba(0,0,0,0)", font: { size: 8 } },
    margin: { t: 25, r: 10, b: 35, l: 45 },
    autosize: true,
    height: h,
  });

  const clean = (arr: (number | null)[]) => arr.map((v) => (v != null ? v : NaN));

  const halfH = Math.round(height * 0.95);

  return (
    <div style={{ display: "flex", gap: 4 }}>
      {/* Left column: IV */}
      <div style={{ flex: "1 1 50%", minWidth: 0 }}>
        <Plot
          data={[
            {
              x: prior.moneyness,
              y: clean(prior.iv_curve).map((v) => v * 100),
              mode: "lines",
              name: priorLabel,
              line: { color: priorColor, dash: "dash", width: 2 },
            },
            ...(current
              ? [
                  {
                    x: current.moneyness,
                    y: clean(current.iv_curve).map((v) => v * 100),
                    mode: "lines" as const,
                    name: currentLabel,
                    line: { color: currentColor, width: 2.5 },
                  },
                ]
              : []),
          ]}
          layout={chartLayout("Implied Volatility", "ln(K/F)", "IV (%)", halfH * 2, { x: 1, xanchor: "right" })}
          style={{ width: "100%", height: halfH * 2 }}
          config={{ displayModeBar: false }}
        />
      </div>

      {/* Right column: CDF + LQD stacked */}
      <div style={{ flex: "1 1 50%", minWidth: 0, display: "flex", flexDirection: "column", gap: 4 }}>
        <Plot
          data={[
            {
              x: prior.cdf_x,
              y: prior.cdf_y,
              mode: "lines",
              name: priorLabel,
              line: { color: priorColor, dash: "dash", width: 2 },
            },
            ...(current
              ? [
                  {
                    x: current.cdf_x,
                    y: current.cdf_y,
                    mode: "lines" as const,
                    name: currentLabel,
                    line: { color: currentColor, width: 2.5 },
                  },
                ]
              : []),
          ]}
          layout={chartLayout("Risk-Neutral CDF", "x / \u03C3", "F(x/\u03C3)", halfH, undefined, [-4, 4])}
          style={{ width: "100%", height: halfH }}
          config={{ displayModeBar: false }}
        />

        <Plot
          data={[
            {
              x: prior.lqd_u,
              y: clean(prior.lqd_psi),
              mode: "lines",
              name: priorLabel,
              line: { color: priorColor, dash: "dash", width: 2 },
            },
            ...(current
              ? [
                  {
                    x: current.lqd_u,
                    y: clean(current.lqd_psi),
                    mode: "lines" as const,
                    name: currentLabel,
                    line: { color: currentColor, width: 2.5 },
                  },
                ]
              : []),
          ]}
          layout={chartLayout("Log-Quantile-Density", "u", "\u03C8(u)", halfH, { x: 0.5, xanchor: "center" }, [0, 1])}
          style={{ width: "100%", height: halfH }}
          config={{ displayModeBar: false }}
        />
      </div>
    </div>
  );
}
