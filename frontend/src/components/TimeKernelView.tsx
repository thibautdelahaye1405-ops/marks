import { useState, useEffect, useCallback } from "react";
import { useEngine } from "../hooks/useEngine";
import { api } from "../api/client";

interface TimeKernel {
  K: number[][];
  expiries: string[];
  T_values: number[];
  labels: string[];
  lambda_T: number;
}

export default function TimeKernelView() {
  const { lambdaT } = useEngine();
  const [kernel, setKernel] = useState<TimeKernel | null>(null);

  const fetchKernel = useCallback(() => {
    api.getTimeKernel(lambdaT).then(setKernel).catch(() => setKernel(null));
  }, [lambdaT]);

  useEffect(() => {
    fetchKernel();
  }, [fetchKernel]);

  if (!kernel || kernel.K.length <= 1) {
    return (
      <div style={{ padding: 20, color: "#64748b", textAlign: "center", fontSize: 12 }}>
        Time kernel requires multiple maturities.
        <br />
        Select expiries in the Referential panel first.
      </div>
    );
  }

  const { K, labels } = kernel;
  const M = K.length;

  // Color scale: 0 = dark, 1 = bright
  const cellColor = (v: number) => {
    const intensity = Math.min(v, 1);
    const r = Math.round(30 + intensity * 70);
    const g = Math.round(41 + intensity * 150);
    const b = Math.round(59 + intensity * 200);
    return `rgb(${r}, ${g}, ${b})`;
  };

  return (
    <div style={{ padding: 12 }}>
      <div style={{ fontSize: 12, fontWeight: 600, color: "#94a3b8", marginBottom: 8 }}>
        Cross-Maturity Influence (K_time)
        <span style={{ fontWeight: 400, fontSize: 10, color: "#64748b", marginLeft: 8 }}>
          {"\u03BB"}_T = {kernel.lambda_T.toFixed(1)}
        </span>
      </div>

      <div style={{ overflowX: "auto" }}>
        <table style={{ borderCollapse: "collapse", fontSize: 10 }}>
          <thead>
            <tr>
              <th style={headerCell}></th>
              {labels.map((l, j) => (
                <th key={j} style={headerCell}>{l}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {K.map((row, i) => (
              <tr key={i}>
                <td style={{ ...headerCell, textAlign: "right" }}>{labels[i]}</td>
                {row.map((v, j) => (
                  <td
                    key={j}
                    style={{
                      ...dataCell,
                      background: i === j ? "#0f172a" : cellColor(v),
                      color: v > 0.3 ? "#e2e8f0" : "#64748b",
                      fontWeight: v > 0.2 ? 600 : 400,
                    }}
                    title={`K[${labels[i]}, ${labels[j]}] = ${v.toFixed(4)}`}
                  >
                    {i === j ? "-" : v.toFixed(2)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div style={{ fontSize: 9, color: "#475569", marginTop: 6 }}>
        K[i,j] = exp(-{"\u03BB"}_T |{"\u221A"}T_i - {"\u221A"}T_j|), row-normalized.
        Higher values = stronger cross-maturity influence.
      </div>
    </div>
  );
}

const headerCell: React.CSSProperties = {
  padding: "4px 8px",
  color: "#94a3b8",
  fontWeight: 600,
  borderBottom: "1px solid #334155",
  whiteSpace: "nowrap",
};

const dataCell: React.CSSProperties = {
  padding: "4px 8px",
  textAlign: "center",
  border: "1px solid #1e293b",
  minWidth: 40,
};
