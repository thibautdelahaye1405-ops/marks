import { useState, useEffect } from "react";
import { api } from "../api/client";
import { useEngine } from "../hooks/useEngine";
import DistributionTripleView from "./DistributionTripleView";
import SmileView from "./SmileView";
import type { NodeDistributionResponse, SmileData, QuoteSnapshot } from "../types";

type ViewMode = "smile" | "distributions";

interface Props {
  ticker: string;
  smileData: SmileData | null;
  quoteData: QuoteSnapshot | null;
}

export default function ObservedNodeView({ ticker, smileData, quoteData }: Props) {
  const {
    excludedQuotes, resetExclusions,
    addedQuotes, resetAdditions,
  } = useEngine();
  const [distData, setDistData] = useState<NodeDistributionResponse | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>("smile");
  const excluded = excludedQuotes[ticker] ?? [];
  const added = addedQuotes[ticker] ?? [];

  useEffect(() => {
    if (!ticker) return;
    api.getNodeDistribution(ticker).then(setDistData).catch(() => setDistData(null));
  }, [ticker]);

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
        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 6 }}>
          {excluded.length > 0 && (
            <button onClick={() => resetExclusions(ticker)} style={actionBtn}>
              Reset exclusions ({excluded.length})
            </button>
          )}
          {added.length > 0 && (
            <button onClick={() => resetAdditions(ticker)} style={actionBtn}>
              Forget additions ({added.length})
            </button>
          )}
          {distData && (
            <span style={{ fontSize: 10, color: "#64748b" }}>
              W2: {distData.wasserstein_dist.toFixed(4)}
            </span>
          )}
        </div>
      </div>

      {viewMode === "smile" && (
        <SmileView smileData={smileData} quoteData={quoteData} ticker={ticker} />
      )}

      {viewMode === "distributions" && distData && (
        <div style={{ padding: "4px 12px" }}>
          <DistributionTripleView
            prior={distData.prior}
            current={distData.marked}
            priorLabel="Prior"
            currentLabel="Marked"
            currentColor="#3b82f6"
            height={165}
          />
        </div>
      )}

      {viewMode === "distributions" && !distData && (
        <div style={{ padding: 20, color: "#94a3b8", textAlign: "center" }}>
          Loading distributions for {ticker}...
        </div>
      )}
    </div>
  );
}

const actionBtn: React.CSSProperties = {
  fontSize: 11,
  color: "#94a3b8",
  background: "#1e293b",
  border: "1px solid #334155",
  borderRadius: 4,
  padding: "3px 10px",
  cursor: "pointer",
};
