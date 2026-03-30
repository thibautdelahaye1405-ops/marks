import { useState, useEffect, useCallback } from "react";
import { api } from "../api/client";
import { useEngine } from "../hooks/useEngine";
import DistributionTripleView from "./DistributionTripleView";
import SmileView from "./SmileView";
import SviSliders from "./SviSliders";
import LqdSliders from "./LqdSliders";
import type { LqdTraderParams } from "./LqdSliders";
import SigmoidSliders from "./SigmoidSliders";
import type { SigmoidTraderParams } from "./SigmoidSliders";
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
    smileModel,
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
            priorLabel={`Prior (${smileModel.toUpperCase()})`}
            currentLabel={`Marked (${smileModel.toUpperCase()})`}
            currentColor="#3b82f6"
            height={165}
          />
          {/* Model-aware sliders below distributions */}
          {smileData?.is_observed && smileData.beta && smileModel === "lqd" && smileData.beta.length >= 6 && (
            <div style={{ borderTop: "1px solid #334155", marginTop: 8, paddingTop: 8 }}>
              <div style={{ fontSize: 11, color: "#94a3b8", marginBottom: 4 }}>LQD Parameters ({"\u03B8"})</div>
              <LqdSliders
                values={{ min_iv: smileData.beta[0], atm_skew: smileData.beta[1], atm_curv: smileData.beta[2], put_slope: smileData.beta[3], call_slope: smileData.beta[4], shoulder: smileData.beta[5] }}
                onChange={(params) => {
                  const theta = [params.min_iv, params.atm_skew, params.atm_curv, params.put_slope, params.call_slope, params.shoulder];
                  api.lqdOverrideSmile(ticker, theta).catch(() => {});
                }}
              />
            </div>
          )}
          {smileData?.is_observed && smileData.beta && smileModel === "sigmoid" && smileData.beta.length >= 6 && (
            <div style={{ borderTop: "1px solid #334155", marginTop: 8, paddingTop: 8 }}>
              <div style={{ fontSize: 11, color: "#94a3b8", marginBottom: 4 }}>Sigmoid Parameters</div>
              <SigmoidSliders
                values={{ sigma_atm: smileData.beta[0], s_atm: smileData.beta[1], k_atm: smileData.beta[2], w_p: smileData.beta[3], w_c: smileData.beta[4], sigma_min: smileData.beta[5] }}
                onChange={(params) => {
                  const p = [params.sigma_atm, params.s_atm, params.k_atm, params.w_p, params.w_c, params.sigma_min];
                  api.sigmoidOverrideSmile(ticker, p).catch(() => {});
                }}
              />
            </div>
          )}
          {smileData?.is_observed && smileData.beta && smileModel === "svi" && smileData.beta.length >= 5 && (
            <div style={{ borderTop: "1px solid #334155", marginTop: 8, paddingTop: 8 }}>
              <div style={{ fontSize: 11, color: "#94a3b8", marginBottom: 4 }}>SVI-JW Parameters</div>
              <SviSliders
                values={{ v: smileData.beta[0], psi_hat: smileData.beta[1], p_hat: smileData.beta[2], c_hat: smileData.beta[3], vt_ratio: smileData.beta[4] }}
                onChange={(params) => {
                  api.sviOverrideSmile(ticker, params).catch(() => {});
                }}
              />
            </div>
          )}
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
