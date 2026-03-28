import { useState, useRef, useCallback, useEffect } from "react";
import { useEngine } from "../hooks/useEngine";
import { api } from "../api/client";

function HoldButton({ onAction, children, style }: {
  onAction: () => void;
  children: React.ReactNode;
  style?: React.CSSProperties;
}) {
  const timer = useRef<ReturnType<typeof setInterval>>();
  const actionRef = useRef(onAction);
  actionRef.current = onAction;
  const stop = useCallback(() => { clearInterval(timer.current); timer.current = undefined; }, []);
  const start = useCallback(() => {
    actionRef.current();
    stop();
    timer.current = setInterval(() => actionRef.current(), 80);
  }, [stop]);
  useEffect(() => stop, [stop]);
  return (
    <button onMouseDown={start} onMouseUp={stop} onMouseLeave={stop} style={style}>
      {children}
    </button>
  );
}

export default function ModellingOptionsPanel() {
  const { lambda, eta, setLambda, setEta } = useEngine();

  const [repoGc, setRepoGc] = useState(0.0);
  const [treasuryCurve, setTreasuryCurve] = useState<{ date: string; tenors: number[]; rates: number[] } | null>(null);

  useEffect(() => {
    api.getTreasuryCurve().then(setTreasuryCurve).catch(() => {});
    api.getRatesConfig().then((c) => setRepoGc(c.repo_rate_gc)).catch(() => {});
  }, []);

  const handleRepoChange = useCallback((val: number) => {
    setRepoGc(val);
    api.updateRatesConfig({ repo_rate_gc: val, repo_overrides: {} }).catch(() => {});
  }, []);

  const repoRef = useRef(repoGc);
  repoRef.current = repoGc;
  const nudgeRepo = useCallback((dir: number) => {
    const next = Math.max(0, Math.min(0.10, repoRef.current + dir * 0.001));
    repoRef.current = next;
    handleRepoChange(next);
  }, [handleRepoChange]);

  const logStep = 0.01;

  // Use refs so HoldButton interval always reads the latest value
  const lambdaRef = useRef(lambda);
  const etaRef = useRef(eta);
  lambdaRef.current = lambda;
  etaRef.current = eta;

  const nudgeLambda = useCallback((dir: number) => {
    const logVal = Math.log10(lambdaRef.current) + dir * logStep;
    const clamped = Math.pow(10, Math.max(-2, Math.min(1, logVal)));
    lambdaRef.current = clamped;
    useEngine.getState().setLambda(clamped);
  }, []);

  const nudgeEta = useCallback((dir: number) => {
    const logVal = Math.log10(etaRef.current) + dir * logStep;
    const clamped = Math.pow(10, Math.max(-4, Math.min(-1, logVal)));
    etaRef.current = clamped;
    useEngine.getState().setEta(clamped);
  }, []);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {/* Lambda slider */}
      <div>
        <div style={sectionTitle}>Regularisation</div>
        <label style={labelStyle}>
          <span style={labelText}>
            {"\u03BB"} {lambda.toFixed(2)}
          </span>
          <div style={sliderRow}>
            <HoldButton onAction={() => nudgeLambda(-1)} style={nudgeBtn}>
              {"\u2212"}
            </HoldButton>
            <input
              type="range"
              min={-2}
              max={1}
              step={logStep}
              value={Math.log10(lambda)}
              onChange={(e) =>
                setLambda(Math.pow(10, parseFloat(e.target.value)))
              }
              style={sliderStyle}
            />
            <HoldButton onAction={() => nudgeLambda(1)} style={nudgeBtn}>
              +
            </HoldButton>
          </div>
          <span style={rangeHint}>0.01 -- 10</span>
        </label>
      </div>

      {/* Eta slider */}
      <div>
        <label style={labelStyle}>
          <span style={labelText}>
            {"\u03B7"} {eta.toFixed(4)}
          </span>
          <div style={sliderRow}>
            <HoldButton onAction={() => nudgeEta(-1)} style={nudgeBtn}>
              {"\u2212"}
            </HoldButton>
            <input
              type="range"
              min={-4}
              max={-1}
              step={logStep}
              value={Math.log10(eta)}
              onChange={(e) => setEta(Math.pow(10, parseFloat(e.target.value)))}
              style={sliderStyle}
            />
            <HoldButton onAction={() => nudgeEta(1)} style={nudgeBtn}>
              +
            </HoldButton>
          </div>
          <span style={rangeHint}>0.0001 -- 0.1</span>
        </label>
      </div>

      {/* Forwards & Rates */}
      <div>
        <div style={sectionTitle}>Forwards & Rates</div>
        <label style={labelStyle}>
          <span style={labelText}>
            GC Repo Rate: {(repoGc * 100).toFixed(2)}%
          </span>
          <div style={sliderRow}>
            <HoldButton onAction={() => nudgeRepo(-1)} style={nudgeBtn}>
              {"\u2212"}
            </HoldButton>
            <input
              type="range"
              min={0}
              max={0.10}
              step={0.001}
              value={repoGc}
              onChange={(e) => handleRepoChange(parseFloat(e.target.value))}
              style={sliderStyle}
            />
            <HoldButton onAction={() => nudgeRepo(1)} style={nudgeBtn}>
              +
            </HoldButton>
          </div>
          <span style={rangeHint}>0% -- 10%</span>
        </label>
        {treasuryCurve && (
          <div style={{ marginTop: 8, fontSize: 10, color: "#64748b" }}>
            UST {treasuryCurve.date}: 1M={((treasuryCurve.rates[0] ?? 0) * 100).toFixed(1)}% 1Y={((treasuryCurve.rates[3] ?? 0) * 100).toFixed(1)}%
          </div>
        )}
      </div>

      {/* Smile model */}
      <div>
        <div style={sectionTitle}>Smile model</div>
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <label style={radioLabel}>
            <input
              type="radio"
              name="smileModel"
              value="svi"
              defaultChecked
              style={radioStyle}
            />
            <span style={{ color: "#e2e8f0" }}>SVI</span>
          </label>
          <label style={{ ...radioLabel, opacity: 0.4 }}>
            <input
              type="radio"
              name="smileModel"
              value="lqd"
              disabled
              style={radioStyle}
            />
            <span style={{ color: "#94a3b8" }}>LQD</span>
          </label>
          <label style={{ ...radioLabel, opacity: 0.4 }}>
            <input
              type="radio"
              name="smileModel"
              value="sigmoid"
              disabled
              style={radioStyle}
            />
            <span style={{ color: "#94a3b8" }}>Sigmoid</span>
          </label>
        </div>
      </div>

      {/* Persistence */}
      <div>
        <div style={sectionTitle}>Persistence</div>
        <span style={{ fontSize: 11, color: "#64748b", fontStyle: "italic" }}>
          Coming soon
        </span>
      </div>
    </div>
  );
}

const sectionTitle: React.CSSProperties = {
  fontSize: 11,
  fontWeight: 700,
  color: "#94a3b8",
  textTransform: "uppercase",
  letterSpacing: 0.5,
  marginBottom: 8,
};

const labelStyle: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: 4,
};

const labelText: React.CSSProperties = {
  fontSize: 12,
  color: "#cbd5e1",
  fontWeight: 600,
};

const sliderRow: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 6,
};

const sliderStyle: React.CSSProperties = {
  flex: 1,
  accentColor: "#6366f1",
};

const rangeHint: React.CSSProperties = {
  fontSize: 10,
  color: "#64748b",
};

const nudgeBtn: React.CSSProperties = {
  width: 22,
  height: 22,
  padding: 0,
  fontSize: 14,
  lineHeight: "20px",
  textAlign: "center",
  background: "#1e293b",
  color: "#94a3b8",
  border: "1px solid #334155",
  borderRadius: 4,
  cursor: "pointer",
  flexShrink: 0,
};

const radioLabel: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 8,
  fontSize: 12,
  cursor: "pointer",
};

const radioStyle: React.CSSProperties = {
  accentColor: "#6366f1",
};
