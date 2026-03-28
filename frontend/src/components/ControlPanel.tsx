import { useEngine } from "../hooks/useEngine";

interface ControlPanelProps {
  onToggleModelling: () => void;
  onToggleReferential: () => void;
  onFetchPriors: () => void;
}

export default function ControlPanel({
  onToggleModelling,
  onToggleReferential,
  onFetchPriors,
}: ControlPanelProps) {
  const {
    loading,
    error,
    quotes,
    priorsCalibrated,
    solveResult,
    fetchSnapshot,
  } = useEngine();

  const hasQuotes = Object.keys(quotes).length > 0;

  return (
    <div style={barStyle}>
      {/* Buttons */}
      <button onClick={onFetchPriors} disabled={loading} style={btnStyle}>
        {loading ? "..." : "Fetch Priors"}
      </button>
      <button
        onClick={fetchSnapshot}
        disabled={loading || !priorsCalibrated}
        style={{
          ...btnStyle,
          background: priorsCalibrated ? "#0ea5e9" : "#475569",
        }}
      >
        Fetch Snapshot
      </button>

      <div style={divider} />

      <button onClick={onToggleModelling} style={panelBtn}>
        Modelling
      </button>
      <button onClick={onToggleReferential} style={panelBtn}>
        Referential
      </button>

      {/* Status */}
      <div style={{ fontSize: 11, color: "#64748b", marginLeft: "auto", whiteSpace: "nowrap" }}>
        {hasQuotes ? `${Object.keys(quotes).length} assets` : ""}
        {priorsCalibrated ? " | priors ok" : ""}
        {solveResult ? ` | ${solveResult.tickers.length} propagated` : ""}
      </div>

      {error && (
        <div style={{ color: "#f87171", fontSize: 11, whiteSpace: "nowrap" }}>
          {error.slice(0, 60)}
        </div>
      )}
    </div>
  );
}

const barStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 8,
  padding: "6px 16px",
  borderBottom: "1px solid #1e293b",
  flexWrap: "nowrap",
  overflow: "hidden",
};

const btnStyle: React.CSSProperties = {
  padding: "5px 12px",
  background: "#6366f1",
  color: "#fff",
  border: "none",
  borderRadius: 5,
  cursor: "pointer",
  fontSize: 12,
  fontWeight: 600,
  whiteSpace: "nowrap",
  flexShrink: 0,
};

const divider: React.CSSProperties = {
  width: 1,
  height: 24,
  background: "#334155",
  flexShrink: 0,
};

const panelBtn: React.CSSProperties = {
  padding: "5px 12px",
  background: "#1e293b",
  color: "#cbd5e1",
  border: "1px solid #334155",
  borderRadius: 5,
  cursor: "pointer",
  fontSize: 12,
  fontWeight: 600,
  whiteSpace: "nowrap",
  flexShrink: 0,
};
