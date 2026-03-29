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
    fetchSnapshot,
    calibrateAllPriors,
    fitAllSnapshots,
  } = useEngine();

  const hasQuotes = Object.keys(quotes).length > 0;

  return (
    <div style={barStyle}>
      {/* Fetch buttons */}
      <button onClick={onFetchPriors} disabled={loading} style={btnStyle}>
        {loading ? "..." : "Fetch Priors"}
      </button>
      <button
        onClick={fetchSnapshot}
        disabled={loading || !hasQuotes}
        style={{
          ...btnStyle,
          background: hasQuotes ? "#0ea5e9" : "#475569",
        }}
      >
        Fetch Snapshot
      </button>

      <div style={divider} />

      {/* Calibrate / Fit buttons */}
      <button
        onClick={calibrateAllPriors}
        disabled={loading || !hasQuotes}
        style={{
          ...btnStyle,
          background: hasQuotes ? "#22c55e" : "#475569",
        }}
      >
        Calibrate Priors
      </button>
      <button
        onClick={fitAllSnapshots}
        disabled={loading || !hasQuotes || !priorsCalibrated}
        style={{
          ...btnStyle,
          background: hasQuotes && priorsCalibrated ? "#0ea5e9" : "#475569",
        }}
      >
        Fit Snapshot
      </button>

      <div style={divider} />

      <button onClick={onToggleModelling} style={panelBtn}>
        Modelling
      </button>
      <button onClick={onToggleReferential} style={panelBtn}>
        Referential
      </button>

      {error && (
        <div style={{ color: "#f87171", fontSize: 11, whiteSpace: "nowrap", marginLeft: "auto" }}>
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
