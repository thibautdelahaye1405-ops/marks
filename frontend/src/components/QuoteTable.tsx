import { useEngine } from "../hooks/useEngine";
import { observedColor, inferredColor } from "../utils/colors";

export default function QuoteTable() {
  const {
    universe,
    quotes,
    selectedNode,
    observedTickers,
    loading,
    propagate,
    setSelectedNode,
    toggleObserved,
  } = useEngine();

  const hasQuotes = Object.keys(quotes).length > 0;

  if (universe.length === 0) return null;

  const allTickers = universe.map((a) => a.ticker);
  const allObserved = allTickers.every((t) => observedTickers.includes(t));
  const noneObserved = observedTickers.length === 0;

  const handleSelectAll = () => {
    const store = useEngine.getState();
    // Set all tickers as observed
    useEngine.setState({ observedTickers: allTickers });
  };

  const handleUnselectAll = () => {
    useEngine.setState({ observedTickers: [] });
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      {/* Propagate button */}
      <div style={{ padding: "8px 10px 0" }}>
        <button
          onClick={propagate}
          disabled={loading || !hasQuotes}
          style={{
            width: "100%",
            padding: "6px 0",
            background: hasQuotes && !loading ? "#3b82f6" : "#475569",
            color: "#fff",
            border: "none",
            borderRadius: 5,
            cursor: hasQuotes && !loading ? "pointer" : "not-allowed",
            fontSize: 12,
            fontWeight: 600,
          }}
        >
          {loading ? "..." : "Propagate"}
        </button>
      </div>

      {/* Header with select/unselect all */}
      <div
        style={{
          padding: "8px 10px 4px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <span style={{ fontSize: 12, fontWeight: 600, color: "#94a3b8" }}>
          Assets
        </span>
        <div style={{ display: "flex", gap: 4 }}>
          <button
            onClick={handleSelectAll}
            disabled={allObserved}
            style={smallBtn}
          >
            All
          </button>
          <button
            onClick={handleUnselectAll}
            disabled={noneObserved}
            style={smallBtn}
          >
            None
          </button>
        </div>
      </div>

      {/* Asset list */}
      <div style={{ flex: 1, overflow: "auto" }}>
        {universe.map((a) => {
          const hasQuote = !!quotes[a.ticker];
          const isSelected = selectedNode === a.ticker;
          const isObs = observedTickers.includes(a.ticker);

          return (
            <div
              key={a.ticker}
              onClick={() =>
                setSelectedNode(isSelected ? null : a.ticker)
              }
              style={{
                display: "flex",
                alignItems: "center",
                gap: 6,
                padding: "5px 10px",
                cursor: "pointer",
                background: isSelected ? "#1e3a5f" : "transparent",
                borderBottom: "1px solid #1e293b",
                fontSize: 12,
              }}
            >
              <input
                type="checkbox"
                checked={isObs}
                onChange={(e) => {
                  e.stopPropagation();
                  toggleObserved(a.ticker);
                }}
                onClick={(e) => e.stopPropagation()}
                style={{
                  accentColor: isObs ? "#3b82f6" : "#f97316",
                  cursor: "pointer",
                  flexShrink: 0,
                }}
                title={
                  isObs
                    ? "Observed - uncheck to infer"
                    : "Inferred - check to observe"
                }
              />
              <span
                style={{
                  display: "inline-block",
                  width: 7,
                  height: 7,
                  borderRadius: "50%",
                  background: isObs
                    ? observedColor(0.7)
                    : inferredColor(0.7),
                  flexShrink: 0,
                }}
              />
              <span
                style={{
                  color: hasQuote ? "#e2e8f0" : "#475569",
                  fontWeight: isSelected ? 700 : 500,
                }}
              >
                {a.ticker}
              </span>
              <span
                style={{
                  color: "#475569",
                  fontSize: 10,
                  marginLeft: "auto",
                }}
              >
                {a.sector}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

const smallBtn: React.CSSProperties = {
  fontSize: 10,
  padding: "2px 8px",
  background: "#1e293b",
  color: "#94a3b8",
  border: "1px solid #334155",
  borderRadius: 3,
  cursor: "pointer",
};
