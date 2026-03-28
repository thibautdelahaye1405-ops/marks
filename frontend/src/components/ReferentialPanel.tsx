import { useEngine } from "../hooks/useEngine";

export default function ReferentialPanel() {
  const { universe } = useEngine();

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {/* Data source */}
      <div>
        <div style={sectionTitle}>Data source</div>
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <label style={radioLabel}>
            <input
              type="radio"
              name="dataSource"
              value="yahoo"
              defaultChecked
              style={radioStyle}
            />
            <span style={{ color: "#e2e8f0" }}>Yahoo Finance</span>
          </label>
          <label style={{ ...radioLabel, opacity: 0.4 }}>
            <input
              type="radio"
              name="dataSource"
              value="bloomberg"
              disabled
              style={radioStyle}
            />
            <span style={{ color: "#94a3b8" }}>Bloomberg</span>
          </label>
          <label style={{ ...radioLabel, opacity: 0.4 }}>
            <input
              type="radio"
              name="dataSource"
              value="refinitiv"
              disabled
              style={radioStyle}
            />
            <span style={{ color: "#94a3b8" }}>Refinitiv</span>
          </label>
          <label style={{ ...radioLabel, opacity: 0.4 }}>
            <input
              type="radio"
              name="dataSource"
              value="csv"
              disabled
              style={radioStyle}
            />
            <span style={{ color: "#94a3b8" }}>CSV</span>
          </label>
        </div>
      </div>

      {/* Universe */}
      <div>
        <div style={sectionTitle}>Universe</div>
        <span
          style={{
            fontSize: 11,
            color: "#64748b",
            fontStyle: "italic",
            display: "block",
            marginBottom: 8,
          }}
        >
          Universe management coming soon
        </span>
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 2,
            padding: "6px 8px",
            background: "#1e293b",
            borderRadius: 4,
            border: "1px solid #334155",
          }}
        >
          {universe.map((a) => (
            <div
              key={a.ticker}
              style={{
                fontSize: 11,
                color: "#cbd5e1",
                padding: "3px 0",
                borderBottom: "1px solid #334155",
              }}
            >
              {a.ticker}
              <span style={{ color: "#64748b", marginLeft: 8, fontSize: 10 }}>
                {a.sector}
              </span>
            </div>
          ))}
          {universe.length === 0 && (
            <span style={{ fontSize: 11, color: "#475569" }}>
              No tickers loaded
            </span>
          )}
        </div>
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
