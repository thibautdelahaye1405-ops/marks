import { useState } from "react";
import { useEngine } from "../hooks/useEngine";

export default function MatrixEditor() {
  const { graphData, updateW } = useEngine();
  const [editCell, setEditCell] = useState<{ i: number; j: number } | null>(null);
  const [editValue, setEditValue] = useState("");

  if (!graphData) return null;

  const { tickers, W } = graphData;
  const N = tickers.length;

  const handleCellClick = (i: number, j: number) => {
    if (i === j) return; // no self-influence
    setEditCell({ i, j });
    setEditValue(W[i][j].toFixed(3));
  };

  const handleSave = () => {
    if (!editCell) return;
    const val = parseFloat(editValue);
    if (isNaN(val) || val < 0) return;

    const newW = W.map((row) => [...row]);
    newW[editCell.i][editCell.j] = val;

    // Check row sum
    const rowSum = newW[editCell.i].reduce((a, b) => a + b, 0);
    if (rowSum > 1.0) {
      // Scale down the row
      const scale = 0.95 / rowSum;
      for (let k = 0; k < N; k++) {
        if (k !== editCell.i) newW[editCell.i][k] *= scale;
      }
    }

    updateW(newW);
    setEditCell(null);
  };

  const maxW = Math.max(...W.flat().filter((v) => v > 0), 0.01);

  return (
    <div style={{ padding: 16 }}>
      <h3 style={{ margin: "0 0 8px", fontSize: 14, color: "#e2e8f0" }}>
        Influence Matrix W
      </h3>
      <div style={{ overflowX: "auto" }}>
        <table style={{ borderCollapse: "collapse", fontSize: 10 }}>
          <thead>
            <tr>
              <th style={headerCell}></th>
              {tickers.map((t) => (
                <th key={t} style={headerCell}>
                  {t}
                </th>
              ))}
              <th style={headerCell}>Sum</th>
            </tr>
          </thead>
          <tbody>
            {tickers.map((rowTicker, i) => {
              const rowSum = W[i].reduce((a, b) => a + b, 0);
              return (
                <tr key={rowTicker}>
                  <td style={headerCell}>{rowTicker}</td>
                  {tickers.map((_, j) => {
                    const val = W[i][j];
                    const isEditing =
                      editCell?.i === i && editCell?.j === j;
                    const intensity = val / maxW;

                    return (
                      <td
                        key={j}
                        onClick={() => handleCellClick(i, j)}
                        style={{
                          ...cellStyle,
                          background:
                            i === j
                              ? "#1e293b"
                              : `rgba(99, 102, 241, ${intensity * 0.6})`,
                          cursor: i === j ? "default" : "pointer",
                        }}
                      >
                        {isEditing ? (
                          <input
                            value={editValue}
                            onChange={(e) => setEditValue(e.target.value)}
                            onBlur={handleSave}
                            onKeyDown={(e) => e.key === "Enter" && handleSave()}
                            autoFocus
                            style={{
                              width: 36,
                              background: "#0f172a",
                              color: "#e2e8f0",
                              border: "1px solid #6366f1",
                              fontSize: 10,
                              textAlign: "center",
                              borderRadius: 2,
                            }}
                          />
                        ) : i === j ? (
                          <span style={{ color: "#334155" }}>-</span>
                        ) : val > 0.005 ? (
                          val.toFixed(2)
                        ) : (
                          ""
                        )}
                      </td>
                    );
                  })}
                  <td
                    style={{
                      ...cellStyle,
                      color: rowSum > 0.95 ? "#f87171" : "#64748b",
                      fontWeight: 600,
                    }}
                  >
                    {rowSum.toFixed(2)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div style={{ fontSize: 10, color: "#475569", marginTop: 4 }}>
        Click a cell to edit. Row sums should be &lt; 1 (residual = self-trust alpha).
      </div>
    </div>
  );
}

const headerCell: React.CSSProperties = {
  padding: "3px 6px",
  color: "#94a3b8",
  fontWeight: 600,
  fontSize: 10,
  textAlign: "center",
};

const cellStyle: React.CSSProperties = {
  padding: "3px 6px",
  color: "#cbd5e1",
  textAlign: "center",
  border: "1px solid #1e293b",
  minWidth: 36,
};
