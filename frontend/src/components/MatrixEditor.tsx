import { useState, useRef, useCallback, useEffect } from "react";
import { useEngine } from "../hooks/useEngine";

export default function MatrixEditor() {
  const {
    graphData, updateW, sparsifyW, resetW,
    configSnapshots, fetchConfigSnapshots, saveConfigSnapshot,
    applyConfigSnapshot, deleteConfigSnapshot,
  } = useEngine();

  const [cursor, setCursor] = useState<{ i: number; j: number } | null>(null);
  const [editValue, setEditValue] = useState("");
  const [isEditing, setIsEditing] = useState(false);
  const [sparseThreshold, setSparseThreshold] = useState(0.01);
  const [showSparsePreview, setShowSparsePreview] = useState(false);
  const [saveLabel, setSaveLabel] = useState("");
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const tableRef = useRef<HTMLTableElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Load config snapshots on mount
  useEffect(() => {
    fetchConfigSnapshots();
  }, [fetchConfigSnapshots]);

  if (!graphData) return null;

  const { tickers, W } = graphData;
  const N = tickers.length;
  const maxW = Math.max(...W.flat().filter((v) => v > 0), 0.01);

  // --- Cell navigation ---

  const moveCursor = useCallback(
    (di: number, dj: number) => {
      setCursor((prev) => {
        if (!prev) return { i: 0, j: 1 };
        let ni = prev.i + di;
        let nj = prev.j + dj;
        // Wrap and skip diagonal
        if (nj >= N) { nj = 0; ni++; }
        if (nj < 0) { nj = N - 1; ni--; }
        if (ni >= N) ni = 0;
        if (ni < 0) ni = N - 1;
        if (ni === nj) {
          nj += dj || 1;
          if (nj >= N) { nj = 0; ni = (ni + 1) % N; }
          if (nj < 0) { nj = N - 1; ni = (ni - 1 + N) % N; }
        }
        return { i: ni, j: nj };
      });
      setIsEditing(false);
    },
    [N],
  );

  const commitEdit = useCallback(() => {
    if (!cursor || !isEditing) return;
    const val = parseFloat(editValue);
    if (isNaN(val) || val < 0) {
      setIsEditing(false);
      return;
    }
    const newW = W.map((row) => [...row]);
    newW[cursor.i][cursor.j] = val;
    // Check row sum — if > 1, scale down non-edited cells in row
    const rowSum = newW[cursor.i].reduce((a, b) => a + b, 0);
    if (rowSum > 1.0) {
      const scale = 0.95 / rowSum;
      for (let k = 0; k < N; k++) {
        if (k !== cursor.i) newW[cursor.i][k] *= scale;
      }
    }
    updateW(newW);
    setIsEditing(false);
  }, [cursor, isEditing, editValue, W, N, updateW]);

  const startEdit = useCallback(
    (i: number, j: number) => {
      if (i === j) return;
      setCursor({ i, j });
      setEditValue(W[i][j].toFixed(3));
      setIsEditing(true);
    },
    [W],
  );

  // --- Keyboard handler ---

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (isEditing) {
        if (e.key === "Enter") { commitEdit(); moveCursor(1, 0); e.preventDefault(); }
        else if (e.key === "Tab") { commitEdit(); moveCursor(0, e.shiftKey ? -1 : 1); e.preventDefault(); }
        else if (e.key === "Escape") { setIsEditing(false); e.preventDefault(); }
        return;
      }
      // Navigation mode
      if (e.key === "ArrowUp") { moveCursor(-1, 0); e.preventDefault(); }
      else if (e.key === "ArrowDown") { moveCursor(1, 0); e.preventDefault(); }
      else if (e.key === "ArrowLeft") { moveCursor(0, -1); e.preventDefault(); }
      else if (e.key === "ArrowRight") { moveCursor(0, 1); e.preventDefault(); }
      else if (e.key === "Tab") { moveCursor(0, e.shiftKey ? -1 : 1); e.preventDefault(); }
      else if (e.key === "Enter" || e.key === "F2") {
        if (cursor && cursor.i !== cursor.j) {
          setEditValue(W[cursor.i][cursor.j].toFixed(3));
          setIsEditing(true);
          e.preventDefault();
        }
      } else if (e.key === "Delete" || e.key === "Backspace") {
        // Zero out selected cell
        if (cursor && cursor.i !== cursor.j) {
          const newW = W.map((row) => [...row]);
          newW[cursor.i][cursor.j] = 0;
          updateW(newW);
          e.preventDefault();
        }
      } else if (/^[0-9.]$/.test(e.key)) {
        // Start typing a number → enter edit mode
        if (cursor && cursor.i !== cursor.j) {
          setEditValue(e.key);
          setIsEditing(true);
          e.preventDefault();
        }
      }
    },
    [isEditing, cursor, commitEdit, moveCursor, W, updateW],
  );

  // --- Paste handler (Excel/Sheets) ---

  const handlePaste = useCallback(
    (e: React.ClipboardEvent) => {
      if (!cursor) return;
      const text = e.clipboardData.getData("text/plain");
      if (!text) return;
      e.preventDefault();

      const rows = text.trim().split(/\r?\n/).map((line) =>
        line.split(/\t/).map((c) => parseFloat(c.trim())),
      );

      const newW = W.map((row) => [...row]);
      for (let di = 0; di < rows.length; di++) {
        for (let dj = 0; dj < rows[di].length; dj++) {
          const ti = cursor.i + di;
          const tj = cursor.j + dj;
          if (ti >= N || tj >= N || ti === tj) continue;
          const val = rows[di][dj];
          if (!isNaN(val) && val >= 0) {
            newW[ti][tj] = val;
          }
        }
      }
      // Validate row sums
      for (let i = 0; i < N; i++) {
        const rowSum = newW[i].reduce((a, b) => a + b, 0);
        if (rowSum > 1.0) {
          const scale = 0.95 / rowSum;
          for (let k = 0; k < N; k++) newW[i][k] *= scale;
        }
      }
      updateW(newW);
    },
    [cursor, W, N, updateW],
  );

  // --- Zero row / zero column ---

  const zeroRow = (i: number) => {
    const newW = W.map((row) => [...row]);
    for (let j = 0; j < N; j++) newW[i][j] = 0;
    updateW(newW);
  };

  const zeroCol = (j: number) => {
    const newW = W.map((row) => [...row]);
    for (let i = 0; i < N; i++) newW[i][j] = 0;
    updateW(newW);
  };

  // --- Sparsification preview ---

  const isBelowThreshold = (val: number) =>
    showSparsePreview && val > 0 && val < sparseThreshold;

  // Focus input when editing starts
  useEffect(() => {
    if (isEditing && inputRef.current) inputRef.current.focus();
  }, [isEditing]);

  // --- Save dialog ---

  const handleSave = () => {
    if (!saveLabel.trim()) return;
    saveConfigSnapshot(saveLabel.trim());
    setSaveLabel("");
    setShowSaveDialog(false);
  };

  return (
    <div
      style={{ padding: 16, outline: "none" }}
      tabIndex={0}
      onKeyDown={handleKeyDown}
      onPaste={handlePaste}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
        <h3 style={{ margin: 0, fontSize: 14, color: "#e2e8f0" }}>
          Influence Matrix W
        </h3>
        <div style={{ flex: 1 }} />
        <button onClick={() => resetW()} style={btnStyle} title="Regenerate W from correlations + liquidity">
          Reset to Auto
        </button>
        <button onClick={() => setShowSaveDialog(!showSaveDialog)} style={btnStyle}>
          Save Config
        </button>
      </div>

      {/* Save dialog */}
      {showSaveDialog && (
        <div style={{ marginBottom: 8, display: "flex", gap: 6, alignItems: "center" }}>
          <input
            value={saveLabel}
            onChange={(e) => setSaveLabel(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSave()}
            placeholder="Config label..."
            style={textInputStyle}
            autoFocus
          />
          <button onClick={handleSave} style={btnStyle}>Save</button>
          <button onClick={() => setShowSaveDialog(false)} style={btnStyle}>Cancel</button>
        </div>
      )}

      {/* Saved configs dropdown */}
      {configSnapshots.length > 0 && (
        <div style={{ marginBottom: 8 }}>
          <div style={{ fontSize: 10, color: "#94a3b8", marginBottom: 4 }}>Saved Configs:</div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
            {configSnapshots.map((snap) => (
              <div key={snap.id} style={chipStyle}>
                <span
                  onClick={() => applyConfigSnapshot(snap.id)}
                  style={{ cursor: "pointer" }}
                  title={`Load: ${snap.label}\n${snap.timestamp}\nModel: ${snap.smile_model}`}
                >
                  {snap.label}
                </span>
                <span
                  onClick={() => deleteConfigSnapshot(snap.id)}
                  style={{ cursor: "pointer", color: "#64748b", marginLeft: 4 }}
                  title="Delete"
                >
                  x
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Sparsification controls */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
        <label style={{ fontSize: 10, color: "#94a3b8" }}>Sparsify threshold:</label>
        <input
          type="range"
          min={0}
          max={0.10}
          step={0.005}
          value={sparseThreshold}
          onChange={(e) => setSparseThreshold(parseFloat(e.target.value))}
          style={{ width: 100 }}
        />
        <span style={{ fontSize: 10, color: "#94a3b8", minWidth: 30 }}>
          {sparseThreshold.toFixed(3)}
        </span>
        <button
          onMouseEnter={() => setShowSparsePreview(true)}
          onMouseLeave={() => setShowSparsePreview(false)}
          style={{ ...btnStyle, fontSize: 10 }}
          title="Hover to preview, click to apply"
          onClick={() => { sparsifyW(sparseThreshold); setShowSparsePreview(false); }}
        >
          Apply
        </button>
      </div>

      {/* Matrix table */}
      <div style={{ overflowX: "auto" }}>
        <table ref={tableRef} style={{ borderCollapse: "collapse", fontSize: 10 }}>
          <thead>
            <tr>
              <th style={headerCell}></th>
              {tickers.map((t, j) => (
                <th
                  key={t}
                  style={{ ...headerCell, cursor: "pointer" }}
                  onClick={() => zeroCol(j)}
                  title={`Click to zero column ${t}`}
                >
                  {t.length > 6 ? t.slice(0, 6) : t}
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
                  <td
                    style={{ ...headerCell, cursor: "pointer" }}
                    onClick={() => zeroRow(i)}
                    title={`Click to zero row ${rowTicker}`}
                  >
                    {rowTicker.length > 6 ? rowTicker.slice(0, 6) : rowTicker}
                  </td>
                  {tickers.map((_, j) => {
                    const val = W[i][j];
                    const isCursor = cursor?.i === i && cursor?.j === j;
                    const editingThis = isCursor && isEditing;
                    const intensity = val / maxW;
                    const dimmed = isBelowThreshold(val);

                    return (
                      <td
                        key={j}
                        onClick={() => {
                          if (i === j) return;
                          setCursor({ i, j });
                          setIsEditing(false);
                        }}
                        onDoubleClick={() => startEdit(i, j)}
                        style={{
                          ...cellStyle,
                          background: i === j
                            ? "#1e293b"
                            : isCursor
                              ? "#334155"
                              : `rgba(99, 102, 241, ${intensity * 0.6})`,
                          cursor: i === j ? "default" : "pointer",
                          outline: isCursor ? "2px solid #6366f1" : "none",
                          outlineOffset: -2,
                          opacity: dimmed ? 0.3 : 1,
                        }}
                      >
                        {editingThis ? (
                          <input
                            ref={inputRef}
                            value={editValue}
                            onChange={(e) => setEditValue(e.target.value)}
                            onBlur={commitEdit}
                            onKeyDown={(e) => {
                              if (e.key === "Enter") { commitEdit(); moveCursor(1, 0); e.preventDefault(); e.stopPropagation(); }
                              else if (e.key === "Tab") { commitEdit(); moveCursor(0, e.shiftKey ? -1 : 1); e.preventDefault(); e.stopPropagation(); }
                              else if (e.key === "Escape") { setIsEditing(false); e.preventDefault(); e.stopPropagation(); }
                            }}
                            style={cellInputStyle}
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
        Arrows/Tab to navigate. Enter or type to edit. Delete to zero. Paste from Excel. Click row/col headers to zero.
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

const cellInputStyle: React.CSSProperties = {
  width: 36,
  background: "#0f172a",
  color: "#e2e8f0",
  border: "1px solid #6366f1",
  fontSize: 10,
  textAlign: "center",
  borderRadius: 2,
};

const btnStyle: React.CSSProperties = {
  padding: "3px 8px",
  fontSize: 10,
  background: "#1e293b",
  color: "#94a3b8",
  border: "1px solid #334155",
  borderRadius: 3,
  cursor: "pointer",
};

const textInputStyle: React.CSSProperties = {
  padding: "3px 8px",
  fontSize: 10,
  background: "#0f172a",
  color: "#e2e8f0",
  border: "1px solid #334155",
  borderRadius: 3,
  width: 160,
};

const chipStyle: React.CSSProperties = {
  padding: "2px 8px",
  fontSize: 10,
  background: "#1e293b",
  color: "#cbd5e1",
  border: "1px solid #334155",
  borderRadius: 12,
  display: "flex",
  alignItems: "center",
  gap: 2,
};
