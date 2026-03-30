import { useMemo } from "react";
import { useEngine } from "../hooks/useEngine";
import { groupByTicker, splitNodeKey, expiryLabel, uniqueTickers } from "../utils/nodeKey";
import { observedColor, inferredColor } from "../utils/colors";

interface Props {
  isOpen: boolean;
  onClose: () => void;
}

export default function ObservedMatrixPanel({ isOpen, onClose }: Props) {
  const {
    quotes, observedTickers, universe,
    toggleObserved, setSelectedNode,
  } = useEngine();

  const quoteKeys = Object.keys(quotes);
  const multiExpiry = useMemo(() => quoteKeys.some((k) => k.includes(":")), [quoteKeys]);

  // Build matrix data
  const { tickers, expiries, matrix } = useMemo(() => {
    if (!multiExpiry) {
      // Single maturity: each ticker is a node
      const tks = universe.map((a) => a.ticker).filter((t) => quotes[t]);
      const expiries = tks.length > 0 ? [quotes[tks[0]]?.expiry ?? ""] : [];
      const mx: Record<string, Record<string, string>> = {};
      for (const t of tks) {
        mx[t] = { [expiries[0]]: t };
      }
      return { tickers: tks, expiries, matrix: mx };
    }

    // Multi-expiry: build from quote keys
    const groups = groupByTicker(quoteKeys);
    const tks = uniqueTickers(quoteKeys);
    const expSet = new Set<string>();
    const T_map: Record<string, number> = {};
    for (const nk of quoteKeys) {
      const { expiry } = splitNodeKey(nk);
      expSet.add(expiry);
      T_map[expiry] = quotes[nk]?.T ?? 0;
    }
    const exps = [...expSet].sort((a, b) => (T_map[a] ?? 0) - (T_map[b] ?? 0));

    const mx: Record<string, Record<string, string>> = {};
    for (const t of tks) {
      mx[t] = {};
      for (const nk of (groups[t] ?? [])) {
        const { expiry } = splitNodeKey(nk);
        mx[t][expiry] = nk; // node key for this (ticker, expiry) cell
      }
    }
    return { tickers: tks, expiries: exps, matrix: mx };
  }, [quotes, quoteKeys, multiExpiry, universe]);

  const observedSet = useMemo(() => new Set(observedTickers), [observedTickers]);

  const handleSelectAll = () => {
    useEngine.setState({ observedTickers: quoteKeys });
  };
  const handleSelectNone = () => {
    useEngine.setState({ observedTickers: [] });
  };
  const handleSelectRow = (ticker: string) => {
    const rowKeys = Object.values(matrix[ticker] ?? {});
    const allObs = rowKeys.every((k) => observedSet.has(k));
    if (allObs) {
      useEngine.setState((s) => ({
        observedTickers: s.observedTickers.filter((t) => !rowKeys.includes(t)),
      }));
    } else {
      useEngine.setState((s) => ({
        observedTickers: [...s.observedTickers, ...rowKeys.filter((k) => !s.observedTickers.includes(k))],
      }));
    }
  };

  if (!isOpen) return null;

  return (
    <div style={overlayStyle} onClick={onClose}>
      <div style={panelStyle} onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div style={headerStyle}>
          <h3 style={{ margin: 0, fontSize: 13, color: "#e2e8f0" }}>
            Observed Nodes
          </h3>
          <div style={{ display: "flex", gap: 6, marginLeft: "auto" }}>
            <button onClick={handleSelectAll} style={tinyBtn}>All</button>
            <button onClick={handleSelectNone} style={tinyBtn}>None</button>
          </div>
          <button onClick={onClose} style={closeBtn}>{"\u2715"}</button>
        </div>

        {/* Matrix */}
        <div style={{ flex: 1, overflow: "auto", padding: "0 12px 12px" }}>
          {quoteKeys.length === 0 ? (
            <div style={{ padding: 20, color: "#64748b", textAlign: "center", fontSize: 12 }}>
              No quotes loaded.
            </div>
          ) : (
            <table style={{ borderCollapse: "collapse", fontSize: 10 }}>
              <thead>
                <tr>
                  <th style={{ ...mxHeader, position: "sticky", left: 0, background: "#0f172a", zIndex: 1 }}></th>
                  {expiries.map((exp) => {
                    const T = Object.values(quotes).find((q) => q.expiry === exp)?.T;
                    return (
                      <th key={exp} style={mxHeader} title={exp}>
                        {expiryLabel(exp, T)}
                      </th>
                    );
                  })}
                </tr>
              </thead>
              <tbody>
                {tickers.map((t) => {
                  const rowKeys = Object.values(matrix[t] ?? {});
                  const obsCount = rowKeys.filter((k) => observedSet.has(k)).length;
                  return (
                    <tr key={t}>
                      <td
                        style={{
                          ...mxHeader, position: "sticky", left: 0, background: "#0f172a",
                          zIndex: 1, textAlign: "right", fontWeight: 600, cursor: "pointer",
                        }}
                        onClick={() => handleSelectRow(t)}
                        title={`Toggle all ${t}`}
                      >
                        {t}
                        <span style={{ fontSize: 8, color: "#475569", marginLeft: 4 }}>
                          {obsCount}/{rowKeys.length}
                        </span>
                      </td>
                      {expiries.map((exp) => {
                        const nk = matrix[t]?.[exp];
                        if (!nk) {
                          return <td key={exp} style={{ padding: 1 }}><div style={emptyCell} /></td>;
                        }
                        const isObs = observedSet.has(nk);
                        return (
                          <td key={exp} style={{ padding: 1, textAlign: "center" }}>
                            <button
                              onClick={() => toggleObserved(nk)}
                              onDoubleClick={() => setSelectedNode(nk)}
                              title={`${t} ${exp}\n${isObs ? "Observed" : "Inferred"}\nDouble-click to select`}
                              style={{
                                width: 26, height: 20, padding: 0, fontSize: 9,
                                border: "1px solid " + (isObs ? "#3b82f6" : "#92400e"),
                                borderRadius: 3, cursor: "pointer",
                                background: isObs ? observedColor(0.6) : inferredColor(0.3),
                                color: isObs ? "#fff" : "#f97316",
                                fontWeight: 600,
                              }}
                            >
                              {isObs ? "O" : "I"}
                            </button>
                          </td>
                        );
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}

const overlayStyle: React.CSSProperties = {
  position: "fixed",
  inset: 0,
  background: "rgba(0,0,0,0.5)",
  display: "flex",
  alignItems: "stretch",
  justifyContent: "flex-start",
  zIndex: 900,
};

const panelStyle: React.CSSProperties = {
  background: "#0f172a",
  border: "1px solid #334155",
  borderRadius: "0 8px 8px 0",
  width: "auto",
  maxWidth: "80vw",
  minWidth: 300,
  display: "flex",
  flexDirection: "column",
  overflow: "hidden",
};

const headerStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 8,
  padding: "10px 12px",
  borderBottom: "1px solid #1e293b",
};

const closeBtn: React.CSSProperties = {
  background: "none",
  border: "none",
  color: "#64748b",
  fontSize: 14,
  cursor: "pointer",
  marginLeft: 8,
};

const tinyBtn: React.CSSProperties = {
  padding: "2px 8px",
  fontSize: 9,
  background: "#1e293b",
  border: "1px solid #334155",
  borderRadius: 3,
  color: "#94a3b8",
  cursor: "pointer",
};

const mxHeader: React.CSSProperties = {
  padding: "4px 6px",
  color: "#94a3b8",
  fontSize: 9,
  fontWeight: 600,
  whiteSpace: "nowrap",
  borderBottom: "1px solid #334155",
};

const emptyCell: React.CSSProperties = {
  width: 26,
  height: 20,
  background: "#0f172a",
  borderRadius: 3,
};
