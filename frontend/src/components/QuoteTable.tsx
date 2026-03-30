import { useState, useCallback, useMemo } from "react";
import { useEngine } from "../hooks/useEngine";
import { observedColor, inferredColor } from "../utils/colors";
import { groupByTicker, tickerOf, splitNodeKey, expiryLabel } from "../utils/nodeKey";

interface QuoteTableProps {
  onOpenObservedMatrix?: () => void;
}

export default function QuoteTable({ onOpenObservedMatrix }: QuoteTableProps) {
  const {
    universe,
    quotes,
    selectedNode,
    observedTickers,
    loading,
    propagate,
    setSelectedNode,
  } = useEngine();

  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});

  const hasQuotes = Object.keys(quotes).length > 0;
  const quoteKeys = Object.keys(quotes);

  const multiExpiry = useMemo(
    () => quoteKeys.some((k) => k.includes(":")),
    [quoteKeys]
  );

  const tickerGroups = useMemo(() => {
    if (!multiExpiry) return null;
    return groupByTicker(quoteKeys);
  }, [multiExpiry, quoteKeys]);

  const toggleCollapse = useCallback((ticker: string) => {
    setCollapsed((c) => ({ ...c, [ticker]: !c[ticker] }));
  }, []);

  if (universe.length === 0) return null;

  const observedSet = new Set(observedTickers);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      {/* Propagate + Observed Matrix buttons */}
      <div style={{ padding: "8px 10px 0", display: "flex", flexDirection: "column", gap: 4 }}>
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
        {hasQuotes && onOpenObservedMatrix && (
          <button
            onClick={onOpenObservedMatrix}
            style={{
              width: "100%",
              padding: "4px 0",
              background: "#1e293b",
              color: "#94a3b8",
              border: "1px solid #334155",
              borderRadius: 4,
              cursor: "pointer",
              fontSize: 10,
            }}
          >
            Observed Matrix
          </button>
        )}
      </div>

      {/* Header */}
      <div style={{ padding: "8px 10px 4px" }}>
        <span style={{ fontSize: 12, fontWeight: 600, color: "#94a3b8" }}>
          Assets
        </span>
      </div>

      {/* Asset list */}
      <div style={{ flex: 1, overflow: "auto" }}>
        {multiExpiry && tickerGroups
          ? /* ── Multi-expiry: 2-level tree ── */
            universe.map((a) => {
              const nodeKeys = tickerGroups[a.ticker] ?? [];
              if (nodeKeys.length === 0) return null;
              const isTickerCollapsed = collapsed[a.ticker];
              const obsCount = nodeKeys.filter((k) => observedSet.has(k)).length;
              const isSelectedTicker = selectedNode ? tickerOf(selectedNode) === a.ticker : false;

              return (
                <div key={a.ticker}>
                  {/* Ticker header row */}
                  <div
                    onClick={() => toggleCollapse(a.ticker)}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 6,
                      padding: "5px 10px",
                      cursor: "pointer",
                      background: isSelectedTicker ? "#1e293b" : "transparent",
                      borderBottom: "1px solid #1e293b",
                      fontSize: 12,
                      userSelect: "none",
                    }}
                  >
                    <span style={{ fontSize: 9, color: "#64748b", width: 10 }}>
                      {isTickerCollapsed ? "\u25B6" : "\u25BC"}
                    </span>
                    <span
                      style={{
                        display: "inline-block",
                        width: 7,
                        height: 7,
                        borderRadius: "50%",
                        background: obsCount > 0 ? observedColor(0.7) : inferredColor(0.7),
                        flexShrink: 0,
                      }}
                    />
                    <span style={{ color: "#e2e8f0", fontWeight: 600 }}>
                      {a.ticker}
                    </span>
                    <span style={{ color: "#475569", fontSize: 10, marginLeft: "auto" }}>
                      {obsCount}/{nodeKeys.length}
                    </span>
                  </div>

                  {/* Maturity sub-rows */}
                  {!isTickerCollapsed &&
                    nodeKeys.map((nk) => {
                      const q = quotes[nk];
                      const isSelected = selectedNode === nk;
                      const isObs = observedSet.has(nk);
                      const { expiry } = splitNodeKey(nk);
                      const label = q ? expiryLabel(expiry, q.T) : expiryLabel(expiry);

                      return (
                        <div
                          key={nk}
                          onClick={() => setSelectedNode(isSelected ? null : nk)}
                          style={{
                            display: "flex",
                            alignItems: "center",
                            gap: 6,
                            padding: "4px 10px 4px 28px",
                            cursor: "pointer",
                            background: isSelected ? "#1e3a5f" : "transparent",
                            borderBottom: "1px solid #0f172a",
                            fontSize: 11,
                          }}
                        >
                          <span
                            style={{
                              display: "inline-block",
                              width: 5,
                              height: 5,
                              borderRadius: "50%",
                              background: isObs ? observedColor(0.5) : inferredColor(0.5),
                              flexShrink: 0,
                            }}
                          />
                          <span
                            style={{
                              color: "#cbd5e1",
                              fontWeight: isSelected ? 700 : 400,
                            }}
                          >
                            {label}
                          </span>
                        </div>
                      );
                    })}
                </div>
              );
            })
          : /* ── Single-maturity: flat list ── */
            universe.map((a) => {
              const hasQuote = !!quotes[a.ticker];
              const isSelected = selectedNode === a.ticker;
              const isObs = observedSet.has(a.ticker);

              return (
                <div
                  key={a.ticker}
                  onClick={() => setSelectedNode(isSelected ? null : a.ticker)}
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
                  <span
                    style={{
                      display: "inline-block",
                      width: 7,
                      height: 7,
                      borderRadius: "50%",
                      background: isObs ? observedColor(0.7) : inferredColor(0.7),
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
