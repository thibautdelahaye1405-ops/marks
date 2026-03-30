import { useState, useMemo, useCallback } from "react";
import { useEngine } from "../hooks/useEngine";
import { sectorColor } from "../utils/colors";
import { expiryLabel } from "../utils/nodeKey";
import type { Asset } from "../types";

export default function ReferentialPanel() {
  const {
    catalog, activeTickers, selectUniverse, saveSelection, addTicker,
    loading, universeUnsaved,
    availableExpiries, selectedExpiries, fetchExpiries, setExpirySelection,
  } = useEngine();

  const [search, setSearch] = useState("");
  const [addInput, setAddInput] = useState("");
  const [addError, setAddError] = useState("");
  const [addLoading, setAddLoading] = useState(false);
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});

  // Group catalog by sector
  const sectors = useMemo(() => {
    const map: Record<string, Asset[]> = {};
    for (const a of catalog) {
      (map[a.sector] ??= []).push(a);
    }
    // Sort sectors: Index first, then alphabetically
    const keys = Object.keys(map).sort((a, b) => {
      if (a === "Index") return -1;
      if (b === "Index") return 1;
      return a.localeCompare(b);
    });
    return keys.map((k) => ({ sector: k, assets: map[k] }));
  }, [catalog]);

  // Filter by search
  const filtered = useMemo(() => {
    if (!search.trim()) return sectors;
    const q = search.toLowerCase();
    return sectors
      .map((s) => ({
        ...s,
        assets: s.assets.filter(
          (a) =>
            a.ticker.toLowerCase().includes(q) ||
            a.name.toLowerCase().includes(q)
        ),
      }))
      .filter((s) => s.assets.length > 0);
  }, [sectors, search]);

  const activeSet = useMemo(() => new Set(activeTickers), [activeTickers]);

  const toggleTicker = useCallback(
    (ticker: string) => {
      const next = activeSet.has(ticker)
        ? activeTickers.filter((t) => t !== ticker)
        : [...activeTickers, ticker];
      selectUniverse(next);
    },
    [activeSet, activeTickers, selectUniverse]
  );

  const toggleSector = useCallback(
    (sectorAssets: Asset[]) => {
      const tickers = sectorAssets.map((a) => a.ticker);
      const allActive = tickers.every((t) => activeSet.has(t));
      const next = allActive
        ? activeTickers.filter((t) => !tickers.includes(t))
        : [...activeTickers, ...tickers.filter((t) => !activeSet.has(t))];
      selectUniverse(next);
    },
    [activeSet, activeTickers, selectUniverse]
  );

  const selectAll = useCallback(() => {
    selectUniverse(catalog.map((a) => a.ticker));
  }, [catalog, selectUniverse]);

  const selectNone = useCallback(() => {
    selectUniverse([]);
  }, [selectUniverse]);

  const handleAddTicker = useCallback(async () => {
    const ticker = addInput.trim().toUpperCase();
    if (!ticker) return;
    setAddError("");
    setAddLoading(true);
    try {
      await addTicker(ticker);
      setAddInput("");
    } catch (e: any) {
      const msg = e.message || "Failed to add ticker";
      // Extract detail from "404: ..." error format
      const match = msg.match(/\d+:\s*"?(.+)"?/);
      setAddError(match ? match[1].replace(/^"|"$/g, "") : msg);
    } finally {
      setAddLoading(false);
    }
  }, [addInput, addTicker]);

  const toggleCollapse = useCallback((sector: string) => {
    setCollapsed((c) => ({ ...c, [sector]: !c[sector] }));
  }, []);

  return (
    <div style={{ display: "flex", gap: 16, height: "100%" }}>
      {/* ── LEFT COLUMN: Data source + Expiry matrix ── */}
      <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", gap: 16 }}>

      {/* Data source */}
      <div>
        <div style={sectionTitle}>Data source</div>
        <div style={{ display: "flex", gap: 12 }}>
          <label style={radioLabel}>
            <input type="radio" name="dataSource" value="yahoo" defaultChecked style={radioStyle} />
            <span style={{ color: "#e2e8f0", fontSize: 11 }}>Yahoo Finance</span>
          </label>
          <label style={{ ...radioLabel, opacity: 0.4 }}>
            <input type="radio" name="dataSource" value="bloomberg" disabled style={radioStyle} />
            <span style={{ color: "#94a3b8", fontSize: 11 }}>Bloomberg</span>
          </label>
        </div>
      </div>

      {/* Expiry matrix */}
      <div style={{ flex: 1, minHeight: 0, display: "flex", flexDirection: "column" }}>
        <div style={sectionTitle}>Expiries (multi-maturity)</div>
        <div style={{ display: "flex", gap: 6, marginBottom: 8, flexWrap: "wrap" }}>
          <button
            onClick={() => { for (const t of activeTickers) fetchExpiries(t); }}
            disabled={loading || activeTickers.length === 0}
            style={smallBtn}
          >
            Fetch expiries
          </button>
          <button onClick={() => {
            const payload = { tickers: activeTickers, expiries: selectedExpiries };
            localStorage.setItem("marks_referential_selection", JSON.stringify(payload));
          }} style={tinyBtn} title="Save current assets + expiries selection">Save</button>
          <button onClick={() => {
            try {
              const raw = localStorage.getItem("marks_referential_selection");
              if (!raw) return;
              const payload = JSON.parse(raw);
              if (payload.tickers) selectUniverse(payload.tickers);
              if (payload.expiries) {
                for (const [t, exps] of Object.entries(payload.expiries)) {
                  setExpirySelection(t, exps as string[]);
                }
              }
            } catch {}
          }} style={tinyBtn} title="Load last saved selection">Load</button>
          {activeTickers.some((t) => availableExpiries[t]) && (
            <>
              <button onClick={() => {
                for (const t of activeTickers) {
                  const ae = availableExpiries[t];
                  if (ae) setExpirySelection(t, ae.expiries.slice(0, 3));
                }
              }} style={tinyBtn}>Front 3</button>
              <button onClick={() => {
                for (const t of activeTickers) {
                  const ae = availableExpiries[t];
                  if (ae) {
                    // Pick monthly: ~30d apart, take first of each month
                    const monthly: string[] = [];
                    let lastMonth = -1;
                    for (const e of ae.expiries) {
                      const m = new Date(e + "T00:00:00").getMonth();
                      if (m !== lastMonth) { monthly.push(e); lastMonth = m; }
                    }
                    setExpirySelection(t, monthly);
                  }
                }
              }} style={tinyBtn}>Monthly</button>
              <button onClick={() => {
                for (const t of activeTickers) {
                  const ae = availableExpiries[t];
                  if (ae) {
                    // Quarterly: Mar, Jun, Sep, Dec
                    const qMonths = new Set([2, 5, 8, 11]);
                    const quarterly = ae.expiries.filter((e) => {
                      const m = new Date(e + "T00:00:00").getMonth();
                      return qMonths.has(m);
                    });
                    setExpirySelection(t, quarterly.length > 0 ? quarterly : ae.expiries.slice(0, 3));
                  }
                }
              }} style={tinyBtn}>Quarterly</button>
              <button onClick={() => {
                for (const t of activeTickers) {
                  const ae = availableExpiries[t];
                  if (ae) setExpirySelection(t, ae.expiries);
                }
              }} style={tinyBtn}>All</button>
              <button onClick={() => {
                for (const t of activeTickers) setExpirySelection(t, []);
              }} style={tinyBtn}>Clear</button>
            </>
          )}
        </div>

        {(() => {
          const expMap = new Map<string, number>();
          for (const t of activeTickers) {
            const ae = availableExpiries[t];
            if (!ae) continue;
            ae.expiries.forEach((e, i) => {
              if (!expMap.has(e)) expMap.set(e, ae.T_values[i]);
            });
          }
          const allExpiries = [...expMap.entries()].sort((a, b) => a[1] - b[1]);

          if (allExpiries.length === 0) {
            return (
              <span style={{ fontSize: 10, color: "#475569" }}>
                Click "Fetch expiries" to load available dates from Yahoo.
              </span>
            );
          }

          return (
            <div style={{ overflowX: "auto", overflowY: "auto", flex: 1, border: "1px solid #334155", borderRadius: 4, background: "#0f172a" }}>
              <table style={{ borderCollapse: "collapse", fontSize: 9 }}>
                <thead>
                  <tr>
                    <th style={{ ...matrixHeader, position: "sticky", left: 0, top: 0, background: "#0f172a", zIndex: 2 }}></th>
                    {allExpiries.map(([exp, T]) => {
                      // Check if all active tickers that have this expiry are selected
                      const tickersWithExp = activeTickers.filter((t) => {
                        const ae = availableExpiries[t];
                        return ae && ae.expiries.includes(exp);
                      });
                      const allSelected = tickersWithExp.length > 0 && tickersWithExp.every((t) =>
                        (selectedExpiries[t] ?? []).includes(exp)
                      );
                      return (
                        <th key={exp} style={{ ...matrixHeader, position: "sticky", top: 0, background: "#0f172a", zIndex: 1 }} title={exp}>
                          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 1 }}>
                            <input
                              type="checkbox"
                              checked={allSelected}
                              onChange={() => {
                                for (const t of tickersWithExp) {
                                  const cur = selectedExpiries[t] ?? [];
                                  const next = allSelected
                                    ? cur.filter((e) => e !== exp)
                                    : cur.includes(exp) ? cur : [...cur, exp];
                                  setExpirySelection(t, next);
                                }
                              }}
                              style={{ accentColor: "#6366f1", width: 10, height: 10, cursor: "pointer" }}
                              title={`Toggle ${exp} for all assets`}
                            />
                            <span>{expiryLabel(exp, T)}</span>
                          </div>
                        </th>
                      );
                    })}
                  </tr>
                </thead>
                <tbody>
                  {activeTickers.map((t) => {
                    const ae = availableExpiries[t];
                    const available = new Set(ae?.expiries ?? []);
                    const sel = new Set(selectedExpiries[t] ?? []);
                    return (
                      <tr key={t}>
                        <td style={{ ...matrixHeader, position: "sticky", left: 0, background: "#0f172a", zIndex: 1, textAlign: "right", fontWeight: 600 }}>
                          {t}
                        </td>
                        {allExpiries.map(([exp]) => {
                          const avail = available.has(exp);
                          const on = sel.has(exp);
                          return (
                            <td key={exp} style={{ padding: 1, textAlign: "center" }}>
                              <button
                                disabled={!avail}
                                onClick={() => {
                                  const next = on
                                    ? (selectedExpiries[t] ?? []).filter((e) => e !== exp)
                                    : [...(selectedExpiries[t] ?? []), exp];
                                  setExpirySelection(t, next);
                                }}
                                style={{
                                  width: 22, height: 18, padding: 0, fontSize: 8,
                                  border: "1px solid " + (on ? "#6366f1" : "#334155"),
                                  borderRadius: 2, cursor: avail ? "pointer" : "default",
                                  background: on ? "#4f46e5" : avail ? "#1e293b" : "#0f172a",
                                  color: on ? "#fff" : avail ? "#64748b" : "#1e293b",
                                  opacity: avail ? 1 : 0.3,
                                }}
                              >
                                {on ? "\u2713" : "\u00B7"}
                              </button>
                            </td>
                          );
                        })}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          );
        })()}
      </div>

      </div>{/* end left column */}

      {/* ── RIGHT COLUMN: Asset selector ── */}
      <div style={{ width: 260, flexShrink: 0, display: "flex", flexDirection: "column", gap: 12 }}>

      {/* Universe */}
      <div style={{ flex: 1, minHeight: 0, display: "flex", flexDirection: "column" }}>
        <div style={{ ...sectionTitle, marginBottom: 6 }}>
          Universe
          <span style={{ fontWeight: 400, fontSize: 10, color: "#64748b", marginLeft: 8 }}>
            {activeTickers.length} / {catalog.length}
          </span>
        </div>

        {/* Search */}
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search tickers..."
          style={searchInput}
        />

        {/* All / None / Save buttons */}
        <div style={{ display: "flex", gap: 8, marginBottom: 6, alignItems: "center" }}>
          <button onClick={selectAll} style={smallBtn} disabled={loading}>
            All
          </button>
          <button onClick={selectNone} style={smallBtn} disabled={loading}>
            None
          </button>
          <div style={{ marginLeft: "auto" }}>
            <button
              onClick={saveSelection}
              disabled={loading || !universeUnsaved}
              style={{
                ...smallBtn,
                background: universeUnsaved ? "#312e81" : "#1e293b",
                borderColor: universeUnsaved ? "#6366f1" : "#334155",
                color: universeUnsaved ? "#a5b4fc" : "#475569",
              }}
            >
              Save
            </button>
          </div>
        </div>

        {/* Sector groups */}
        <div
          style={{
            flex: 1,
            overflowY: "auto",
            border: "1px solid #334155",
            borderRadius: 4,
            background: "#0f172a",
          }}
        >
          {filtered.map(({ sector, assets }) => {
            const sectorActive = assets.filter((a) => activeSet.has(a.ticker));
            const allOn = sectorActive.length === assets.length;
            const someOn = sectorActive.length > 0 && !allOn;
            const isCollapsed = collapsed[sector];

            return (
              <div key={sector}>
                {/* Sector header */}
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 6,
                    padding: "5px 8px",
                    background: "#1e293b",
                    borderBottom: "1px solid #334155",
                    cursor: "pointer",
                    userSelect: "none",
                  }}
                  onClick={() => toggleCollapse(sector)}
                >
                  <span style={{ fontSize: 9, color: "#64748b", width: 10 }}>
                    {isCollapsed ? "+" : "-"}
                  </span>
                  <input
                    type="checkbox"
                    checked={allOn}
                    ref={(el) => { if (el) el.indeterminate = someOn; }}
                    onChange={(e) => {
                      e.stopPropagation();
                      toggleSector(assets);
                    }}
                    onClick={(e) => e.stopPropagation()}
                    style={{ accentColor: sectorColor(sector) }}
                  />
                  <span
                    style={{
                      fontSize: 11,
                      fontWeight: 600,
                      color: sectorColor(sector),
                    }}
                  >
                    {sector}
                  </span>
                  <span style={{ fontSize: 10, color: "#475569", marginLeft: "auto" }}>
                    {sectorActive.length}/{assets.length}
                  </span>
                </div>

                {/* Ticker rows */}
                {!isCollapsed &&
                  assets.map((a) => (
                    <label
                      key={a.ticker}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 6,
                        padding: "3px 8px 3px 24px",
                        borderBottom: "1px solid #1e293b",
                        cursor: "pointer",
                        fontSize: 11,
                        background: activeSet.has(a.ticker)
                          ? "rgba(99, 102, 241, 0.06)"
                          : "transparent",
                      }}
                    >
                      <input
                        type="checkbox"
                        checked={activeSet.has(a.ticker)}
                        onChange={() => toggleTicker(a.ticker)}
                        disabled={loading}
                        style={{ accentColor: sectorColor(a.sector) }}
                      />
                      <span
                        style={{
                          color: activeSet.has(a.ticker) ? "#e2e8f0" : "#64748b",
                          fontWeight: 500,
                          width: 52,
                          fontFamily: "monospace",
                        }}
                      >
                        {a.ticker}
                      </span>
                      <span
                        style={{
                          color: activeSet.has(a.ticker) ? "#94a3b8" : "#475569",
                          fontSize: 10,
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                          whiteSpace: "nowrap",
                        }}
                      >
                        {a.name}
                      </span>
                    </label>
                  ))}
              </div>
            );
          })}
          {filtered.length === 0 && (
            <div style={{ padding: 12, color: "#475569", fontSize: 11, textAlign: "center" }}>
              No tickers match "{search}"
            </div>
          )}
        </div>
      </div>

      {/* Add ticker */}
      <div>
        <div style={sectionTitle}>Add ticker</div>
        <div style={{ display: "flex", gap: 6 }}>
          <input
            type="text"
            value={addInput}
            onChange={(e) => { setAddInput(e.target.value); setAddError(""); }}
            onKeyDown={(e) => e.key === "Enter" && handleAddTicker()}
            placeholder="e.g. COIN"
            style={{
              ...searchInput,
              marginBottom: 0,
              flex: 1,
              borderColor: addError ? "#ef4444" : "#334155",
            }}
            disabled={addLoading}
          />
          <button
            onClick={handleAddTicker}
            disabled={!addInput.trim() || addLoading}
            style={{
              ...smallBtn,
              opacity: !addInput.trim() || addLoading ? 0.4 : 1,
            }}
          >
            {addLoading ? "..." : "Add"}
          </button>
        </div>
        {addError ? (
          <span style={{ fontSize: 10, color: "#ef4444", marginTop: 4, display: "block" }}>
            {addError}
          </span>
        ) : (
          <span style={{ fontSize: 10, color: "#475569", marginTop: 4, display: "block" }}>
            Validates on Yahoo Finance before adding to catalog.
          </span>
        )}
      </div>

      </div>{/* end right column */}
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

const searchInput: React.CSSProperties = {
  width: "100%",
  padding: "5px 8px",
  fontSize: 11,
  background: "#1e293b",
  border: "1px solid #334155",
  borderRadius: 4,
  color: "#e2e8f0",
  outline: "none",
  marginBottom: 6,
  boxSizing: "border-box",
};

const smallBtn: React.CSSProperties = {
  padding: "3px 10px",
  fontSize: 10,
  fontWeight: 600,
  background: "#1e293b",
  border: "1px solid #334155",
  borderRadius: 3,
  color: "#94a3b8",
  cursor: "pointer",
};

const tinyBtn: React.CSSProperties = {
  padding: "1px 6px",
  fontSize: 9,
  background: "#0f172a",
  border: "1px solid #334155",
  borderRadius: 2,
  color: "#64748b",
  cursor: "pointer",
};

const matrixHeader: React.CSSProperties = {
  padding: "3px 6px",
  color: "#94a3b8",
  fontSize: 9,
  fontWeight: 600,
  whiteSpace: "nowrap",
  borderBottom: "1px solid #334155",
};
