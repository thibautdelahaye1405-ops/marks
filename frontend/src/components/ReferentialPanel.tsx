import { useState, useMemo, useCallback } from "react";
import { useEngine } from "../hooks/useEngine";
import { sectorColor } from "../utils/colors";
import type { Asset } from "../types";

export default function ReferentialPanel() {
  const {
    catalog, activeTickers, selectUniverse, saveSelection, addTicker,
    loading, universeUnsaved,
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
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
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
            <input type="radio" name="dataSource" value="bloomberg" disabled style={radioStyle} />
            <span style={{ color: "#94a3b8" }}>Bloomberg</span>
          </label>
          <label style={{ ...radioLabel, opacity: 0.4 }}>
            <input type="radio" name="dataSource" value="refinitiv" disabled style={radioStyle} />
            <span style={{ color: "#94a3b8" }}>Refinitiv</span>
          </label>
        </div>
      </div>

      {/* Universe */}
      <div>
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
            maxHeight: 380,
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
