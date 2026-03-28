import { useState, useEffect } from "react";
import { useEngine } from "../hooks/useEngine";
import { api } from "../api/client";

type PriorSource = "fetch" | "file" | "skip";

interface Props {
  isOpen: boolean;
  onClose: () => void;
}

export default function FetchPriorsModal({ isOpen, onClose }: Props) {
  const { universe } = useEngine();
  const [savedPriors, setSavedPriors] = useState<
    Record<string, { ticker: string; filename: string; timestamp: string }>
  >({});
  const [choices, setChoices] = useState<Record<string, PriorSource>>({});
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState("");

  // Load saved priors list when modal opens
  useEffect(() => {
    if (!isOpen) return;
    api.listSavedPriors().then(setSavedPriors).catch(() => setSavedPriors({}));
    // Default all to "fetch"
    const defaults: Record<string, PriorSource> = {};
    for (const a of universe) {
      defaults[a.ticker] = "fetch";
    }
    setChoices(defaults);
    setProgress("");
  }, [isOpen, universe]);

  const setChoice = (ticker: string, source: PriorSource) => {
    setChoices((prev) => ({ ...prev, [ticker]: source }));
  };

  const setAllTo = (source: PriorSource) => {
    const next: Record<string, PriorSource> = {};
    for (const a of universe) {
      // Only set "file" if file exists
      if (source === "file" && !savedPriors[a.ticker]) {
        next[a.ticker] = choices[a.ticker] ?? "fetch";
      } else {
        next[a.ticker] = source;
      }
    }
    setChoices(next);
  };

  const handleExecute = async () => {
    setRunning(true);
    try {
      // Step 1: Fetch quotes for all non-skipped tickers (needed for both "fetch" and "file")
      const fetchTickers = universe
        .map((a) => a.ticker)
        .filter((t) => choices[t] !== "skip");

      if (fetchTickers.length > 0) {
        setProgress("Fetching market data...");
        const quotes = await api.fetchQuotes();
        useEngine.setState({
          quotes,
          observedTickers: Object.keys(quotes),
          excludedQuotes: {},
          excludedPriorQuotes: {},
          addedQuotes: {},
          addedPriorQuotes: {},
          solveResult: null,
        });
      }

      // Step 2: Calibrate priors for "fetch" tickers
      const fetchSources = universe
        .map((a) => a.ticker)
        .filter((t) => choices[t] === "fetch");

      if (fetchSources.length > 0) {
        setProgress(`Calibrating priors from source (${fetchSources.length} assets)...`);
        await api.calibratePriors();
      }

      // Step 3: Load saved priors for "file" tickers and restore their markers
      const fileSources = universe
        .map((a) => a.ticker)
        .filter((t) => choices[t] === "file");

      const restoredExcl: Record<string, number[]> = {};
      const restoredAdded: Record<string, [number, number][]> = {};

      const updatedQuotes: Record<string, any> = {};

      for (const ticker of fileSources) {
        setProgress(`Loading saved prior for ${ticker}...`);
        const result = await api.loadPriorFromFile(ticker);
        if (result.excluded_indices?.length > 0) {
          restoredExcl[ticker] = result.excluded_indices;
        }
        if (result.added_quotes?.length > 0) {
          restoredAdded[ticker] = result.added_quotes as [number, number][];
        }
        if (result.chain_snapshot) {
          updatedQuotes[ticker] = result.chain_snapshot;
        }
      }

      // Restore exclusion/addition markers and updated chain data
      useEngine.setState((s) => ({
        quotes: { ...s.quotes, ...updatedQuotes },
        excludedPriorQuotes: { ...s.excludedPriorQuotes, ...restoredExcl },
        addedPriorQuotes: { ...s.addedPriorQuotes, ...restoredAdded },
      }));

      useEngine.setState((s) => ({
        priorsCalibrated: true,
        priorsVersion: s.priorsVersion + 1,
      }));
      setProgress("Done!");
      setTimeout(() => onClose(), 500);
    } catch (e: any) {
      setProgress(`Error: ${e.message}`);
    } finally {
      setRunning(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div style={overlayStyle} onClick={running ? undefined : onClose}>
      <div style={modalStyle} onClick={(e) => e.stopPropagation()}>
        <div style={headerStyle}>
          <h3 style={{ margin: 0, fontSize: 14, color: "#e2e8f0" }}>
            Fetch Priors
          </h3>
          <button onClick={onClose} disabled={running} style={closeBtn}>
            {"\u2715"}
          </button>
        </div>

        {/* Bulk actions */}
        <div style={{ display: "flex", gap: 6, padding: "0 16px 8px" }}>
          <button onClick={() => setAllTo("fetch")} style={bulkBtn}>
            All from source
          </button>
          <button onClick={() => setAllTo("file")} style={bulkBtn}>
            All from file
          </button>
          <button onClick={() => setAllTo("skip")} style={bulkBtn}>
            Skip all
          </button>
        </div>

        {/* Per-asset table */}
        <div style={{ flex: 1, overflow: "auto", padding: "0 16px" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
            <thead>
              <tr style={{ color: "#94a3b8", borderBottom: "1px solid #334155" }}>
                <th style={thStyle}>Asset</th>
                <th style={thStyle}>Fetch</th>
                <th style={thStyle}>File</th>
                <th style={thStyle}>Skip</th>
                <th style={thStyle}>Saved</th>
              </tr>
            </thead>
            <tbody>
              {universe.map((a) => {
                const hasFile = !!savedPriors[a.ticker];
                const choice = choices[a.ticker] ?? "fetch";
                return (
                  <tr
                    key={a.ticker}
                    style={{ borderBottom: "1px solid #1e293b" }}
                  >
                    <td style={tdStyle}>{a.ticker}</td>
                    <td style={tdCenter}>
                      <input
                        type="radio"
                        name={`prior-${a.ticker}`}
                        checked={choice === "fetch"}
                        onChange={() => setChoice(a.ticker, "fetch")}
                        disabled={running}
                      />
                    </td>
                    <td style={tdCenter}>
                      <input
                        type="radio"
                        name={`prior-${a.ticker}`}
                        checked={choice === "file"}
                        onChange={() => setChoice(a.ticker, "file")}
                        disabled={running || !hasFile}
                        style={{ opacity: hasFile ? 1 : 0.3 }}
                      />
                    </td>
                    <td style={tdCenter}>
                      <input
                        type="radio"
                        name={`prior-${a.ticker}`}
                        checked={choice === "skip"}
                        onChange={() => setChoice(a.ticker, "skip")}
                        disabled={running}
                      />
                    </td>
                    <td style={{ ...tdStyle, fontSize: 10, color: "#64748b" }}>
                      {hasFile
                        ? savedPriors[a.ticker].timestamp.slice(0, 16)
                        : "-"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Footer */}
        <div style={footerStyle}>
          {progress && (
            <span style={{ fontSize: 11, color: "#94a3b8" }}>{progress}</span>
          )}
          <button
            onClick={handleExecute}
            disabled={running}
            style={{
              ...execBtn,
              opacity: running ? 0.6 : 1,
              marginLeft: "auto",
            }}
          >
            {running ? "Running..." : "Execute"}
          </button>
        </div>
      </div>
    </div>
  );
}

const overlayStyle: React.CSSProperties = {
  position: "fixed",
  inset: 0,
  background: "rgba(0,0,0,0.6)",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  zIndex: 1000,
};

const modalStyle: React.CSSProperties = {
  background: "#0f172a",
  border: "1px solid #334155",
  borderRadius: 8,
  width: 520,
  maxHeight: "80vh",
  display: "flex",
  flexDirection: "column",
  overflow: "hidden",
};

const headerStyle: React.CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "center",
  padding: "12px 16px",
  borderBottom: "1px solid #1e293b",
};

const closeBtn: React.CSSProperties = {
  background: "none",
  border: "none",
  color: "#64748b",
  fontSize: 16,
  cursor: "pointer",
};

const bulkBtn: React.CSSProperties = {
  fontSize: 10,
  padding: "3px 8px",
  background: "#1e293b",
  color: "#94a3b8",
  border: "1px solid #334155",
  borderRadius: 3,
  cursor: "pointer",
};

const thStyle: React.CSSProperties = {
  textAlign: "left",
  padding: "4px 6px",
  fontWeight: 600,
  fontSize: 11,
};

const tdStyle: React.CSSProperties = {
  padding: "5px 6px",
  color: "#cbd5e1",
};

const tdCenter: React.CSSProperties = {
  ...tdStyle,
  textAlign: "center",
};

const footerStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 8,
  padding: "12px 16px",
  borderTop: "1px solid #1e293b",
};

const execBtn: React.CSSProperties = {
  padding: "6px 20px",
  background: "#6366f1",
  color: "#fff",
  border: "none",
  borderRadius: 5,
  cursor: "pointer",
  fontSize: 12,
  fontWeight: 600,
};
