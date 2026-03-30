import { useState, useEffect, useRef, useCallback } from "react";

interface Props {
  /** Cytoscape core instance from GraphView */
  cy: any;
  /** Full W matrix */
  W: number[][];
  /** All tickers in order */
  tickers: string[];
  /** Neumann series terms: [n_terms][n_unobs][M] */
  neumannTerms: number[][][] | null;
  /** Per-node observed flag */
  nodeObserved: Record<string, boolean>;
  /** Whether animation mode is active */
  active: boolean;
  onClose: () => void;
}

/** Hop pulse colors — from cool to warm as propagation deepens */
const HOP_COLORS = [
  "#6366f1", // indigo — direct from observed
  "#8b5cf6", // violet
  "#a855f7", // purple
  "#d946ef", // fuchsia
  "#f43f5e", // rose — deepest hop
];

const EDGE_PULSE_COLOR = "#e2e8f0";
const GLOW_COLOR_OBS = "rgba(99, 102, 241, 0.7)";

/**
 * Overlay controls + animation logic for hop-by-hop propagation visualization.
 * Drives the existing Cytoscape instance: animates edge opacity/color and
 * node border glow to show influence "current" flowing through the graph.
 */
export default function PropagationAnimation({
  cy,
  W,
  tickers,
  neumannTerms,
  nodeObserved,
  active,
  onClose,
}: Props) {
  const [playing, setPlaying] = useState(false);
  const [currentHop, setCurrentHop] = useState(-1); // -1 = reset state
  const [speed, setSpeed] = useState(1200); // ms per hop
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const maxHop = neumannTerms ? neumannTerms.length - 1 : 0;

  // Derived: ordered observed/unobserved indices
  const obsIdx = tickers.map((t, i) => ({ t, i })).filter(({ t }) => nodeObserved[t]).map(({ i }) => i);
  const unobsIdx = tickers.map((t, i) => ({ t, i })).filter(({ t }) => !nodeObserved[t]).map(({ i }) => i);
  const unobsTickers = unobsIdx.map((i) => tickers[i]);

  // Compute per-node ATM vol contribution at each hop (cumulative)
  // neumannTerms[k][ui][0] = cumulative ATM vol delta at hop k for unobserved node ui
  const hopValues = useRef<number[][]>([]); // [hop][unobsIdx] — incremental ATM vol
  const cumulValues = useRef<number[][]>([]); // [hop][unobsIdx] — cumulative ATM vol

  useEffect(() => {
    if (!neumannTerms || neumannTerms.length === 0) return;
    const nHops = neumannTerms.length;
    const nUnobs = unobsIdx.length;

    const cumul: number[][] = [];
    const incr: number[][] = [];

    for (let k = 0; k < nHops; k++) {
      const c: number[] = [];
      const d: number[] = [];
      for (let ui = 0; ui < nUnobs; ui++) {
        // ATM vol component = index 0 in beta vector
        const val = neumannTerms[k]?.[ui]?.[0] ?? 0;
        c.push(val);
        // Incremental = this hop's contribution minus previous
        if (k === 0) {
          d.push(val);
        } else {
          const prev = neumannTerms[k - 1]?.[ui]?.[0] ?? 0;
          d.push(val - prev);
        }
      }
      cumul.push(c);
      incr.push(d);
    }

    cumulValues.current = cumul;
    hopValues.current = incr;
  }, [neumannTerms, unobsIdx.length]);

  // Reset visual state
  const resetVisuals = useCallback(() => {
    if (!cy) return;
    // Reset all edges to default
    cy.edges().forEach((e: any) => {
      e.removeClass("hop-active hop-pulse");
      e.data("animColor", "");
      e.data("animWidth", "");
      e.data("animOpacity", "");
    });
    // Reset all nodes
    cy.nodes().forEach((n: any) => {
      n.removeClass("hop-reached hop-source");
      n.data("hopGlow", "");
      n.data("hopBorderWidth", "");
      n.data("hopLabel", "");
    });
  }, [cy]);

  // Apply visuals for a given hop
  const applyHop = useCallback(
    (hop: number) => {
      if (!cy || !neumannTerms || hop < 0) return;

      // First reset everything
      resetVisuals();

      // Mark observed nodes as sources (always glowing)
      for (const oi of obsIdx) {
        const node = cy.getElementById(tickers[oi]);
        if (node.empty()) continue;
        node.addClass("hop-source");
      }

      // Max cumulative value for normalization
      const allCumul = cumulValues.current[hop] ?? [];
      const maxCumul = Math.max(...allCumul.map(Math.abs), 1e-10);

      // For each hop up to current, determine which edges were active
      // Hop 0: edges from observed → unobserved
      // Hop k>0: edges from unobserved → unobserved (where source was reached at k-1)

      // Track which unobserved nodes are "reached" at each step
      const reachedAtHop: Set<number>[] = []; // [hop] -> set of unobs indices reached
      for (let k = 0; k <= hop; k++) {
        const reached = new Set<number>();
        const incrVals = hopValues.current[k] ?? [];
        for (let ui = 0; ui < unobsTickers.length; ui++) {
          if (Math.abs(incrVals[ui]) > 1e-8) reached.add(ui);
        }
        reachedAtHop.push(reached);
      }

      // Light up edges for each hop
      for (let k = 0; k <= hop; k++) {
        const hopColor = HOP_COLORS[Math.min(k, HOP_COLORS.length - 1)];
        const isCurrent = k === hop;

        if (k === 0) {
          // Edges: observed → unobserved
          for (const oi of obsIdx) {
            for (let ui = 0; ui < unobsTickers.length; ui++) {
              const gi = unobsIdx[ui]; // global index
              const w = W[gi]?.[oi] ?? 0;
              if (w < 0.005) continue;
              const edgeId = `${tickers[oi]}->${tickers[gi]}`;
              const edge = cy.getElementById(edgeId);
              if (edge.empty()) continue;
              edge.addClass(isCurrent ? "hop-pulse" : "hop-active");
              edge.style({
                "line-color": isCurrent ? EDGE_PULSE_COLOR : hopColor,
                "target-arrow-color": isCurrent ? EDGE_PULSE_COLOR : hopColor,
                opacity: isCurrent ? 1.0 : 0.5,
                width: isCurrent ? 2 + w * 8 : 1 + w * 4,
              });
            }
          }
        } else {
          // Edges: unobserved (reached at k-1) → unobserved (reached at k)
          const prevReached = reachedAtHop[k - 1];
          const currReached = reachedAtHop[k];

          for (const srcUi of prevReached) {
            const srcGi = unobsIdx[srcUi];
            for (const tgtUi of currReached) {
              const tgtGi = unobsIdx[tgtUi];
              if (srcGi === tgtGi) continue;
              const w = W[tgtGi]?.[srcGi] ?? 0;
              if (w < 0.005) continue;
              const edgeId = `${tickers[srcGi]}->${tickers[tgtGi]}`;
              const edge = cy.getElementById(edgeId);
              if (edge.empty()) continue;
              edge.addClass(isCurrent ? "hop-pulse" : "hop-active");
              edge.style({
                "line-color": isCurrent ? EDGE_PULSE_COLOR : hopColor,
                "target-arrow-color": isCurrent ? EDGE_PULSE_COLOR : hopColor,
                opacity: isCurrent ? 1.0 : 0.4,
                width: isCurrent ? 2 + w * 6 : 1 + w * 3,
              });
            }
          }
        }
      }

      // Glow unobserved nodes proportional to cumulative ATM vol
      for (let ui = 0; ui < unobsTickers.length; ui++) {
        const node = cy.getElementById(unobsTickers[ui]);
        if (node.empty()) continue;
        const cumVal = allCumul[ui] ?? 0;
        const intensity = Math.min(Math.abs(cumVal) / maxCumul, 1);
        if (intensity < 0.01) continue;

        node.addClass("hop-reached");
        const hopColor = HOP_COLORS[Math.min(hop, HOP_COLORS.length - 1)];
        node.style({
          "border-color": hopColor,
          "border-width": 2 + intensity * 4,
          "border-opacity": 0.5 + intensity * 0.5,
        });
      }
    },
    [cy, neumannTerms, obsIdx, unobsIdx, unobsTickers, tickers, W, resetVisuals],
  );

  // Auto-play timer
  useEffect(() => {
    if (!playing || !active) return;
    timerRef.current = setTimeout(() => {
      setCurrentHop((prev) => {
        const next = prev + 1;
        if (next > maxHop) {
          setPlaying(false);
          return prev;
        }
        return next;
      });
    }, speed);
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [playing, currentHop, speed, maxHop, active]);

  // Apply visuals when hop changes
  useEffect(() => {
    if (!active) return;
    if (currentHop >= 0) {
      applyHop(currentHop);
    } else {
      resetVisuals();
    }
  }, [currentHop, active, applyHop, resetVisuals]);

  // Reset when deactivating
  useEffect(() => {
    if (!active) {
      resetVisuals();
      setCurrentHop(-1);
      setPlaying(false);
    }
  }, [active, resetVisuals]);

  if (!active || !neumannTerms || neumannTerms.length === 0) return null;

  const canPlay = currentHop < maxHop;
  const hopLabel =
    currentHop < 0
      ? "Ready"
      : currentHop === 0
        ? "Hop 0: Direct influence"
        : `Hop ${currentHop}: ${currentHop}-step feedback`;

  // Compute current cumulative ATM vol summary
  let totalATM = 0;
  if (currentHop >= 0) {
    const cumul = cumulValues.current[currentHop] ?? [];
    totalATM = cumul.reduce((s, v) => s + Math.abs(v), 0);
  }

  return (
    <div style={controlsStyle}>
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        {/* Play/Pause */}
        <button
          onClick={() => {
            if (!playing && currentHop >= maxHop) setCurrentHop(-1);
            setPlaying(!playing);
            if (!playing && currentHop < 0) setCurrentHop(0);
          }}
          style={ctrlBtn}
          title={playing ? "Pause" : "Play"}
        >
          {playing ? "\u23F8" : "\u25B6"}
        </button>

        {/* Step */}
        <button
          onClick={() => {
            setPlaying(false);
            setCurrentHop((p) => Math.min(p + 1, maxHop));
          }}
          disabled={currentHop >= maxHop}
          style={{ ...ctrlBtn, opacity: canPlay ? 1 : 0.4 }}
          title="Step forward"
        >
          {"\u23ED"}
        </button>

        {/* Reset */}
        <button
          onClick={() => {
            setPlaying(false);
            setCurrentHop(-1);
          }}
          style={ctrlBtn}
          title="Reset"
        >
          {"\u23EE"}
        </button>

        {/* Hop indicator */}
        <div style={{ fontSize: 11, color: "#e2e8f0", marginLeft: 4 }}>
          {hopLabel}
        </div>

        {/* Cumulative ATM vol */}
        {currentHop >= 0 && (
          <div style={{ fontSize: 10, color: "#94a3b8", marginLeft: 8 }}>
            ATM vol flow: {(totalATM * 100).toFixed(1)}%
          </div>
        )}

        <div style={{ flex: 1 }} />

        {/* Speed */}
        <label style={{ fontSize: 10, color: "#64748b" }}>Speed:</label>
        <input
          type="range"
          min={300}
          max={3000}
          step={100}
          value={3300 - speed}
          onChange={(e) => setSpeed(3300 - parseInt(e.target.value))}
          style={{ width: 70 }}
          title={`${speed}ms per hop`}
        />

        {/* Close */}
        <button onClick={onClose} style={{ ...ctrlBtn, color: "#64748b" }} title="Exit animation">
          {"\u2715"}
        </button>
      </div>

      {/* Hop progress dots */}
      <div style={{ display: "flex", gap: 4, marginTop: 4 }}>
        {Array.from({ length: maxHop + 1 }, (_, k) => (
          <div
            key={k}
            onClick={() => { setPlaying(false); setCurrentHop(k); }}
            style={{
              width: 12,
              height: 12,
              borderRadius: "50%",
              background: k <= currentHop
                ? HOP_COLORS[Math.min(k, HOP_COLORS.length - 1)]
                : "#334155",
              cursor: "pointer",
              border: k === currentHop ? "2px solid #e2e8f0" : "2px solid transparent",
              transition: "background 0.3s, border 0.3s",
            }}
            title={`Hop ${k}`}
          />
        ))}
      </div>
    </div>
  );
}

const controlsStyle: React.CSSProperties = {
  position: "absolute",
  top: 8,
  left: 8,
  right: 8,
  background: "rgba(15, 23, 42, 0.92)",
  borderRadius: 8,
  padding: "8px 12px",
  zIndex: 20,
  backdropFilter: "blur(8px)",
};

const ctrlBtn: React.CSSProperties = {
  width: 28,
  height: 28,
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  background: "#1e293b",
  color: "#e2e8f0",
  border: "1px solid #334155",
  borderRadius: 6,
  cursor: "pointer",
  fontSize: 14,
};
