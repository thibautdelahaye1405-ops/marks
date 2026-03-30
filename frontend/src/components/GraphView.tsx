import { useEffect, useRef, useCallback } from "react";
import cytoscape from "cytoscape";
import type { GraphData, SolveResponse } from "../types";
import { observedColor, inferredColor, weightToOpacity, weightToWidth } from "../utils/colors";
import { isCompoundKey, splitNodeKey, expiryLabel } from "../utils/nodeKey";

interface Props {
  graphData: GraphData | null;
  solveResult: SolveResponse | null;
  selectedNode: string | null;
  onSelectNode: (ticker: string | null) => void;
}

/**
 * Compute total influence score for each node from W alone.
 * Score(j) = sum_i W[i][j]  (how much j is used as a source by others).
 * Resolves the full (I-W)^{-1} propagation for a proper ranking.
 */
function computeInfluenceRanks(W: number[][]): number[] {
  const N = W.length;
  // Total outgoing influence: column sums of W
  const colSums = new Array(N).fill(0);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      colSums[j] += W[i][j];
    }
  }
  const maxCol = Math.max(...colSums, 1e-9);
  return colSums.map((s) => s / maxCol); // 0..1, 1 = most influential
}

/**
 * Build deterministic preset positions.
 * Y: most influential at top, leaves at bottom.
 * X: spread horizontally within each tier to avoid overlap.
 */
function computePositions(
  tickers: string[],
  ranks: number[],
  width: number,
  height: number
): Record<string, { x: number; y: number }> {
  // Sort by rank descending to assign y layers
  const indexed = tickers.map((t, i) => ({ ticker: t, rank: ranks[i], idx: i }));
  indexed.sort((a, b) => b.rank - a.rank);

  const pad = 40;
  const usableW = width - pad * 2;
  const usableH = height - pad * 2;

  const positions: Record<string, { x: number; y: number }> = {};

  // Group into tiers by quantising rank into bins
  const nTiers = Math.min(tickers.length, 5);
  const tiers: typeof indexed[] = Array.from({ length: nTiers }, () => []);
  for (const node of indexed) {
    const tier = Math.min(
      nTiers - 1,
      Math.floor((1 - node.rank) * nTiers)
    );
    tiers[tier].push(node);
  }

  for (let t = 0; t < nTiers; t++) {
    const row = tiers[t];
    if (row.length === 0) continue;
    const y = pad + (t / Math.max(nTiers - 1, 1)) * usableH;
    for (let i = 0; i < row.length; i++) {
      const x =
        row.length === 1
          ? width / 2
          : pad + (i / (row.length - 1)) * usableW;
      positions[row[i].ticker] = { x, y };
    }
  }

  return positions;
}

export default function GraphView({
  graphData,
  solveResult,
  selectedNode,
  onSelectNode,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<cytoscape.Core | null>(null);
  const selectedRef = useRef(selectedNode);
  const onSelectRef = useRef(onSelectNode);
  selectedRef.current = selectedNode;
  onSelectRef.current = onSelectNode;

  // Rebuild graph only when graphData (W matrix) changes
  useEffect(() => {
    if (!containerRef.current || !graphData) return;

    const { tickers, W, assets } = graphData;
    const ranks = computeInfluenceRanks(W);
    const rect = containerRef.current.getBoundingClientRect();
    const positions = computePositions(
      tickers,
      ranks,
      Math.max(rect.width, 400),
      Math.max(rect.height, 300)
    );

    const elements: cytoscape.ElementDefinition[] = [];

    // Nodes — initial neutral styling; colors updated by separate effect
    // In multi-expiry mode, tickers.length > assets.length (multiple nodes per asset)
    const assetMap = Object.fromEntries(assets.map((a) => [a.ticker, a]));
    for (let i = 0; i < tickers.length; i++) {
      const nodeKey = tickers[i];
      let label: string;
      if (isCompoundKey(nodeKey)) {
        const { ticker: tk, expiry } = splitNodeKey(nodeKey);
        label = `${tk}\n${expiryLabel(expiry)}`;
      } else {
        label = nodeKey;
      }
      elements.push({
        data: {
          id: nodeKey,
          label,
          color: observedColor(0.3),
          size: 60 + ranks[i] * 40,
        },
        position: positions[nodeKey],
      });
    }

    // Edges — in multi-expiry mode, only show edges above threshold
    // and limit to avoid visual clutter with large tensors
    const threshold = tickers.length > 20 ? 0.03 : 0.01;
    for (let i = 0; i < tickers.length; i++) {
      if (!W[i]) continue;
      for (let j = 0; j < tickers.length; j++) {
        if (i === j || W[i][j] < threshold) continue;
        elements.push({
          data: {
            id: `${tickers[j]}->${tickers[i]}`,
            source: tickers[j],
            target: tickers[i],
            weight: W[i][j],
            weightLabel: W[i][j].toFixed(2),
            width: weightToWidth(W[i][j]),
            opacity: weightToOpacity(W[i][j]),
          },
        });
      }
    }

    if (cyRef.current) cyRef.current.destroy();

    const cy = cytoscape({
      container: containerRef.current,
      elements,
      style: [
        {
          selector: "node",
          style: {
            "background-color": "data(color)",
            width: "data(size)",
            height: "data(size)",
            "border-width": 1,
            "border-color": "rgba(255,255,255,0.3)",
            label: "data(label)",
            "font-size": "10px",
            "text-valign": "center",
            "text-halign": "center",
            "text-wrap": "wrap",
            color: "#fff",
            "font-weight": "700",
            "text-outline-color": "data(color)",
            "text-outline-opacity": 0.6,
            "text-outline-width": 0,
          },
        },
        {
          selector: "node.selected",
          style: {
            "border-width": 3,
            "border-color": "#fff",
          },
        },
        {
          selector: "node.hover",
          style: {
            "border-width": 2.5,
            "border-color": "#fff",
          },
        },
        {
          selector: "edge",
          style: {
            width: "data(width)",
            "line-color": "#5b6b8a",
            "target-arrow-color": "#5b6b8a",
            "target-arrow-shape": "triangle",
            "arrow-scale": 0.7,
            "curve-style": "bezier",
            opacity: "data(opacity)",
            label: "",
          },
        },
        {
          selector: "edge.highlighted",
          style: {
            "line-color": "#e2e8f0",
            "target-arrow-color": "#e2e8f0",
            opacity: 1,
            "z-index": 10,
            label: "data(weightLabel)",
            "font-size": "9px",
            color: "#e2e8f0",
            "text-background-color": "#0f172a",
            "text-background-opacity": 0.85,
            "text-background-padding": "2px",
            "text-rotation": "autorotate",
          } as any,
        },
      ],
      layout: { name: "preset" },
      // Lock positions so they don't change
      autoungrabify: false,
    });

    cy.on("tap", "node", (e) => {
      const ticker = e.target.id();
      onSelectRef.current(ticker === selectedRef.current ? null : ticker);
    });

    cy.on("tap", (e) => {
      if (e.target === cy) onSelectRef.current(null);
    });

    cy.on("mouseover", "node", (e) => {
      e.target.addClass("hover");
      e.target.connectedEdges().addClass("highlighted");
    });
    cy.on("mouseout", "node", (e) => {
      e.target.removeClass("hover");
      e.target.connectedEdges().removeClass("highlighted");
    });

    cyRef.current = cy;

    return () => { cy.destroy(); };
  }, [graphData]);

  // Update node colors/sizes from solveResult without relayout
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy || !graphData) return;

    const { tickers } = graphData;
    const infScores = solveResult?.influence_scores ?? null;
    const w2Dists = solveResult?.wasserstein_distances ?? {};

    let influenceRanks: number[] = tickers.map(() => 0.5);
    if (infScores) {
      const sorted = [...infScores].sort((a, b) => b - a);
      influenceRanks = infScores.map((s) => {
        const rank = sorted.indexOf(s);
        return 1 - rank / Math.max(sorted.length - 1, 1);
      });
    }

    const w2Values = tickers.map((t) => w2Dists[t] ?? 0);
    const maxW2 = Math.max(...w2Values, 1e-6);

    for (let i = 0; i < tickers.length; i++) {
      const ticker = tickers[i];
      const node = cy.getElementById(ticker);
      if (node.empty()) continue;

      const isObserved = solveResult?.nodes[ticker]?.is_observed ?? true;
      const w2Intensity = Math.min(w2Values[i] / maxW2, 1);
      const color = isObserved
        ? observedColor(w2Intensity)
        : inferredColor(w2Intensity);

      node.data("color", color);
      node.data("size", 60 + influenceRanks[i] * 40);
    }
  }, [solveResult, graphData]);

  // Update selection highlight
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.nodes().removeClass("selected");
    if (selectedNode) {
      cy.getElementById(selectedNode).addClass("selected");
    }
  }, [selectedNode]);

  // Resize when container changes
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const observer = new ResizeObserver(() => {
      cyRef.current?.resize();
      cyRef.current?.fit(undefined, 20);
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return (
    <div style={{ width: "100%", height: "100%", position: "relative" }}>
      <div
        ref={containerRef}
        style={{
          width: "100%",
          height: "100%",
          background: "#0f172a",
          borderRadius: 8,
        }}
      />
      {/* Legend */}
      <div
        style={{
          position: "absolute",
          bottom: 12,
          left: 12,
          background: "rgba(15,23,42,0.85)",
          borderRadius: 6,
          padding: "8px 12px",
          fontSize: 10,
          color: "#94a3b8",
          display: "flex",
          gap: 16,
        }}
      >
        <span>
          <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: "50%", background: observedColor(0.7), marginRight: 4 }} />
          Observed
        </span>
        <span>
          <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: "50%", background: inferredColor(0.7), marginRight: 4 }} />
          Inferred
        </span>
        <span style={{ color: "#475569" }}>
          Top = most influential | Hover for W coefficients
        </span>
      </div>
    </div>
  );
}
