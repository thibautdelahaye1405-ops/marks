import { useEffect, useRef } from "react";
import cytoscape from "cytoscape";
import type { SolveResponse, GraphData } from "../types";
import { observedColor, inferredColor } from "../utils/colors";

interface Props {
  solveResult: SolveResponse | null;
  graphData: GraphData | null;
  selectedNode: string | null;
}

/** Viridis-like color from t in [0,1]: dark purple → teal → yellow. */
function viridis(t: number): string {
  const c = Math.max(0, Math.min(1, t));
  const r = Math.round(68 + c * (253 - 68));
  const g = Math.round(1 + c * (231 - 1));
  const b = Math.round(84 + (c < 0.5 ? c * 2 * (168 - 84) : (168 - (c - 0.5) * 2 * (168 - 37))));
  return `rgb(${r}, ${g}, ${b})`;
}

/**
 * Ego-graph showing total propagation influence FROM the selected (observed)
 * asset TO every unobserved asset it affects.
 *
 * Edge value = P[i,j]: total influence (direct + multi-hop) — how much
 * unobserved node i moves when observed node j moves by 1.
 */
export default function PropagationViz({
  solveResult,
  graphData,
  selectedNode,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<cytoscape.Core | null>(null);

  useEffect(() => {
    if (!containerRef.current || !solveResult || !selectedNode) return;

    const P = solveResult.propagation_matrix;
    if (!P) return;

    const tickers = solveResult.tickers;
    const observed = tickers.filter((t) => solveResult.nodes[t]?.is_observed);
    const unobserved = tickers.filter((t) => !solveResult.nodes[t]?.is_observed);

    // Selected node must be observed to be a propagation source
    const obsIdx = observed.indexOf(selectedNode);
    if (obsIdx < 0) return;

    // P shape: [N_unobs][N_obs] — extract column for selectedNode
    const threshold = 0.005;
    const edges: { target: string; value: number }[] = [];
    for (let i = 0; i < unobserved.length; i++) {
      const val = P[i]?.[obsIdx] ?? 0;
      if (val > threshold) {
        edges.push({ target: unobserved[i], value: val });
      }
    }

    if (edges.length === 0) {
      if (cyRef.current) { cyRef.current.destroy(); cyRef.current = null; }
      return;
    }

    // Scale for color
    const maxVal = Math.max(...edges.map((e) => e.value), 0.01);

    // Build elements
    const elements: cytoscape.ElementDefinition[] = [];

    // Center node (observed / source)
    elements.push({
      data: {
        id: selectedNode,
        label: selectedNode,
        color: observedColor(0.8),
        size: 30,
        borderWidth: 3,
        borderColor: "#6366f1",
      },
    });

    // Target nodes + edges
    for (const e of edges) {
      const intensity = e.value / maxVal;
      elements.push({
        data: {
          id: e.target,
          label: e.target,
          color: inferredColor(intensity),
          size: 14 + intensity * 10,
          borderWidth: 1,
          borderColor: "rgba(255,255,255,0.15)",
        },
      });
      elements.push({
        data: {
          id: `${selectedNode}->${e.target}`,
          source: selectedNode,
          target: e.target,
          weight: e.value,
          label: e.value.toFixed(2),
          width: 1.5 + (e.value / maxVal) * 5,
          color: viridis(e.value / maxVal),
        },
      });
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
            "border-width": "data(borderWidth)",
            "border-color": "data(borderColor)",
            label: "data(label)",
            "font-size": "10px",
            "text-valign": "bottom",
            "text-halign": "center",
            "text-margin-y": 5,
            color: "#cbd5e1",
            "font-weight": "600",
          },
        },
        {
          selector: "edge",
          style: {
            width: "data(width)",
            "line-color": "data(color)",
            "target-arrow-color": "data(color)",
            "target-arrow-shape": "triangle",
            "arrow-scale": 0.8,
            "curve-style": "straight",
            label: "data(label)",
            "font-size": "10px",
            color: "#e2e8f0",
            "text-background-color": "#0f172a",
            "text-background-opacity": 0.85,
            "text-background-padding": "2px",
            "text-rotation": "autorotate",
          } as any,
        },
      ],
      layout: {
        name: "concentric",
        concentric: (node: any) => (node.id() === selectedNode ? 2 : 1),
        levelWidth: () => 1,
        minNodeSpacing: 60,
        animate: true,
        animationDuration: 400,
      },
    });

    cyRef.current = cy;
    return () => { cy.destroy(); };
  }, [solveResult, selectedNode]);

  if (!selectedNode) {
    return (
      <div style={msgStyle}>Select a node to see its propagation</div>
    );
  }

  const isObs = solveResult?.nodes[selectedNode]?.is_observed;
  if (solveResult && !isObs) {
    return (
      <div style={msgStyle}>
        <b>{selectedNode}</b> is inferred — select an observed node to see outward propagation
      </div>
    );
  }

  if (!solveResult?.propagation_matrix) {
    return <div style={msgStyle}>Run Propagate first</div>;
  }

  return (
    <div style={{ height: "100%", display: "flex", flexDirection: "column" }}>
      <div style={{ padding: "8px 12px 4px", fontSize: 11, color: "#94a3b8" }}>
        Total influence from <b style={{ color: "#e2e8f0" }}>{selectedNode}</b> through the network
        <span style={{ marginLeft: 8, fontSize: 10, color: "#475569" }}>
          value = full propagation (direct + multi-hop)
        </span>
      </div>
      <div
        ref={containerRef}
        style={{
          flex: 1,
          minHeight: 200,
          background: "#0f172a",
          borderRadius: 6,
          margin: "0 8px 8px",
        }}
      />
    </div>
  );
}

const msgStyle: React.CSSProperties = {
  padding: 20,
  color: "#94a3b8",
  textAlign: "center",
  fontSize: 12,
};
