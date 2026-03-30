import { useEffect, useState, useCallback, lazy, Suspense } from "react";
import { useEngine } from "./hooks/useEngine";
import GraphView from "./components/GraphView";
import ControlPanel from "./components/ControlPanel";
import QuoteTable from "./components/QuoteTable";
import MatrixEditor from "./components/MatrixEditor";
import ErrorBoundary from "./components/ErrorBoundary";
import SlideOutPanel from "./components/SlideOutPanel";
import ModellingOptionsPanel from "./components/ModellingOptionsPanel";
import ReferentialPanel from "./components/ReferentialPanel";
import FetchPriorsModal from "./components/FetchPriorsModal";

const SmileView = lazy(() => import("./components/SmileView"));
const ObservedNodeView = lazy(() => import("./components/ObservedNodeView"));
const InferredNodeView = lazy(() => import("./components/InferredNodeView"));
const SurfaceView = lazy(() => import("./components/SurfaceView"));
const PropagationViz = lazy(() => import("./components/PropagationViz"));
const PriorCalibrationView = lazy(
  () => import("./components/PriorCalibrationView")
);

type MainView = "graph" | "matrix" | "allSmiles";
type DetailTab = "smile" | "prior" | "propagation";

const Loading = () => (
  <div style={{ padding: 20, color: "#64748b" }}>Loading...</div>
);

function App() {
  const {
    quotes,
    solveResult,
    graphData,
    selectedNode,
    universe,
    fetchUniverse,
    fetchCatalog,
    fetchGraph,
    restoreQuotes,
    setSelectedNode,
  } = useEngine();

  const [mainView, setMainView] = useState<MainView>("graph");
  const [detailTab, setDetailTab] = useState<DetailTab>("smile");
  const [modellingOpen, setModellingOpen] = useState(false);
  const [referentialOpen, setReferentialOpen] = useState(false);
  const [fetchPriorsOpen, setFetchPriorsOpen] = useState(false);

  const toggleModelling = useCallback(() => setModellingOpen((v) => !v), []);
  const toggleReferential = useCallback(() => setReferentialOpen((v) => !v), []);
  const openFetchPriors = useCallback(() => setFetchPriorsOpen(true), []);
  const closeFetchPriors = useCallback(() => setFetchPriorsOpen(false), []);

  useEffect(() => {
    fetchCatalog();
    fetchUniverse();
    fetchGraph();
    restoreQuotes();
  }, [fetchCatalog, fetchUniverse, fetchGraph, restoreQuotes]);

  useEffect(() => {
    if (selectedNode) setDetailTab("smile");
  }, [selectedNode]);

  const selectedSmile = selectedNode
    ? solveResult?.nodes[selectedNode] ?? null
    : null;
  const selectedQuote = selectedNode ? quotes[selectedNode] ?? null : null;
  const isObserved = selectedSmile?.is_observed ?? true;
  const hasSelection = !!selectedNode;

  return (
    <div style={containerStyle}>
      {/* Top bar: title + inline controls */}
      <header style={headerStyle}>
        <h1 style={{ margin: 0, fontSize: 16, fontWeight: 700, flexShrink: 0 }}>
          Vol Marking
        </h1>
        <ControlPanel
          onToggleModelling={toggleModelling}
          onToggleReferential={toggleReferential}
          onFetchPriors={openFetchPriors}
        />
      </header>

      {/* Body: sidebar + main */}
      <div style={bodyStyle}>
        {/* Left sidebar: asset selection */}
        <div style={sidebarStyle}>
          <QuoteTable />
        </div>

        {/* Main content area */}
        <div style={mainAreaStyle}>
          {/* View toggle bar */}
          <div style={viewToggleBar}>
            {(
              [
                ["graph", "Graph"],
                ["matrix", "W Matrix"],
                ["allSmiles", "All Smiles"],
              ] as [MainView, string][]
            ).map(([key, label]) => (
              <button
                key={key}
                onClick={() => setMainView(key)}
                style={{
                  ...toggleBtn,
                  background: mainView === key ? "#334155" : "transparent",
                  color: mainView === key ? "#e2e8f0" : "#64748b",
                }}
              >
                {label}
              </button>
            ))}
          </div>

          {/* Content below toggle */}
          <div style={contentRow}>
            {/* Main panel: graph / matrix / all smiles */}
            <div
              style={{
                flex: hasSelection && mainView === "graph" ? "0 0 280px" : "1 1 100%",
                minWidth: 0,
                minHeight: 0,
                transition: "flex 0.2s ease",
              }}
            >
              <ErrorBoundary>
                <Suspense fallback={<Loading />}>
                  {mainView === "graph" && (
                    <GraphView
                      graphData={graphData}
                      solveResult={solveResult}
                      selectedNode={selectedNode}
                      onSelectNode={setSelectedNode}
                    />
                  )}
                  {mainView === "matrix" && <MatrixEditor />}
                  {mainView === "allSmiles" && (
                    <SurfaceView solveResult={solveResult} />
                  )}
                </Suspense>
              </ErrorBoundary>
            </div>

            {/* Detail panel: only shown when a node is selected AND on graph view */}
            {hasSelection && mainView === "graph" && (
              <div style={detailPanelStyle}>
                {/* Asset header */}
                {(() => {
                  const q = selectedQuote;
                  const asset = universe.find((a) => a.ticker === selectedNode);
                  const name = asset?.name ?? "";
                  const spot = q?.spot;
                  const prevSpot = q?.prev_spot;
                  const change = spot != null && prevSpot != null && prevSpot > 0
                    ? ((spot - prevSpot) / prevSpot * 100)
                    : null;
                  const changeColor = change != null ? (change >= 0 ? "#22c55e" : "#ef4444") : "#94a3b8";
                  return (
                    <div style={assetHeaderStyle}>
                      <span style={{ fontWeight: 700, color: "#e2e8f0", fontSize: 13 }}>
                        {selectedNode}
                      </span>
                      <span style={{ color: "#94a3b8", fontSize: 11 }}>{name}</span>
                      {spot != null && (
                        <span style={{ color: "#cbd5e1", fontSize: 11 }}>
                          Spot = {spot.toFixed(2)}
                        </span>
                      )}
                      {change != null && (
                        <span style={{ color: changeColor, fontSize: 11 }}>
                          {change >= 0 ? "+" : ""}{change.toFixed(2)}%
                        </span>
                      )}
                      {prevSpot != null && (
                        <span style={{ color: "#64748b", fontSize: 10 }}>
                          Prev Close = {prevSpot.toFixed(2)}
                        </span>
                      )}
                      {q?.expiry && (() => {
                        const d = new Date(q.expiry + "T00:00:00");
                        const months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
                        const fmt = `${d.getDate()}-${months[d.getMonth()]}-${String(d.getFullYear()).slice(2)}`;
                        return (
                          <span style={{ color: "#e2e8f0", fontSize: 11, fontWeight: 700 }}>
                            Expiry = {fmt}
                          </span>
                        );
                      })()}
                    </div>
                  );
                })()}

                {/* Detail tabs */}
                <div style={detailTabBar}>
                  {(
                    [
                      ["smile", "Smile"],
                      ["prior", "Prior"],
                      ["propagation", "Propagation"],
                    ] as [DetailTab, string][]
                  ).map(([key, label]) => (
                    <button
                      key={key}
                      onClick={() => setDetailTab(key)}
                      style={{
                        ...tabBtn,
                        borderBottom:
                          detailTab === key
                            ? "2px solid #6366f1"
                            : "2px solid transparent",
                        color: detailTab === key ? "#e2e8f0" : "#64748b",
                      }}
                    >
                      {label}
                    </button>
                  ))}
                  <button
                    onClick={() => setSelectedNode(null)}
                    style={{
                      ...tabBtn,
                      marginLeft: "auto",
                      color: "#475569",
                      fontSize: 14,
                      padding: "4px 8px",
                    }}
                    title="Close detail panel"
                  >
                    {"\u2715"}
                  </button>
                </div>

                {/* Detail content */}
                <div style={{ flex: 1, overflow: "auto", minHeight: 0 }}>
                  <ErrorBoundary>
                    <Suspense fallback={<Loading />}>
                      {detailTab === "smile" &&
                        selectedSmile &&
                        !isObserved && (
                          <InferredNodeView ticker={selectedNode!} />
                        )}
                      {detailTab === "smile" &&
                        (isObserved || !selectedSmile) && (
                          <ObservedNodeView
                            ticker={selectedNode!}
                            smileData={selectedSmile}
                            quoteData={selectedQuote}
                          />
                        )}
                      {detailTab === "prior" && <PriorCalibrationView />}
                      {detailTab === "propagation" && (
                        <PropagationViz
                          solveResult={solveResult}
                          graphData={graphData}
                          selectedNode={selectedNode}
                        />
                      )}
                    </Suspense>
                  </ErrorBoundary>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Slide-out panels */}
      <SlideOutPanel
        isOpen={modellingOpen}
        onClose={() => setModellingOpen(false)}
        title="Modelling Options"
      >
        <ModellingOptionsPanel />
      </SlideOutPanel>

      <SlideOutPanel
        isOpen={referentialOpen}
        onClose={() => setReferentialOpen(false)}
        title="Referential"
      >
        <ReferentialPanel />
      </SlideOutPanel>

      <FetchPriorsModal isOpen={fetchPriorsOpen} onClose={closeFetchPriors} />
    </div>
  );
}

const containerStyle: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  height: "100vh",
  background: "#0f172a",
  color: "#e2e8f0",
  fontFamily:
    "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
};

const headerStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 16,
  padding: "0 16px",
  borderBottom: "1px solid #1e293b",
  flexShrink: 0,
};

const bodyStyle: React.CSSProperties = {
  display: "flex",
  flex: 1,
  overflow: "hidden",
};

const sidebarStyle: React.CSSProperties = {
  width: 160,
  minWidth: 140,
  borderRight: "1px solid #1e293b",
  overflow: "hidden",
  display: "flex",
  flexDirection: "column",
  flexShrink: 0,
};

const mainAreaStyle: React.CSSProperties = {
  flex: 1,
  display: "flex",
  flexDirection: "column",
  minWidth: 0,
  overflow: "hidden",
};

const viewToggleBar: React.CSSProperties = {
  display: "flex",
  gap: 2,
  padding: "4px 8px",
  borderBottom: "1px solid #1e293b",
  flexShrink: 0,
};

const toggleBtn: React.CSSProperties = {
  fontSize: 11,
  fontWeight: 600,
  padding: "4px 14px",
  border: "1px solid #334155",
  borderRadius: 4,
  cursor: "pointer",
  background: "transparent",
  color: "#64748b",
};

const contentRow: React.CSSProperties = {
  display: "flex",
  flex: 1,
  overflow: "hidden",
  minHeight: 0,
};

const detailPanelStyle: React.CSSProperties = {
  flex: "1 1 0%",
  minWidth: 400,
  borderLeft: "1px solid #1e293b",
  display: "flex",
  flexDirection: "column",
  overflow: "hidden",
};

const assetHeaderStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 10,
  padding: "6px 14px",
  borderBottom: "1px solid #1e293b",
  background: "#0f172a",
  flexShrink: 0,
  flexWrap: "wrap",
};

const detailTabBar: React.CSSProperties = {
  display: "flex",
  borderBottom: "1px solid #1e293b",
  flexShrink: 0,
};

const tabBtn: React.CSSProperties = {
  flex: "0 0 auto",
  padding: "6px 14px",
  background: "none",
  border: "none",
  cursor: "pointer",
  fontSize: 11,
  fontWeight: 600,
};

export default App;
