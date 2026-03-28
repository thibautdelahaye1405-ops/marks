# UX & Frontend Guide

## Architecture
- React 18 + TypeScript + Vite (port 5173, proxies /api to backend port 5000)
- Zustand store (`useEngine.ts`) — single source of truth for all app state
- Plotly.js via custom `Plot.tsx` wrapper (uses `Plotly.react` directly, not react-plotly)
- Cytoscape.js for graph visualisation

## Layout
- **Top bar**: Fetch Priors (modal), Fetch Snapshot, Modelling (slide-out), Referential (slide-out), status
- **Left sidebar** (160px): Propagate button, asset list with All/None + checkboxes
- **Main area**: Graph | W Matrix | All Smiles toggle
- **Detail panel** (appears on node selection, graph shrinks to 280px): Smile | Prior | Propagation tabs

## Key Components
- `SlideOutPanel.tsx` — reusable right-side drawer with backdrop
- `FetchPriorsModal.tsx` — per-asset source selection (fetch/file/skip)
- `SviSliders.tsx` — 5 SVI param sliders, centered on fitted values, hold-to-repeat +/- buttons via `HoldButton`
- `SmileView.tsx` — IV chart with click-to-exclude, double-click-to-add, SVI overrides
- `PriorCalibrationView.tsx` — prior smile + IV/CDF/LQD tabs, quote editing, save button
- `ObservedNodeView.tsx` / `InferredNodeView.tsx` — node detail with view mode toggle
- `DistributionTripleView.tsx` — IV left column, CDF+LQD right column, fixed axes (CDF: [-4,4], LQD: [0,1])
- `GraphView.tsx` — deterministic preset layout (top=influential), hover shows W coefficients, ResizeObserver
- `PropagationViz.tsx` — ego-graph centered on selected node, Viridis color scale, total influence values

## Patterns
- Double-click detection: mousedown timing in `Plot.tsx` (Plotly swallows dblclick)
- Hold-to-repeat buttons: `HoldButton` component with `actionRef` pattern for stale closure avoidance
- Slider ranges: centered on `baseValues` prop with per-param half-widths
- Dark theme: #0f172a (bg), #1e293b (panels), #334155 (borders), #e2e8f0/#94a3b8/#cbd5e1 (text)
- Legend positions: IV top-right, CDF top-left, LQD top-center

## State Flow
- `fit()` — lightweight, no propagation. Called on: fetchSnapshot, toggleObserved, toggleQuotePoint, resetExclusions, addQuotePoint, removeAddedQuote, resetAdditions
- `propagate()` — full graph solve. Called ONLY by the Propagate button
- `_propagateSeq` counter ensures only latest result is applied (race condition guard)
