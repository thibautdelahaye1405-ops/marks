import { useRef, useCallback, useEffect } from "react";

const DEFAULT_LABELS = ["Level (a)", "Wings (b)", "Skew (rho)", "Shift (m)", "Curvature (sig)"];

// Per-SVI-parameter slider ranges
const DEFAULT_RANGES: [number, number][] = [
  [-0.03, 0.03],    // Level (a): ~3 vol pt shift
  [-0.03, 0.03],    // Wings (b): wing steepness
  [-0.03, 0.03],    // Skew (rho): skew direction
  [-0.03, 0.03],    // Shift (m): horizontal smile shift
  [-0.03, 0.03],    // Curvature (sigma): smile curvature
];

function HoldButton({ onAction, children, style }: {
  onAction: () => void;
  children: React.ReactNode;
  style?: React.CSSProperties;
}) {
  const timer = useRef<ReturnType<typeof setInterval>>();
  const actionRef = useRef(onAction);
  actionRef.current = onAction;

  const stop = useCallback(() => { clearInterval(timer.current); timer.current = undefined; }, []);
  const start = useCallback(() => {
    actionRef.current();
    stop();
    timer.current = setInterval(() => actionRef.current(), 80);
  }, [stop]);

  useEffect(() => stop, [stop]);

  return (
    <button
      onMouseDown={start}
      onMouseUp={stop}
      onMouseLeave={stop}
      style={style}
    >
      {children}
    </button>
  );
}

interface Props {
  values: number[];
  onChange: (beta: number[]) => void;
  labels?: string[];
  ranges?: [number, number][];
}

export default function BetaSliders({
  values,
  onChange,
  labels = DEFAULT_LABELS,
  ranges = DEFAULT_RANGES,
}: Props) {
  const timerRef = useRef<ReturnType<typeof setTimeout>>();
  const valuesRef = useRef(values);
  const onChangeRef = useRef(onChange);
  valuesRef.current = values;
  onChangeRef.current = onChange;

  const handleSlider = useCallback(
    (index: number, val: number) => {
      const next = [...valuesRef.current];
      next[index] = val;
      valuesRef.current = next;
      clearTimeout(timerRef.current);
      timerRef.current = setTimeout(() => onChangeRef.current(next), 150);
    },
    []
  );

  const nudge = useCallback(
    (index: number, dir: number) => {
      const [min, max] = ranges[index] ?? [-0.01, 0.01];
      const step = (max - min) / 200;
      const cur = valuesRef.current[index];
      const val = Math.max(min, Math.min(max, cur + dir * step));
      const next = [...valuesRef.current];
      next[index] = val;
      valuesRef.current = next;
      onChangeRef.current(next);
    },
    [ranges]
  );

  return (
    <div style={{ padding: "4px 0" }}>
      <div style={gridStyle}>
        {values.map((v, i) => {
          const [min, max] = ranges[i] ?? [-0.01, 0.01];
          const step = (max - min) / 200;
          return (
            <div key={i} style={cellStyle}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 1 }}>
                <span style={{ fontSize: 9, color: "#94a3b8" }}>
                  {labels[i] ?? `beta_${i}`}
                </span>
                <span style={{ fontSize: 9, color: "#cbd5e1" }}>
                  {v.toFixed(4)}
                </span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 3 }}>
                <HoldButton
                  onAction={() => nudge(i, -1)}
                  style={nudgeBtn}
                >
                  {"\u2212"}
                </HoldButton>
                <input
                  type="range"
                  min={min}
                  max={max}
                  step={step}
                  value={Math.max(min, Math.min(max, v))}
                  onChange={(e) => handleSlider(i, parseFloat(e.target.value))}
                  style={{ flex: 1, accentColor: "#6366f1", height: 3 }}
                />
                <HoldButton
                  onAction={() => nudge(i, 1)}
                  style={nudgeBtn}
                >
                  +
                </HoldButton>
              </div>
            </div>
          );
        })}
      </div>
      <button
        onClick={() => onChange(values.map(() => 0))}
        style={{
          fontSize: 9,
          color: "#64748b",
          background: "none",
          border: "1px solid #334155",
          borderRadius: 3,
          padding: "1px 6px",
          cursor: "pointer",
          marginTop: 4,
          float: "right",
        }}
      >
        Reset all
      </button>
    </div>
  );
}

const gridStyle: React.CSSProperties = {
  display: "grid",
  gridTemplateColumns: "1fr 1fr",
  gap: "4px 12px",
};

const cellStyle: React.CSSProperties = {
  minWidth: 0,
};

const nudgeBtn: React.CSSProperties = {
  width: 18,
  height: 18,
  padding: 0,
  fontSize: 12,
  lineHeight: "16px",
  textAlign: "center",
  background: "#1e293b",
  color: "#94a3b8",
  border: "1px solid #334155",
  borderRadius: 3,
  cursor: "pointer",
  flexShrink: 0,
};
