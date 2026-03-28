import { useRef, useCallback, useEffect, useMemo } from "react";

interface SviParams {
  a: number;
  b: number;
  rho: number;
  m: number;
  sigma: number;
}

interface Props {
  values: SviParams;
  baseValues?: SviParams;  // fitted values — sliders center around these
  onChange: (params: SviParams) => void;
  onReset?: () => void;
}

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
    <button onMouseDown={start} onMouseUp={stop} onMouseLeave={stop} style={style}>
      {children}
    </button>
  );
}

interface ParamDef {
  key: keyof SviParams;
  label: string;
  hardMin: number;   // absolute minimum (e.g. a ≥ 0)
  hardMax: number;   // absolute maximum (e.g. rho ≤ 0.999)
  halfWidth: number;  // slider range = base ± halfWidth
  nSteps: number;    // number of slider steps across the full range
  fmt: (v: number) => string;
}

const PARAM_DEFS: ParamDef[] = [
  { key: "a",     label: "a (level)",     hardMin: 0,      hardMax: 1.0,   halfWidth: 0.01,  nSteps: 200, fmt: (v) => v.toFixed(5) },
  { key: "b",     label: "b (wings)",     hardMin: 0,      hardMax: 1.0,   halfWidth: 0.1,   nSteps: 200, fmt: (v) => v.toFixed(4) },
  { key: "rho",   label: "\u03C1 (skew)", hardMin: -0.999, hardMax: 0.999, halfWidth: 0.2,   nSteps: 200, fmt: (v) => v.toFixed(3) },
  { key: "m",     label: "m (shift)",     hardMin: -0.5,   hardMax: 0.5,   halfWidth: 0.05,  nSteps: 200, fmt: (v) => v.toFixed(4) },
  { key: "sigma", label: "\u03C3 (curv)", hardMin: 0.001,  hardMax: 2.0,   halfWidth: 0.1,   nSteps: 200, fmt: (v) => v.toFixed(4) },
];

export default function SviSliders({ values, baseValues, onChange, onReset }: Props) {
  const base = baseValues ?? values;

  // Compute slider ranges centered on the base (fitted) values,
  // expanded to always include the current value
  const ranges = useMemo(() => {
    return PARAM_DEFS.map((def) => {
      const center = base[def.key];
      const cur = values[def.key];
      const lo = Math.max(def.hardMin, Math.min(center - def.halfWidth, cur - def.halfWidth * 0.1));
      const hi = Math.min(def.hardMax, Math.max(center + def.halfWidth, cur + def.halfWidth * 0.1));
      const step = (hi - lo) / def.nSteps;
      return { ...def, min: lo, max: hi, step: Math.max(step, 1e-8) };
    });
  }, [base, values]);

  const valRef = useRef(values);
  const onChangeRef = useRef(onChange);
  valRef.current = values;
  onChangeRef.current = onChange;

  const nudge = useCallback((index: number, dir: number) => {
    const r = ranges[index];
    if (!r) return;
    const cur = valRef.current[r.key];
    const next = Math.max(r.min, Math.min(r.max, cur + dir * r.step));
    const updated = { ...valRef.current, [r.key]: next };
    valRef.current = updated;
    onChangeRef.current(updated);
  }, [ranges]);

  const handleSlider = useCallback((key: keyof SviParams, val: number) => {
    const updated = { ...valRef.current, [key]: val };
    valRef.current = updated;
    onChangeRef.current(updated);
  }, []);

  return (
    <div style={{ padding: "4px 0" }}>
      <div style={gridStyle}>
        {ranges.map((r, idx) => {
          const v = values[r.key];
          return (
            <div key={r.key} style={cellStyle}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 1 }}>
                <span style={{ fontSize: 9, color: "#94a3b8" }}>{r.label}</span>
                <span style={{ fontSize: 9, color: "#cbd5e1" }}>{r.fmt(v)}</span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 3 }}>
                <HoldButton onAction={() => nudge(idx, -1)} style={nudgeBtn}>
                  {"\u2212"}
                </HoldButton>
                <input
                  type="range"
                  min={r.min}
                  max={r.max}
                  step={r.step}
                  value={Math.max(r.min, Math.min(r.max, v))}
                  onChange={(e) => handleSlider(r.key, parseFloat(e.target.value))}
                  style={{ flex: 1, accentColor: "#6366f1", height: 3 }}
                />
                <HoldButton onAction={() => nudge(idx, 1)} style={nudgeBtn}>
                  +
                </HoldButton>
              </div>
            </div>
          );
        })}
      </div>
      {onReset && (
        <button onClick={onReset} style={resetBtn}>
          Reset to fit
        </button>
      )}
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

const resetBtn: React.CSSProperties = {
  fontSize: 9,
  color: "#64748b",
  background: "none",
  border: "1px solid #334155",
  borderRadius: 3,
  padding: "1px 6px",
  cursor: "pointer",
  marginTop: 4,
  float: "right",
};
