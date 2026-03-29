import { useRef, useCallback, useEffect, useMemo } from "react";

export interface LqdTraderParams {
  min_iv: number;      // Min IV Level
  atm_skew: number;    // ATM Skew
  atm_curv: number;    // ATM Curvature
  put_slope: number;   // Put Wing Slope
  call_slope: number;  // Call Wing Slope
  shoulder: number;    // Shoulder
}

interface Props {
  values: LqdTraderParams;
  baseValues?: LqdTraderParams;  // fitted values — sliders center around these
  onChange: (params: LqdTraderParams) => void;
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
  key: keyof LqdTraderParams;
  label: string;
  hardMin: number;
  hardMax: number;
  halfWidth: number;
  nSteps: number;
  fmt: (v: number) => string;
}

const PARAM_DEFS: ParamDef[] = [
  { key: "min_iv",     label: "Put Wing (p)",      hardMin: -3.0,  hardMax: 3.0,   halfWidth: 0.5,   nSteps: 200, fmt: (v) => v.toFixed(3) },
  { key: "atm_skew",   label: "Call Wing (c)",     hardMin: -3.0,  hardMax: 3.0,   halfWidth: 0.5,   nSteps: 200, fmt: (v) => v.toFixed(3) },
  { key: "atm_curv",   label: "Belly (m)",         hardMin: -3.0,  hardMax: 3.0,   halfWidth: 0.5,   nSteps: 200, fmt: (v) => v.toFixed(3) },
  { key: "put_slope",  label: "Skew (s)",          hardMin: -3.0,  hardMax: 3.0,   halfWidth: 0.5,   nSteps: 200, fmt: (v) => v.toFixed(3) },
  { key: "call_slope", label: "Kurtosis (\u03BA)", hardMin: -3.0,  hardMax: 3.0,   halfWidth: 0.5,   nSteps: 200, fmt: (v) => v.toFixed(3) },
  { key: "shoulder",   label: "Shoulder (h)",       hardMin: -3.0,  hardMax: 3.0,   halfWidth: 0.5,   nSteps: 200, fmt: (v) => v.toFixed(3) },
];

export default function LqdSliders({ values, baseValues, onChange, onReset }: Props) {
  const base = baseValues ?? values;

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

  const handleSlider = useCallback((key: keyof LqdTraderParams, val: number) => {
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
