import { useEffect, useRef } from "react";
import Plotly from "plotly.js-dist-min";

interface PlotProps {
  data: Plotly.Data[];
  layout?: Partial<Plotly.Layout>;
  config?: Partial<Plotly.Config>;
  style?: React.CSSProperties;
  useResizeHandler?: boolean;
  onClick?: (event: any) => void;
  onDoubleClick?: (coords: { x: number; y: number }) => void;
}

function pixelToData(
  el: HTMLDivElement,
  clientX: number,
  clientY: number
): { x: number; y: number } | null {
  const fullLayout = (el as any)._fullLayout;
  if (!fullLayout) return null;

  const xaxis = fullLayout.xaxis;
  const yaxis = fullLayout.yaxis;
  if (!xaxis?.range || !yaxis?.range) return null;

  const rect = el.getBoundingClientRect();
  const ml = fullLayout.margin?.l ?? 0;
  const mt = fullLayout.margin?.t ?? 0;
  const plotW = (fullLayout.width ?? rect.width) - ml - (fullLayout.margin?.r ?? 0);
  const plotH = (fullLayout.height ?? rect.height) - mt - (fullLayout.margin?.b ?? 0);

  const px = clientX - rect.left - ml;
  const py = clientY - rect.top - mt;

  if (plotW <= 0 || plotH <= 0 || px < 0 || px > plotW || py < 0 || py > plotH) return null;

  const [xMin, xMax] = xaxis.range;
  const [yMin, yMax] = yaxis.range;
  const x = xMin + (px / plotW) * (xMax - xMin);
  const y = yMax - (py / plotH) * (yMax - yMin);

  return { x, y };
}

export default function Plot({ data, layout, config, style, onClick, onDoubleClick }: PlotProps) {
  const ref = useRef<HTMLDivElement>(null);
  const onDblRef = useRef(onDoubleClick);
  onDblRef.current = onDoubleClick;

  // Track rapid clicks to detect double-click (avoids Plotly event interception)
  const lastClickRef = useRef<{ time: number; x: number; y: number }>({ time: 0, x: 0, y: 0 });

  useEffect(() => {
    if (!ref.current) return;

    Plotly.react(ref.current, data, layout as Plotly.Layout, config);

    const el = ref.current as any;
    if (onClick) {
      el.on("plotly_click", onClick);
    }

    return () => {
      if (ref.current) {
        const el2 = ref.current as any;
        if (onClick) el2.removeAllListeners?.("plotly_click");
        Plotly.purge(ref.current);
      }
    };
  }, [data, layout, config, onClick]);

  // Detect double-click via two rapid mousedowns (bypasses Plotly event handling)
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const handler = (event: MouseEvent) => {
      if (!onDblRef.current) return;
      const now = Date.now();
      const last = lastClickRef.current;
      const dx = Math.abs(event.clientX - last.x);
      const dy = Math.abs(event.clientY - last.y);
      if (now - last.time < 400 && dx < 10 && dy < 10) {
        // Double-click detected
        const coords = pixelToData(el, event.clientX, event.clientY);
        if (coords) {
          onDblRef.current(coords);
        }
        lastClickRef.current = { time: 0, x: 0, y: 0 }; // reset
      } else {
        lastClickRef.current = { time: now, x: event.clientX, y: event.clientY };
      }
    };
    el.addEventListener("mousedown", handler, true);
    return () => el.removeEventListener("mousedown", handler, true);
  }, []);

  useEffect(() => {
    if (!ref.current) return;
    const observer = new ResizeObserver(() => {
      if (ref.current) Plotly.Plots.resize(ref.current);
    });
    observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);

  return <div ref={ref} style={style} />;
}
