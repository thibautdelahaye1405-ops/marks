export const SECTOR_COLORS: Record<string, string> = {
  Index: "#6366f1",
  Technology: "#06b6d4",
  Consumer: "#f59e0b",
  Financials: "#10b981",
};

export function sectorColor(sector: string): string {
  return SECTOR_COLORS[sector] ?? "#94a3b8";
}

export function weightToOpacity(w: number): number {
  return Math.min(0.3 + w * 1.5, 1.0);
}

export function weightToWidth(w: number): number {
  return 0.5 + w * 8;
}

/**
 * Observed node color: blue with intensity proportional to Wasserstein distance.
 * intensity in [0, 1]: 0 = faint, 1 = full saturation.
 */
export function observedColor(intensity: number): string {
  const t = Math.max(0.15, Math.min(1, intensity));
  // Interpolate from faint blue-gray to vivid blue
  const r = Math.round(100 - t * 41);   // 100 → 59
  const g = Math.round(140 - t * 10);   // 140 → 130
  const b = Math.round(200 + t * 46);   // 200 → 246
  return `rgb(${r}, ${g}, ${b})`;
}

/**
 * Inferred node color: orange with intensity proportional to Wasserstein distance.
 */
export function inferredColor(intensity: number): string {
  const t = Math.max(0.15, Math.min(1, intensity));
  const r = Math.round(180 + t * 69);   // 180 → 249
  const g = Math.round(140 - t * 35);   // 140 → 105 (more saturated orange)
  const b = Math.round(80 - t * 57);    // 80 → 23
  return `rgb(${r}, ${g}, ${b})`;
}

/** Diverging color scale: negative=red, zero=white, positive=blue */
export function divergingColor(value: number, maxAbs: number): string {
  const t = Math.max(-1, Math.min(1, value / (maxAbs || 1)));
  if (t >= 0) {
    const r = Math.round(255 * (1 - t));
    const g = Math.round(255 * (1 - t * 0.6));
    return `rgb(${r}, ${g}, 255)`;
  } else {
    const g = Math.round(255 * (1 + t * 0.6));
    const b = Math.round(255 * (1 + t));
    return `rgb(255, ${g}, ${b})`;
  }
}
