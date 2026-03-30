/**
 * Node key utilities for multi-expiry support.
 *
 * A node in the influence graph is identified by a compound key "TICKER:EXPIRY"
 * (e.g. "SPY:2025-04-17"). In single-maturity mode, a plain ticker string is
 * also accepted.
 */

const SEPARATOR = ":";

export function makeNodeKey(ticker: string, expiry: string): string {
  return `${ticker}${SEPARATOR}${expiry}`;
}

export function splitNodeKey(key: string): { ticker: string; expiry: string } {
  const idx = key.lastIndexOf(SEPARATOR);
  if (idx === -1) return { ticker: key, expiry: "" };
  return { ticker: key.substring(0, idx), expiry: key.substring(idx + 1) };
}

export function isCompoundKey(key: string): boolean {
  return key.includes(SEPARATOR);
}

export function tickerOf(key: string): string {
  return splitNodeKey(key).ticker;
}

export function expiryOf(key: string): string {
  return splitNodeKey(key).expiry;
}

export function expiryLabel(expiry: string, T?: number): string {
  if (!expiry) return "";
  try {
    const dt = new Date(expiry + "T00:00:00");
    const month = dt.toLocaleString("en-US", { month: "short" });
    const day = dt.getDate();
    let label = `${month} ${day}`;
    if (T !== undefined) {
      const days = Math.round(T * 365);
      label += ` (${days}d)`;
    }
    return label;
  } catch {
    return expiry;
  }
}

export function groupByTicker(nodeKeys: string[]): Record<string, string[]> {
  const groups: Record<string, string[]> = {};
  for (const key of nodeKeys) {
    const tk = tickerOf(key);
    if (!groups[tk]) groups[tk] = [];
    groups[tk].push(key);
  }
  return groups;
}

export function uniqueTickers(nodeKeys: string[]): string[] {
  const seen = new Set<string>();
  const result: string[] = [];
  for (const key of nodeKeys) {
    const tk = tickerOf(key);
    if (!seen.has(tk)) {
      seen.add(tk);
      result.push(tk);
    }
  }
  return result;
}
