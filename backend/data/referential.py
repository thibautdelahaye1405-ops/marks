"""
Asset universe referential data.

Manages the full catalog and the active (selected) universe.
Persists:
  - selections/{name}.json  — saved universe selections
  - catalog_custom.json     — user-added tickers that validated on Yahoo
"""
import json
from pathlib import Path
from typing import List, Dict, Optional
from ..config import AssetDef, CATALOG, CATALOG_MAP, DEFAULT_ACTIVE_TICKERS

PROJECT_ROOT = Path(__file__).parent.parent.parent
SELECTIONS_DIR = PROJECT_ROOT / "selections"
CUSTOM_CATALOG_PATH = PROJECT_ROOT / "catalog_custom.json"
DEFAULT_SELECTION = "default"


def _ensure_dirs():
    SELECTIONS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Custom catalog persistence
# ---------------------------------------------------------------------------

def _load_custom_catalog():
    """Load user-added tickers from disk and merge into in-memory catalog."""
    if not CUSTOM_CATALOG_PATH.exists():
        return
    try:
        data = json.loads(CUSTOM_CATALOG_PATH.read_text())
        for entry in data:
            ticker = entry["ticker"]
            if ticker not in CATALOG_MAP:
                asset = AssetDef(
                    ticker=ticker,
                    name=entry.get("name", ticker),
                    sector=entry.get("sector", "Other"),
                    is_index=entry.get("is_index", False),
                    index_weight=entry.get("index_weight", 0.0),
                    liquidity_score=entry.get("liquidity_score", 2.0),
                )
                CATALOG.append(asset)
                CATALOG_MAP[ticker] = asset
    except (json.JSONDecodeError, KeyError):
        pass


def _save_custom_catalog():
    """Persist user-added tickers (those not in the built-in catalog) to disk."""
    from ..config import CATALOG as _cat
    # Built-in tickers are the first 101 (or however many were in config.py at import)
    # We track custom ones by checking a marker
    customs = [a for a in _cat if getattr(a, '_custom', False)]
    data = [
        {
            "ticker": a.ticker,
            "name": a.name,
            "sector": a.sector,
            "is_index": a.is_index,
            "index_weight": a.index_weight,
            "liquidity_score": a.liquidity_score,
        }
        for a in customs
    ]
    CUSTOM_CATALOG_PATH.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Selection persistence
# ---------------------------------------------------------------------------

def _selection_path(name: str = DEFAULT_SELECTION) -> Path:
    return SELECTIONS_DIR / f"{name}.json"


def save_selection(tickers: List[str], name: str = DEFAULT_SELECTION) -> Path:
    """Save the current universe selection to disk."""
    _ensure_dirs()
    path = _selection_path(name)
    path.write_text(json.dumps({"tickers": tickers}, indent=2))
    return path


def load_selection(name: str = DEFAULT_SELECTION) -> Optional[List[str]]:
    """Load a saved selection. Returns None if file doesn't exist."""
    path = _selection_path(name)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return data.get("tickers")
    except (json.JSONDecodeError, KeyError):
        return None


# ---------------------------------------------------------------------------
# Module-level init: load custom catalog, then load saved selection
# ---------------------------------------------------------------------------

_load_custom_catalog()

_saved = load_selection()
_active_tickers: List[str] = (
    [t for t in _saved if t in CATALOG_MAP] if _saved else list(DEFAULT_ACTIVE_TICKERS)
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_catalog() -> List[AssetDef]:
    """Return the full available catalog."""
    return CATALOG


def get_catalog_map() -> Dict[str, AssetDef]:
    """Return ticker -> AssetDef for the full catalog."""
    return CATALOG_MAP


def get_universe() -> List[AssetDef]:
    """Return the currently active universe (selected subset of catalog)."""
    return [CATALOG_MAP[t] for t in _active_tickers if t in CATALOG_MAP]


def get_active_tickers() -> List[str]:
    """Return list of currently active ticker strings."""
    return list(_active_tickers)


def set_active_tickers(tickers: List[str], persist: bool = False) -> List[AssetDef]:
    """
    Replace the active universe with the given tickers.
    Unknown tickers (not in catalog) are silently ignored.
    Only writes to disk when persist=True (explicit save action).
    Returns the new active universe.
    """
    global _active_tickers
    valid = [t for t in tickers if t in CATALOG_MAP]
    _active_tickers = valid
    if persist:
        save_selection(valid)
    return get_universe()


def add_ticker(ticker: str, name: str = "", sector: str = "Other",
               is_index: bool = False, liquidity_score: float = 2.0) -> AssetDef:
    """
    Add an arbitrary ticker to the catalog and active universe.
    If already in catalog, just ensure it's in the active set.
    Returns the AssetDef.
    """
    global _active_tickers
    if ticker not in CATALOG_MAP:
        asset = AssetDef(
            ticker=ticker,
            name=name or ticker,
            sector=sector,
            is_index=is_index,
            liquidity_score=liquidity_score,
        )
        asset._custom = True  # type: ignore[attr-defined]
        CATALOG.append(asset)
        CATALOG_MAP[ticker] = asset
        _save_custom_catalog()
    if ticker not in _active_tickers:
        _active_tickers.append(ticker)
        save_selection(_active_tickers)
    return CATALOG_MAP[ticker]


def confirm_ticker(ticker: str) -> None:
    """
    Called after Yahoo Finance validates a custom ticker.
    Ensures it's persisted in catalog_custom.json.
    """
    asset = CATALOG_MAP.get(ticker)
    if asset and getattr(asset, '_custom', False):
        _save_custom_catalog()


def remove_ticker(ticker: str) -> None:
    """Remove a ticker from the active universe (not from catalog)."""
    global _active_tickers
    _active_tickers = [t for t in _active_tickers if t != ticker]
    save_selection(_active_tickers)


def get_asset_map() -> Dict[str, AssetDef]:
    """Return ticker -> AssetDef for the active universe."""
    return {a.ticker: a for a in get_universe()}


def get_sectors() -> Dict[str, List[str]]:
    """Return sector -> list of tickers for the active universe."""
    sectors: Dict[str, List[str]] = {}
    for a in get_universe():
        sectors.setdefault(a.sector, []).append(a.ticker)
    return sectors
