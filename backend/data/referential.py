"""
Asset universe referential data.
"""
from typing import List, Dict
from ..config import AssetDef, DEFAULT_UNIVERSE


def get_universe() -> List[AssetDef]:
    return DEFAULT_UNIVERSE


def get_asset_map() -> Dict[str, AssetDef]:
    return {a.ticker: a for a in DEFAULT_UNIVERSE}


def get_sectors() -> Dict[str, List[str]]:
    sectors: Dict[str, List[str]] = {}
    for a in DEFAULT_UNIVERSE:
        sectors.setdefault(a.sector, []).append(a.ticker)
    return sectors
