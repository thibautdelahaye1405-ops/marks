# US Treasury yield curve fetching and interpolation via FRED public CSV endpoint.

import logging
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

logger = logging.getLogger(__name__)

FRED_URL = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv"
    "?id=DGS1MO,DGS3MO,DGS6MO,DGS1,DGS2,DGS3,DGS5,DGS7,DGS10,DGS20,DGS30"
)

ALL_TENORS = np.array([1 / 12, 3 / 12, 6 / 12, 1, 2, 3, 5, 7, 10, 20, 30])

_cache: "TreasuryCurve | None" = None


@dataclass
class TreasuryCurve:
    date: str
    tenors: np.ndarray
    rates: np.ndarray
    fetched_at: datetime = field(default_factory=datetime.now)

    def rate_at(self, T: float) -> float:
        if len(self.tenors) == 0:
            return 0.045
        if T <= self.tenors[0]:
            return float(self.rates[0])
        if T >= self.tenors[-1]:
            return float(self.rates[-1])
        log_tenors = np.log(self.tenors)
        log_T = np.log(T)
        return float(np.interp(log_T, log_tenors, self.rates))


def _parse_csv(text: str) -> TreasuryCurve:
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    # Last non-empty data row
    row = lines[-1].split(",")
    date_str = row[0]
    values = row[1:]

    tenors = []
    rates = []
    for tenor, val in zip(ALL_TENORS, values):
        if val == "." or val == "":
            continue
        tenors.append(tenor)
        rates.append(float(val) / 100.0)

    return TreasuryCurve(
        date=date_str,
        tenors=np.array(tenors),
        rates=np.array(rates),
    )


def _fallback_curve() -> TreasuryCurve:
    return TreasuryCurve(
        date="fallback",
        tenors=ALL_TENORS.copy(),
        rates=np.full(len(ALL_TENORS), 0.045),
    )


def fetch_treasury_curve() -> TreasuryCurve:
    global _cache
    if _cache is not None and (datetime.now() - _cache.fetched_at) < timedelta(hours=1):
        return _cache

    try:
        with urllib.request.urlopen(FRED_URL, timeout=10) as resp:
            text = resp.read().decode("utf-8")
        _cache = _parse_csv(text)
    except Exception as e:
        logger.warning("Failed to fetch Treasury curve from FRED: %s. Using fallback 4.5%% flat.", e)
        _cache = _fallback_curve()

    return _cache
