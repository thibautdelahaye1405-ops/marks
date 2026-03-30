"""
SQLite persistence layer for snapshots and configuration.
"""
import sqlite3
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

DB_PATH = Path(__file__).parent.parent.parent / "db" / "marks.sqlite"


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            ticker TEXT NOT NULL,
            expiry TEXT NOT NULL,
            T REAL NOT NULL,
            spot REAL NOT NULL,
            forward REAL NOT NULL,
            atm_iv REAL NOT NULL,
            strikes TEXT NOT NULL,
            mid_ivs TEXT NOT NULL,
            bid_ask_spread TEXT NOT NULL,
            open_interest TEXT NOT NULL,
            call_mids TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS w_matrices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            tickers TEXT NOT NULL,
            matrix TEXT NOT NULL,
            alphas TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_snap_ticker ON snapshots(ticker, timestamp);
        CREATE INDEX IF NOT EXISTS idx_snap_ticker_expiry ON snapshots(ticker, expiry, timestamp);
    """)
    conn.commit()
    conn.close()


def save_snapshot(chain_data) -> int:
    """Save an OptionChainData to the database."""
    conn = _get_conn()
    cur = conn.execute(
        """INSERT INTO snapshots
           (timestamp, ticker, expiry, T, spot, forward, atm_iv,
            strikes, mid_ivs, bid_ask_spread, open_interest, call_mids)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now().isoformat(),
            chain_data.ticker,
            chain_data.expiry,
            chain_data.T,
            chain_data.spot,
            chain_data.forward,
            chain_data.atm_iv,
            json.dumps(chain_data.strikes.tolist()),
            json.dumps(chain_data.mid_ivs.tolist()),
            json.dumps(chain_data.bid_ask_spread.tolist()),
            json.dumps(chain_data.open_interest.tolist()),
            json.dumps(chain_data.call_mids.tolist()),
        ),
    )
    conn.commit()
    snap_id = cur.lastrowid
    conn.close()
    return snap_id


def get_latest_snapshots(keys: List[str]) -> Dict[str, dict]:
    """Get the most recent snapshot for each key.

    Keys can be:
    - Plain tickers ("SPY") — returns latest snapshot regardless of expiry
    - Node keys ("SPY:2025-04-17") — returns latest snapshot for that specific expiry
    """
    from ..utils.node_key import split_node_key, is_compound_key, make_node_key

    conn = _get_conn()
    results = {}
    for key in keys:
        ticker, expiry = split_node_key(key)
        if expiry:
            row = conn.execute(
                "SELECT * FROM snapshots WHERE ticker=? AND expiry=? ORDER BY timestamp DESC LIMIT 1",
                (ticker, expiry),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM snapshots WHERE ticker=? ORDER BY timestamp DESC LIMIT 1",
                (ticker,),
            ).fetchone()
        if row:
            result_key = make_node_key(row["ticker"], row["expiry"]) if is_compound_key(key) else row["ticker"]
            results[result_key] = {
                "ticker": row["ticker"],
                "expiry": row["expiry"],
                "T": row["T"],
                "spot": row["spot"],
                "forward": row["forward"],
                "atm_iv": row["atm_iv"],
                "strikes": np.array(json.loads(row["strikes"])),
                "mid_ivs": np.array(json.loads(row["mid_ivs"])),
                "bid_ask_spread": np.array(json.loads(row["bid_ask_spread"])),
                "open_interest": np.array(json.loads(row["open_interest"])),
                "call_mids": np.array(json.loads(row["call_mids"])),
            }
    conn.close()
    return results


def save_w_matrix(tickers: List[str], W: np.ndarray, alphas: np.ndarray) -> int:
    """Persist a W matrix configuration."""
    conn = _get_conn()
    cur = conn.execute(
        "INSERT INTO w_matrices (timestamp, tickers, matrix, alphas) VALUES (?, ?, ?, ?)",
        (
            datetime.now().isoformat(),
            json.dumps(tickers),
            json.dumps(W.tolist()),
            json.dumps(alphas.tolist()),
        ),
    )
    conn.commit()
    w_id = cur.lastrowid
    conn.close()
    return w_id


# Initialise on import
init_db()
