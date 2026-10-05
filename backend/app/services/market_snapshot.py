"""
Market snapshot for /api/analyze.

Priority:

    1. Fresh Finnhub WebSocket last-trade (already in memory)
    2. Existing get_quote()  → Finnhub REST, then yfinance fallback

Does not wait for a tick, does not open a second WebSocket, and does
not change get_quote(). A streaming miss must never fail analysis.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from app.services.finnhub_websocket import FinnhubWebSocketManager, get_stream_manager
from app.services.market_data import get_last_quote_meta, get_quote


def _from_websocket(latest: Dict[str, Any], ticker: str) -> Optional[Dict[str, Any]]:
    if not latest or latest.get("price") is None:
        return None
    if latest.get("is_stale"):
        return None

    return {
        "symbol": ticker,
        "price": latest["price"],
        "volume": latest.get("volume"),
        "timestamp": latest.get("timestamp"),
        "received_at": latest.get("received_at"),
        "age_seconds": latest.get("age_seconds"),
        "is_stale": False,
        "source": "finnhub_websocket",
    }


def _from_quote(ticker: str) -> Optional[Dict[str, Any]]:
    quote = get_quote(ticker, "US")
    meta = get_last_quote_meta(ticker, "US") or {}
    provider = meta.get("provider") or "finnhub"

    if provider in {"yfinance", "yfinance_fallback"}:
        source = "yfinance"
    else:
        source = "finnhub_rest"

    return {
        "symbol": ticker,
        "price": quote.get("last_price"),
        "volume": quote.get("volume"),
        "timestamp": meta.get("quote_timestamp"),
        "received_at": None,
        "age_seconds": None,
        "is_stale": False,
        "source": source,
    }


def build_market_snapshot(
    ticker: str,
    manager: Optional[FinnhubWebSocketManager] = None,
) -> Optional[Dict[str, Any]]:
    """
    Subscribe (idempotent) and return the best available snapshot.

    Never blocks waiting for a trade. Returns None only if REST/yfinance
    also fail — callers must still continue analysis.
    """

    ticker = (ticker or "").upper().strip()
    if not ticker:
        return None

    stream = manager if manager is not None else get_stream_manager()

    try:
        if ticker not in stream.subscribed_symbols():
            stream.subscribe(ticker)
        snapshot = _from_websocket(stream.get_latest(ticker) or {}, ticker)
        if snapshot is not None:
            return snapshot
    except Exception as exc:
        print(
            f"[market_snapshot] WebSocket unavailable for {ticker}: "
            f"{type(exc).__name__}"
        )

    try:
        return _from_quote(ticker)
    except Exception as exc:
        print(
            f"[market_snapshot] REST quote unavailable for {ticker}: "
            f"{type(exc).__name__}"
        )
        return None
