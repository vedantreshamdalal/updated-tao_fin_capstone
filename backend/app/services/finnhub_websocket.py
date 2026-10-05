"""
Finnhub WebSocket stream — one connection, many US-equity subscriptions.

This module is an addition to the existing REST quote path in
market_data.py. It never replaces get_quote(), never writes ticks to
the database, and never invokes TAO / the LLM.

                  TAO-Fin Backend
                        │
                ┌───────┴────────┐
                │                │
          Finnhub REST      Finnhub WebSocket
          Current Quote     Live Trades
                │                │
                │          Stream Manager
                │          Latest State
                └───────┬────────┘
                        │
                 Market Snapshot (on request only)
                        │
                 TAO / Verifier / API

yfinance remains the REST fallback and the source for history,
ratios, and NSE/BSE.

The WebSocket URL contains the API token. Logs never include it;
the endpoint is written as wss://ws.finnhub.io?token=REDACTED.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

_WS_HOST = "wss://ws.finnhub.io"
_REDACTED_ENDPOINT = f"{_WS_HOST}?token=REDACTED"

_DEFAULT_STALE_SECONDS = 30.0
_MIN_BACKOFF_SECONDS = 1.0
_MAX_BACKOFF_SECONDS = 30.0
_STABLE_RESET_SECONDS = 5.0


def _finnhub_api_key() -> Optional[str]:
    """Same source as market_data.py: FINNHUB_API_KEY, read at call time."""
    key = os.getenv("FINNHUB_API_KEY")
    return key.strip() if key and key.strip() else None


def _stale_threshold_seconds() -> float:
    raw = os.getenv("FINNHUB_WS_STALE_SECONDS")
    if not raw:
        return _DEFAULT_STALE_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_STALE_SECONDS
    return value if value > 0 else _DEFAULT_STALE_SECONDS


def _normalize_symbol(symbol: str) -> str:
    return (symbol or "").upper().strip()


def _log(message: str) -> None:
    print(f"[market_stream] {message}")


@dataclass
class MarketTrade:
    symbol: str
    price: float
    volume: Optional[float]
    timestamp: int
    conditions: Optional[List[str]] = None
    received_at: float = field(default_factory=time.time)

    def to_state(self, stale_after: float) -> Dict[str, Any]:
        now = time.time()
        age = max(0.0, now - self.received_at)
        return {
            "symbol": self.symbol,
            "price": self.price,
            "volume": self.volume,
            "timestamp": self.timestamp,
            "conditions": self.conditions,
            "received_at": datetime.fromtimestamp(
                self.received_at, tz=timezone.utc
            ).isoformat(),
            "age_seconds": round(age, 3),
            "is_stale": age > stale_after,
        }


def parse_trade(item: Any, received_at: Optional[float] = None) -> Optional[MarketTrade]:
    """
    Normalize one Finnhub trade object. Returns None when required
    fields are missing or the wrong type — callers ignore those.
    """

    if not isinstance(item, dict):
        return None

    symbol = item.get("s")
    if not isinstance(symbol, str) or not symbol.strip():
        return None

    price = item.get("p")
    if not isinstance(price, (int, float)):
        return None

    timestamp = item.get("t")
    if not isinstance(timestamp, (int, float)):
        return None

    volume = item.get("v")
    if volume is not None and not isinstance(volume, (int, float)):
        volume = None

    conditions = item.get("c")
    if conditions is not None and not isinstance(conditions, list):
        conditions = None

    return MarketTrade(
        symbol=_normalize_symbol(symbol),
        price=float(price),
        volume=float(volume) if volume is not None else None,
        timestamp=int(timestamp),
        conditions=list(conditions) if conditions is not None else None,
        received_at=received_at if received_at is not None else time.time(),
    )


def parse_message(raw: str, received_at: Optional[float] = None) -> Dict[str, Any]:
    """
    Defensively classify a WebSocket text frame.

    Returns a dict with at least {"kind": ...}. Never raises on bad input.
    """

    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return {"kind": "empty"}

    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError:
            return {"kind": "invalid_json"}

    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return {"kind": "invalid_json"}

    if not isinstance(payload, dict):
        return {"kind": "unknown"}

    msg_type = payload.get("type")

    if msg_type == "ping":
        return {"kind": "ping"}

    if msg_type == "error":
        return {
            "kind": "error",
            "message": str(payload.get("msg") or payload.get("message") or "error"),
        }

    if msg_type == "trade":
        data = payload.get("data")
        if not isinstance(data, list):
            return {"kind": "trade", "trades": []}

        trades = []
        for item in data:
            trade = parse_trade(item, received_at=received_at)
            if trade is not None:
                trades.append(trade)
        return {"kind": "trade", "trades": trades}

    return {"kind": "unknown", "type": msg_type}


class FinnhubWebSocketManager:
    """
    One Finnhub WebSocket connection shared by every subscribed symbol.

    Public API is synchronous so it matches the existing FastAPI routes.
    The socket itself runs on a background thread and never blocks the
    event loop. A streaming failure leaves get_quote() / yfinance alone.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ws = None
        self._connected = False
        self._subscriptions: Set[str] = set()
        self._latest: Dict[str, MarketTrade] = {}
        self._backoff = _MIN_BACKOFF_SECONDS
        self._connected_at: Optional[float] = None

    def connect(self) -> bool:
        """Start the background connection if it is not already running."""

        if _finnhub_api_key() is None:
            _log("FINNHUB_API_KEY is not configured; WebSocket will not connect")
            return False

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return True
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run_loop,
                name="finnhub-websocket",
                daemon=True,
            )
            self._thread.start()
        return True

    def disconnect(self) -> None:
        """Close the socket, stop reconnects, and join the worker thread."""

        self._stop.set()
        ws = None
        thread = None
        with self._lock:
            ws = self._ws
            thread = self._thread
            self._ws = None
            self._connected = False
            self._connected_at = None

        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=8)

        with self._lock:
            if self._thread is thread:
                self._thread = None

        _log("Finnhub WebSocket disconnected")

    def subscribe(self, symbol: str) -> Dict[str, Any]:
        symbol = _normalize_symbol(symbol)
        if not symbol:
            return {"symbol": symbol, "subscribed": False, "error": "symbol is required"}

        with self._lock:
            already = symbol in self._subscriptions
            self._subscriptions.add(symbol)

        if not already:
            _log(f"Subscribed to {symbol}")
            self._send({"type": "subscribe", "symbol": symbol})

        if not self.is_connected():
            self.connect()

        return {
            "symbol": symbol,
            "subscribed": True,
            "connected": self.is_connected(),
        }

    def unsubscribe(self, symbol: str) -> Dict[str, Any]:
        symbol = _normalize_symbol(symbol)

        with self._lock:
            present = symbol in self._subscriptions
            self._subscriptions.discard(symbol)
            self._latest.pop(symbol, None)

        if present:
            _log(f"Unsubscribed from {symbol}")
            self._send({"type": "unsubscribe", "symbol": symbol})

        return {
            "symbol": symbol,
            "subscribed": False,
            "connected": self.is_connected(),
        }

    def is_connected(self) -> bool:
        with self._lock:
            return self._connected

    def subscribed_symbols(self) -> List[str]:
        with self._lock:
            return sorted(self._subscriptions)

    def get_latest(self, symbol: str) -> Optional[Dict[str, Any]]:
        symbol = _normalize_symbol(symbol)
        with self._lock:
            trade = self._latest.get(symbol)
        if trade is None:
            return None
        return trade.to_state(_stale_threshold_seconds())

    def get_all_latest(self) -> Dict[str, Dict[str, Any]]:
        stale_after = _stale_threshold_seconds()
        with self._lock:
            items = list(self._latest.items())
        return {symbol: trade.to_state(stale_after) for symbol, trade in items}

    def status(self) -> Dict[str, Any]:
        with self._lock:
            connected = self._connected
            symbols = sorted(self._subscriptions)
            has_trades = {
                symbol: symbol in self._latest for symbol in symbols
            }

        return {
            "connected": connected,
            "endpoint": _REDACTED_ENDPOINT,
            "subscribed": symbols,
            "has_latest_trade": has_trades,
            "stale_after_seconds": _stale_threshold_seconds(),
            "note": (
                "connected=true with no latest trade usually means the "
                "US market is closed, not that the socket is broken."
            ),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _send(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            ws = self._ws
            connected = self._connected
        if ws is None or not connected:
            return
        try:
            ws.send(json.dumps(payload))
        except Exception as exc:
            _log(f"send failed: {type(exc).__name__}")

    def _resubscribe(self) -> None:
        with self._lock:
            symbols = list(self._subscriptions)
        for symbol in symbols:
            self._send({"type": "subscribe", "symbol": symbol})
        if symbols:
            _log(f"Restored subscriptions: {', '.join(symbols)}")

    def _next_backoff(self) -> float:
        with self._lock:
            delay = self._backoff
            self._backoff = min(self._backoff * 2.0, _MAX_BACKOFF_SECONDS)
        return delay

    def _reset_backoff_if_stable(self) -> None:
        with self._lock:
            connected_at = self._connected_at
        if connected_at is not None and (time.time() - connected_at) >= _STABLE_RESET_SECONDS:
            with self._lock:
                self._backoff = _MIN_BACKOFF_SECONDS

    def _on_open(self, _ws) -> None:
        with self._lock:
            self._connected = True
            self._connected_at = time.time()
        _log("Finnhub WebSocket connected")
        self._resubscribe()

    def _on_close(self, _ws, status_code, _msg) -> None:
        with self._lock:
            self._connected = False
            self._connected_at = None
            if self._ws is _ws:
                self._ws = None
        _log(f"Finnhub WebSocket closed (code={status_code})")

    def _on_error(self, _ws, error) -> None:
        # Exception strings from the client can embed the request URL,
        # which carries the token. Log the class name only.
        _log(f"Finnhub WebSocket error: {type(error).__name__}")

    def _on_message(self, _ws, message) -> None:
        self._handle_message(message)

    def _handle_message(self, message: Any) -> None:
        parsed = parse_message(message)

        if parsed["kind"] == "ping":
            self._send({"type": "pong"})
            return

        if parsed["kind"] == "error":
            _log(f"Finnhub server error: {parsed.get('message')}")
            return

        if parsed["kind"] in {"empty", "invalid_json", "unknown"}:
            if parsed["kind"] == "invalid_json":
                _log("Finnhub WebSocket message parsing error")
            return

        trades: List[MarketTrade] = parsed.get("trades") or []
        if not trades:
            return

        with self._lock:
            wanted = self._subscriptions
            for trade in trades:
                if trade.symbol in wanted:
                    self._latest[trade.symbol] = trade

        self._reset_backoff_if_stable()

    def _run_loop(self) -> None:
        try:
            import websocket
        except ImportError:
            _log("websocket-client is not installed; streaming disabled")
            return

        while not self._stop.is_set():
            api_key = _finnhub_api_key()
            if api_key is None:
                _log("FINNHUB_API_KEY is not configured; WebSocket will not connect")
                break

            _log("Finnhub WebSocket connecting")
            url = f"{_WS_HOST}?token={api_key}"

            ws = websocket.WebSocketApp(
                url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )

            with self._lock:
                self._ws = ws

            try:
                ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as exc:
                _log(f"Finnhub WebSocket run failed: {type(exc).__name__}")

            with self._lock:
                self._connected = False
                if self._ws is ws:
                    self._ws = None

            if self._stop.is_set():
                break

            delay = self._next_backoff()
            _log(f"Finnhub WebSocket reconnecting in {delay:.0f}s")
            if self._stop.wait(delay):
                break

        with self._lock:
            self._connected = False
            self._thread = None


_manager: Optional[FinnhubWebSocketManager] = None
_manager_lock = threading.Lock()


def get_stream_manager() -> FinnhubWebSocketManager:
    """Process-wide singleton. Created lazily; does not connect on import."""

    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = FinnhubWebSocketManager()
        return _manager


def reset_stream_manager() -> None:
    """Test helper: drop the singleton after disconnect."""

    global _manager
    with _manager_lock:
        existing = _manager
        _manager = None
    if existing is not None:
        existing.disconnect()
