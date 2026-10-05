"""
Finnhub WebSocket stream checks.

Run from the backend directory:

    python tests/test_finnhub_websocket.py
"""

import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services import finnhub_websocket as ws
from app.services.finnhub_websocket import (
    FinnhubWebSocketManager,
    parse_message,
    parse_trade,
)
from app.services.market_data import get_quote, get_last_quote_meta

results = []


def record(name, passed, detail=""):
    results.append((name, passed, detail))
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))


def skip(name, detail):
    results.append((name, None, detail))
    print(f"[SKIP] {name} -- {detail}")


# ---------------------------------------------------------------------------
# Test 2 / 3 -- import + configuration (never prints the secret)
# ---------------------------------------------------------------------------

record("import finnhub_websocket", True)

has_key = ws._finnhub_api_key() is not None
record("API key configured", has_key, f"present={has_key}")


# ---------------------------------------------------------------------------
# Test 6 / 13 -- parsing
# ---------------------------------------------------------------------------

trade = parse_trade(
    {"p": 191.5, "s": "AAPL", "t": 1700000000000, "v": 100, "c": ["@"]},
    received_at=1000.0,
)
record(
    "parse valid trade",
    trade is not None
    and trade.symbol == "AAPL"
    and trade.price == 191.5
    and trade.volume == 100.0
    and trade.timestamp == 1700000000000
    and trade.conditions == ["@"],
)

parsed = parse_message(
    json.dumps({
        "type": "trade",
        "data": [
            {"p": 10, "s": "MSFT", "t": 1, "v": 2},
            {"p": 11, "s": "NVDA", "t": 2, "v": 3},
        ],
    })
)
record(
    "parse trade batch",
    parsed["kind"] == "trade" and [t.symbol for t in parsed["trades"]] == ["MSFT", "NVDA"],
)

record("empty message ignored", parse_message("")["kind"] == "empty")
record("invalid JSON ignored", parse_message("not-json")["kind"] == "invalid_json")
record("unknown type ignored", parse_message('{"type":"hello"}')["kind"] == "unknown")
record("missing symbol ignored", parse_trade({"p": 1, "t": 1}) is None)
record("missing price ignored", parse_trade({"s": "AAPL", "t": 1}) is None)
record("missing timestamp ignored", parse_trade({"s": "AAPL", "p": 1}) is None)
record("ping classified", parse_message('{"type":"ping"}')["kind"] == "ping")
record("error classified", parse_message('{"type":"error","msg":"bad"}')["kind"] == "error")


# ---------------------------------------------------------------------------
# Latest state, subscribe / unsubscribe (no network)
# ---------------------------------------------------------------------------

manager = FinnhubWebSocketManager()
manager.subscribe("AAPL")
manager.subscribe("aapl")
manager.subscribe("MSFT")
manager.subscribe("NVDA")

record(
    "one manager holds many subscriptions",
    manager.subscribed_symbols() == ["AAPL", "MSFT", "NVDA"],
)

manager._handle_message(json.dumps({
    "type": "trade",
    "data": [
        {"p": 100.0, "s": "AAPL", "t": 111, "v": 10},
        {"p": 200.0, "s": "MSFT", "t": 222, "v": 20},
        {"p": 300.0, "s": "NVDA", "t": 333, "v": 30},
        {"p": 9.0, "s": "TSLA", "t": 444, "v": 1},
    ],
}))

aapl = manager.get_latest("AAPL")
msft = manager.get_latest("MSFT")
nvda = manager.get_latest("NVDA")
record(
    "latest state per subscribed symbol",
    aapl["price"] == 100.0
    and msft["price"] == 200.0
    and nvda["price"] == 300.0
    and manager.get_latest("TSLA") is None,
    f"AAPL={aapl and aapl['price']} MSFT={msft and msft['price']}",
)
record(
    "latest state has freshness fields",
    aapl is not None
    and {"symbol", "price", "volume", "timestamp", "received_at", "age_seconds", "is_stale"}
    <= set(aapl),
)

manager._handle_message(json.dumps({
    "type": "trade",
    "data": [{"p": 101.5, "s": "AAPL", "t": 112, "v": 11}],
}))
record(
    "timestamp updates on new trade",
    manager.get_latest("AAPL")["timestamp"] == 112
    and manager.get_latest("AAPL")["price"] == 101.5,
)

manager.unsubscribe("AAPL")
record(
    "unsubscribe removes symbol and latest state",
    "AAPL" not in manager.subscribed_symbols()
    and manager.get_latest("AAPL") is None
    and manager.subscribed_symbols() == ["MSFT", "NVDA"],
)

# Reconnection restores the in-memory subscription set by resending.
sent = []
manager._connected = True

class FakeWs:
    def send(self, payload):
        sent.append(json.loads(payload))

manager._ws = FakeWs()
manager._resubscribe()
record(
    "reconnect restores remaining subscriptions",
    {item["symbol"] for item in sent if item.get("type") == "subscribe"} == {"MSFT", "NVDA"},
    f"sent={sent}",
)

# Invalid messages must not crash
try:
    manager._handle_message("")
    manager._handle_message("{{{")
    manager._handle_message('{"type":"weird"}')
    manager._handle_message('{"type":"trade","data":[{"s":null}]}')
    record("invalid messages do not crash", True)
except Exception as exc:
    record("invalid messages do not crash", False, type(exc).__name__)

manager.disconnect()


# ---------------------------------------------------------------------------
# Test 14 -- missing API key
# ---------------------------------------------------------------------------

saved = os.environ.pop("FINNHUB_API_KEY", None)
try:
    bare = FinnhubWebSocketManager()
    started = bare.connect()
    record(
        "missing key: connect refused, no crash",
        started is False and not bare.is_connected(),
        f"started={started} connected={bare.is_connected()}",
    )
finally:
    if saved is not None:
        os.environ["FINNHUB_API_KEY"] = saved


# ---------------------------------------------------------------------------
# Test 4 / 5 / 8 / 10 / 12 -- live socket when a key is present
# ---------------------------------------------------------------------------

live = None
if not has_key:
    skip("WebSocket connection", "no API key")
    skip("AAPL subscription", "no API key")
    skip("live latest state", "no API key")
    skip("multiple live subscriptions", "no API key")
    skip("live unsubscribe", "no API key")
    skip("reconnection", "no API key")
    skip("shutdown", "no API key")
    skip("REST vs WebSocket comparison", "no API key")
else:
    live = FinnhubWebSocketManager()
    live.subscribe("AAPL")
    connected = False
    for _ in range(40):
        if live.is_connected():
            connected = True
            break
        time.sleep(0.25)

    record("WebSocket connection", connected, f"connected={connected}")
    record(
        "AAPL subscription accepted",
        connected and "AAPL" in live.subscribed_symbols(),
        f"subscribed={live.subscribed_symbols()}",
    )

    # Trades only arrive when the market is open. Wait briefly.
    saw_trade = False
    for _ in range(20):
        if live.get_latest("AAPL") is not None:
            saw_trade = True
            break
        time.sleep(0.5)

    if saw_trade:
        record("live latest state", True, f"price={live.get_latest('AAPL')['price']}")
    else:
        skip(
            "live latest state",
            "connected but no trade received (US market likely closed)",
        )

    live.subscribe("MSFT")
    live.subscribe("NVDA")
    time.sleep(0.5)
    record(
        "multiple subscriptions on one connection",
        live.is_connected()
        and live.subscribed_symbols() == ["AAPL", "MSFT", "NVDA"]
        and live._thread is not None
        and live._thread.is_alive(),
        f"symbols={live.subscribed_symbols()}",
    )

    live.unsubscribe("AAPL")
    record(
        "live unsubscribe",
        "AAPL" not in live.subscribed_symbols()
        and live.subscribed_symbols() == ["MSFT", "NVDA"],
    )

    # Force a disconnect of the current socket; the loop should reconnect
    # and restore MSFT/NVDA.
    existing_ws = live._ws
    if existing_ws is not None and live.is_connected():
        existing_ws.close()
        reconnected = False
        for _ in range(50):
            if live.is_connected() and live._ws is not existing_ws:
                reconnected = True
                break
            time.sleep(0.25)
        record(
            "reconnection restores subscriptions",
            reconnected and live.subscribed_symbols() == ["MSFT", "NVDA"],
            f"reconnected={reconnected} symbols={live.subscribed_symbols()}",
        )
    else:
        skip("reconnection", "not connected, cannot force close")

    # Shutdown
    thread = live._thread
    live.disconnect()
    time.sleep(0.3)
    alive = thread.is_alive() if thread is not None else False
    record(
        "shutdown stops worker",
        not live.is_connected() and not alive,
        f"connected={live.is_connected()} thread_alive={alive}",
    )


# ---------------------------------------------------------------------------
# Test 11 -- REST isolation while stream is down
# ---------------------------------------------------------------------------

try:
    quote = get_quote("AAPL", "US")
    meta = get_last_quote_meta("AAPL", "US")
    record(
        "REST isolation: get_quote still works",
        quote.get("symbol") == "AAPL" and quote.get("last_price") is not None,
        f"provider={meta and meta.get('provider')} price={quote.get('last_price')}",
    )
except Exception as exc:
    record("REST isolation: get_quote still works", False, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Test 15 -- REST vs WebSocket, only if a live trade was seen
# ---------------------------------------------------------------------------

if live is not None:
    # Reconnect briefly for the comparison if we previously shut down.
    if not live.is_connected():
        live.subscribe("AAPL")
        for _ in range(40):
            if live.is_connected():
                break
            time.sleep(0.25)
        for _ in range(16):
            if live.get_latest("AAPL") is not None:
                break
            time.sleep(0.5)

    latest = live.get_latest("AAPL")
    try:
        rest = get_quote("AAPL", "US")
    except Exception as exc:
        skip("REST vs WebSocket comparison", f"REST failed: {type(exc).__name__}")
    else:
        if latest is None:
            skip(
                "REST vs WebSocket comparison",
                "no WebSocket trade to compare (market likely closed)",
            )
            print(
                f"    REST price={rest.get('last_price')}  "
                f"WS price=None  (stream connected={live.is_connected()})"
            )
        else:
            rest_price = rest.get("last_price")
            ws_price = latest["price"]
            diff = abs(float(rest_price) - float(ws_price))
            pct = (diff / float(rest_price) * 100) if rest_price else None
            print(
                f"    AAPL REST price={rest_price}  "
                f"WS price={ws_price}  "
                f"diff={diff:.4f}  "
                f"pct={pct:.4f}%  "
                f"WS timestamp={latest['timestamp']}"
            )
            record(
                "REST vs WebSocket comparison",
                rest.get("symbol") == "AAPL" and latest["symbol"] == "AAPL",
                f"diff={diff:.4f}",
            )

    live.disconnect()


# ---------------------------------------------------------------------------
# FastAPI stream routes + app startup/shutdown
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.routes.market import router as market_router

    isolated = FastAPI()
    isolated.include_router(market_router)
    client = TestClient(isolated)
    status = client.get("/api/market/stream/status")
    body = status.json()
    dumped = json.dumps(body)
    key = os.getenv("FINNHUB_API_KEY") or ""
    record(
        "API stream status (no secrets)",
        status.status_code == 200
        and "connected" in body
        and body.get("endpoint") == "wss://ws.finnhub.io?token=REDACTED"
        and (not key or key not in dumped),
        f"keys={sorted(body)} endpoint={body.get('endpoint')}",
    )
except Exception as exc:
    record("API stream status (no secrets)", False, f"{type(exc).__name__}: {exc}")


print()
passed = sum(1 for _, p, _ in results if p is True)
failed = sum(1 for _, p, _ in results if p is False)
skipped = sum(1 for _, p, _ in results if p is None)
print(f"passed={passed} failed={failed} skipped={skipped}")
sys.exit(1 if failed else 0)
