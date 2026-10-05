"""
Market snapshot integration for /api/analyze.

Run from the backend directory:

    python tests/test_analyze_market_snapshot.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if not (os.getenv("SEC_USER_AGENT") or "").strip():
    os.environ["SEC_USER_AGENT"] = "TAO-Fin snapshot-test contact@example.com"

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routes.analyze import router as analyze_router
from app.db.database import SessionLocal
from app.db.models import Company
from app.services.finnhub_websocket import FinnhubWebSocketManager, get_stream_manager
from app.services.market_data import get_last_quote_meta, get_quote
from app.services.market_snapshot import build_market_snapshot
from app.services.tao_pipeline import TAOPipeline, _wants_current_price

results = []


def record(name, passed, detail=""):
    results.append((name, passed, detail))
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))


def skip(name, detail):
    results.append((name, None, detail))
    print(f"[SKIP] {name} -- {detail}")


class FakeManager:
    def __init__(self, latest=None, fail_subscribe=False):
        self._latest = latest
        self._subs = set()
        self.fail_subscribe = fail_subscribe
        self.subscribe_calls = 0

    def subscribed_symbols(self):
        return sorted(self._subs)

    def subscribe(self, symbol):
        self.subscribe_calls += 1
        if self.fail_subscribe:
            raise RuntimeError("websocket down")
        self._subs.add(symbol.upper())
        return {"symbol": symbol.upper(), "subscribed": True, "connected": False}

    def get_latest(self, symbol):
        if symbol.upper() in (self._latest or {}):
            return self._latest[symbol.upper()]
        return None


# ---------------------------------------------------------------------------
# Adapter unit tests (no TAO / no RAG)
# ---------------------------------------------------------------------------

fresh = {
    "AAPL": {
        "symbol": "AAPL",
        "price": 330.69,
        "volume": 100,
        "timestamp": 1760000000000,
        "received_at": "2026-10-02T12:00:00+00:00",
        "age_seconds": 1.2,
        "is_stale": False,
    }
}
snap = build_market_snapshot("AAPL", manager=FakeManager(latest=fresh))
record(
    "fresh WebSocket snapshot used",
    snap is not None
    and snap["source"] == "finnhub_websocket"
    and snap["price"] == 330.69,
    f"source={snap and snap.get('source')} price={snap and snap.get('price')}",
)

stale = {
    "AAPL": {**fresh["AAPL"], "is_stale": True, "price": 1.0}
}
snap = build_market_snapshot("AAPL", manager=FakeManager(latest=stale))
record(
    "stale WebSocket falls back to REST/yfinance",
    snap is not None and snap["source"] in {"finnhub_rest", "yfinance"} and snap["price"] != 1.0,
    f"source={snap and snap.get('source')} price={snap and snap.get('price')}",
)

snap = build_market_snapshot("AAPL", manager=FakeManager(latest={}))
record(
    "no WebSocket state falls back to REST/yfinance",
    snap is not None and snap["source"] in {"finnhub_rest", "yfinance"},
    f"source={snap and snap.get('source')}",
)

fake_down = FakeManager(latest={}, fail_subscribe=True)
snap = build_market_snapshot("AAPL", manager=fake_down)
record(
    "WebSocket disconnected falls back to REST/yfinance",
    snap is not None and snap["source"] in {"finnhub_rest", "yfinance"},
    f"source={snap and snap.get('source')}",
)

record(
    "current-price detector",
    _wants_current_price("What is AAPL's current stock price?")
    and not _wants_current_price("What was AAPL FY2018 revenue?"),
)

# Duplicate subscribe: second call should still be idempotent on real manager API
fake = FakeManager(latest=fresh)
build_market_snapshot("AAPL", manager=fake)
build_market_snapshot("AAPL", manager=fake)
record(
    "no duplicate subscribe when already in set",
    fake.subscribe_calls == 1 and fake.subscribed_symbols() == ["AAPL"],
    f"calls={fake.subscribe_calls}",
)


# ---------------------------------------------------------------------------
# REST isolation
# ---------------------------------------------------------------------------

try:
    quote = get_quote("AAPL", "US")
    meta = get_last_quote_meta("AAPL", "US")
    record(
        "REST isolation get_quote",
        quote.get("symbol") == "AAPL" and "last_price" in quote,
        f"provider={meta and meta.get('provider')}",
    )
except Exception as exc:
    record("REST isolation get_quote", False, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Analyze route + snapshot (AAPL already indexed from prior work)
# ---------------------------------------------------------------------------

# Isolated app: reuse already-synced tickers and skip RAG rebuild.
# Spot-price questions do not call the LLM.
app = FastAPI()
app.include_router(analyze_router)
app.state.tao_pipeline = TAOPipeline(
    rag_pipeline=object(),
    company_index={
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corporation",
        "NVDA": "NVIDIA",
    },
)
app.state.indexed_tickers = {"AAPL", "MSFT", "NVDA"}
client = TestClient(app)


def analyze(question):
    return client.post("/api/analyze", json={"question": question})


invalid = analyze("What is INVALIDXYZ current stock price?")
record(
    "invalid ticker still 404",
    invalid.status_code == 404,
    f"status={invalid.status_code} body={invalid.json()}",
)
db = SessionLocal()
try:
    record(
        "invalid ticker created no company",
        db.query(Company).filter(Company.ticker == "INVALIDXYZ").one_or_none() is None,
    )
finally:
    db.close()


# Live WS: subscribe, wait briefly for a trade, then analyze current price.
stream = get_stream_manager()
stream.subscribe("AAPL")
stream.subscribe("MSFT")
stream.subscribe("NVDA")
one_connection = stream._thread is not None
for _ in range(24):
    if stream.get_latest("AAPL") is not None:
        break
    time.sleep(0.5)

live = stream.get_latest("AAPL")
existing = analyze("What is AAPL's current stock price?")
body = existing.json() if existing.headers.get("content-type", "").startswith("application/json") else {}
if existing.status_code == 200 and body.get("market_snapshot"):
    src = body["market_snapshot"].get("source")
    if live and not live.get("is_stale"):
        record(
            "fresh WebSocket analysis",
            src == "finnhub_websocket" and "answer" in body and "verification" in body,
            f"source={src} price={body['market_snapshot'].get('price')}",
        )
    else:
        record(
            "no WebSocket state uses REST/yfinance",
            src in {"finnhub_rest", "yfinance"} and "answer" in body,
            f"source={src}",
        )
        skip("fresh WebSocket analysis", "no live AAPL trade (market likely closed)")
elif existing.status_code == 200:
    record("fresh WebSocket analysis", False, f"missing market_snapshot keys={sorted(body)}")
else:
    record(
        "fresh WebSocket analysis",
        False,
        f"status={existing.status_code} body={str(body)[:240]}",
    )

record(
    "multiple symbols one connection",
    {"AAPL", "MSFT", "NVDA"} <= set(stream.subscribed_symbols()),
    f"subs={stream.subscribed_symbols()}",
)

# Disconnect: clear latest by unsubscribing, disconnect, then snapshot/analyze.
# subscribe() will try to reconnect but get_latest is empty → REST.
for symbol in list(stream.subscribed_symbols()):
    stream.unsubscribe(symbol)
stream.disconnect()
time.sleep(0.3)

disconnected = analyze("What is AAPL's current stock price?")
dbody = disconnected.json() if disconnected.status_code == 200 else {}
record(
    "WebSocket disconnected still analyzes",
    disconnected.status_code == 200
    and dbody.get("market_snapshot", {}).get("source") in {"finnhub_rest", "yfinance", "finnhub_websocket"}
    and "answer" in dbody,
    f"status={disconnected.status_code} source={dbody.get('market_snapshot', {}).get('source')}",
)

# Company auto-sync still works for a listed ticker already in DB (MSFT).
msft = analyze("What is MSFT's current stock price?")
record(
    "company auto-sync / existing MSFT analyze",
    msft.status_code == 200 and "answer" in msft.json() and msft.json().get("market_snapshot"),
    f"status={msft.status_code} source={(msft.json() or {}).get('market_snapshot', {}).get('source')}",
)

# WebSocket isolation after analyze
iso = FinnhubWebSocketManager()
try:
    iso.subscribe("AAPL")
    connected = False
    for _ in range(40):
        if iso.is_connected():
            connected = True
            break
        time.sleep(0.25)
    iso.unsubscribe("AAPL")
    record(
        "WebSocket isolation",
        connected and "AAPL" not in iso.subscribed_symbols(),
        f"connected={connected}",
    )
finally:
    iso.disconnect()

e2e = (
    existing.status_code == 200
    and "answer" in body
    and "verification" in body
    and "tao" in body
    and body.get("market_snapshot", {}).get("source")
)
record("full end-to-end analyze + snapshot", e2e, f"source={body.get('market_snapshot', {}).get('source')}")


print()
passed = sum(1 for _, p, _ in results if p is True)
failed = sum(1 for _, p, _ in results if p is False)
skipped = sum(1 for _, p, _ in results if p is None)
print(f"passed={passed} failed={failed} skipped={skipped}")
sys.exit(1 if failed else 0)
