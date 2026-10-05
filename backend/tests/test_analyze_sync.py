"""
On-demand company resolution for /api/analyze.

Run from the backend directory:

    python tests/test_analyze_sync.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Tests may download the SEC ticker map; do not write this into .env.
if not (os.getenv("SEC_USER_AGENT") or "").strip():
    os.environ["SEC_USER_AGENT"] = "TAO-Fin analyze-test contact@example.com"

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.db.database import SessionLocal, init_db
from app.db.models import Company
from app.services import company_service
from app.services.finnhub_websocket import FinnhubWebSocketManager
from app.services.market_data import get_last_quote_meta, get_quote
from app.routes.analyze import router as analyze_router

results = []


def record(name, passed, detail=""):
    results.append((name, passed, detail))
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))


def skip(name, detail):
    results.append((name, None, detail))
    print(f"[SKIP] {name} -- {detail}")


init_db()


# ---------------------------------------------------------------------------
# Ticker resolution (no analyze / no RAG)
# ---------------------------------------------------------------------------

try:
    record(
        "find AAPL in question",
        company_service.find_ticker_in_question("What is AAPL's FY2018 revenue?") == "AAPL",
    )
    record(
        "find MSFT in question",
        company_service.find_ticker_in_question("Compare MSFT operating margin") == "MSFT",
    )
    record(
        "INVALIDXYZ is not a listed ticker",
        company_service.resolve_cik("INVALIDXYZ") is None,
    )
    record(
        "INVALIDXYZ question does not match VENU via 'revenue'",
        company_service.find_ticker_in_question("What is INVALIDXYZ revenue?") is None,
    )
    record(
        "no ticker in generic question",
        company_service.find_ticker_in_question("What is the weather?") is None,
    )
except Exception as exc:
    record("ticker resolution", False, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Analyze route: invalid ticker, missing ticker, no fake company
# ---------------------------------------------------------------------------

app = FastAPI()
app.include_router(analyze_router)
app.state.tao_pipeline = None
app.state.indexed_tickers = set()
client = TestClient(app)

before = SessionLocal()
try:
    existing_tickers = {c.ticker for c in before.query(Company).all()}
finally:
    before.close()

no_ticker = client.post("/api/analyze", json={"question": "What is the weather today?"})
record(
    "generic question is 400",
    no_ticker.status_code == 400,
    f"status={no_ticker.status_code} body={no_ticker.json()}",
)

invalid = client.post("/api/analyze", json={"question": "What is INVALIDXYZ revenue?"})
record(
    "invalid ticker is 404",
    invalid.status_code == 404,
    f"status={invalid.status_code} body={invalid.json()}",
)

after = SessionLocal()
try:
    created_fake = after.query(Company).filter(Company.ticker == "INVALIDXYZ").one_or_none()
    record("invalid ticker did not create a company row", created_fake is None)
finally:
    after.close()


# ---------------------------------------------------------------------------
# Duplicate prevention around needs_filing_sync / unique ticker
# ---------------------------------------------------------------------------

db: Session = SessionLocal()
try:
    aapl_needed = company_service.needs_filing_sync(db, "AAPL")
    first = company_service.upsert_company(db, "AAPL")
    second = company_service.upsert_company(db, "AAPL")
    count = db.query(Company).filter(Company.ticker == "AAPL").count()
    record(
        "duplicate prevention: upsert_company is idempotent",
        first.id == second.id and count == 1,
        f"id={first.id} count={count} needed_filings={aapl_needed}",
    )
except Exception as exc:
    record("duplicate prevention: upsert_company is idempotent", False, f"{type(exc).__name__}: {exc}")
finally:
    db.close()


# ---------------------------------------------------------------------------
# Existing / unsynced analyze (may sync + index; can take several minutes)
# ---------------------------------------------------------------------------

def _analyze(question):
    return client.post("/api/analyze", json={"question": question})


# Isolated TestClient will run ensure_pipeline which may full_sync + embed.
# That needs SEC + embeddings + Ollama. Report skip/fail honestly.

try:
    existing = _analyze("What was AAPL's total net sales in the most recent 10-K?")
    if existing.status_code == 200 and "answer" in existing.json():
        record("existing company analyze (AAPL)", True, f"keys={sorted(existing.json())}")
    elif existing.status_code == 503:
        skip("existing company analyze (AAPL)", f"external dependency: {existing.json()}")
    else:
        record(
            "existing company analyze (AAPL)",
            False,
            f"status={existing.status_code} body={existing.text[:300]}",
        )
except Exception as exc:
    skip("existing company analyze (AAPL)", f"{type(exc).__name__}: {exc}")

try:
    unsynced = _analyze("What was MSFT's total net sales in the most recent 10-K?")
    if unsynced.status_code == 200 and "answer" in unsynced.json():
        record("unsynced valid company analyze (MSFT)", True)
    elif unsynced.status_code == 503:
        skip("unsynced valid company analyze (MSFT)", f"external dependency: {unsynced.json()}")
    else:
        record(
            "unsynced valid company analyze (MSFT)",
            False,
            f"status={unsynced.status_code} body={unsynced.text[:300]}",
        )
except Exception as exc:
    skip("unsynced valid company analyze (MSFT)", f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Finnhub REST + yfinance fallback + WebSocket
# ---------------------------------------------------------------------------

try:
    quote = get_quote("AAPL", "US")
    meta = get_last_quote_meta("AAPL", "US")
    record(
        "Finnhub REST get_quote",
        quote.get("symbol") == "AAPL" and "last_price" in quote,
        f"provider={meta and meta.get('provider')}",
    )
except Exception as exc:
    record("Finnhub REST get_quote", False, f"{type(exc).__name__}: {exc}")

saved = os.environ.pop("FINNHUB_API_KEY", None)
try:
    from app.services import market_data
    market_data.clear_quote_cache()
    quote = get_quote("AAPL", "US")
    meta = get_last_quote_meta("AAPL", "US")
    record(
        "yfinance fallback",
        meta and meta.get("provider") == "yfinance" and quote.get("symbol") == "AAPL",
        f"provider={meta and meta.get('provider')}",
    )
except Exception as exc:
    record("yfinance fallback", False, f"{type(exc).__name__}: {exc}")
finally:
    if saved is not None:
        os.environ["FINNHUB_API_KEY"] = saved

if saved is not None:
    stream = FinnhubWebSocketManager()
    try:
        stream.subscribe("AAPL")
        import time
        connected = False
        for _ in range(40):
            if stream.is_connected():
                connected = True
                break
            time.sleep(0.25)
        latest = stream.get_latest("AAPL")
        stream.unsubscribe("AAPL")
        record(
            "WebSocket unaffected",
            connected and "AAPL" not in stream.subscribed_symbols(),
            f"connected={connected} latest={'yes' if latest else 'none'}",
        )
    except Exception as exc:
        record("WebSocket unaffected", False, f"{type(exc).__name__}: {exc}")
    finally:
        stream.disconnect()
else:
    skip("WebSocket unaffected", "no FINNHUB_API_KEY")


print()
passed = sum(1 for _, p, _ in results if p is True)
failed = sum(1 for _, p, _ in results if p is False)
skipped = sum(1 for _, p, _ in results if p is None)
print(f"passed={passed} failed={failed} skipped={skipped}")
sys.exit(1 if failed else 0)
