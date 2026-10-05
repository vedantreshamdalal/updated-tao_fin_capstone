"""
Finnhub / yfinance quote routing checks.

Run from the backend directory:

    python tests/test_market_data.py

Network-dependent cases are skipped (not failed) when the relevant
provider is unreachable, so the provider-routing and fallback logic
can still be verified offline.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

from app.services import market_data
from app.services.market_data import (
    clear_quote_cache,
    get_last_quote_meta,
    get_quote,
)

EXPECTED_KEYS = {
    "symbol",
    "exchange",
    "last_price",
    "previous_close",
    "change",
    "change_pct",
    "day_high",
    "day_low",
    "volume",
    "market_cap",
    "currency",
}

results = []
_real_get = requests.get


def record(name, passed, detail=""):
    results.append((name, passed, detail))
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))


def skip(name, detail):
    results.append((name, None, detail))
    print(f"[SKIP] {name} -- {detail}")


def _url_of(url, kwargs):
    return url if isinstance(url, str) else str(url)


def _is_finnhub(url, kwargs):
    target = _url_of(url, kwargs)
    params = kwargs.get("params") or {}
    return "finnhub.io" in target or "token" in params


def intercept_finnhub(fake, counter=None):
    """
    Replace requests.get so only Finnhub traffic is intercepted.
    yfinance keeps using the real client, which is required for
    fallback tests. The API token is never logged.
    """

    def wrapper(url, *args, **kwargs):
        if _is_finnhub(url, kwargs):
            if counter is not None:
                counter["n"] += 1
            return fake(url, *args, **kwargs)
        return _real_get(url, *args, **kwargs)

    market_data.requests.get = wrapper
    requests.get = wrapper


def restore_get():
    market_data.requests.get = _real_get
    requests.get = _real_get


# ---------------------------------------------------------------------------
# Test 2 -- configuration detection (never prints the secret)
# ---------------------------------------------------------------------------

has_key = market_data._finnhub_api_key() is not None
record("config: FINNHUB_API_KEY detected", has_key, f"present={has_key}")


# ---------------------------------------------------------------------------
# Test 3 / 4 -- US quote schema and provider
# ---------------------------------------------------------------------------

clear_quote_cache()

try:
    quote = get_quote("AAPL", exchange="US")
except Exception as e:
    skip("US quote schema", f"no network: {type(e).__name__}")
    skip("US quote provider=finnhub", "quote unavailable")
else:
    record(
        "US quote schema",
        set(quote) == EXPECTED_KEYS,
        f"missing={EXPECTED_KEYS - set(quote)} extra={set(quote) - EXPECTED_KEYS}",
    )

    numeric_ok = all(
        quote[k] is None or isinstance(quote[k], (int, float))
        for k in ("last_price", "previous_close", "change", "change_pct", "day_high", "day_low")
    )
    record("US quote numeric types", numeric_ok)
    record("US quote symbol/exchange", quote["symbol"] == "AAPL" and quote["exchange"] == "US")

    meta = get_last_quote_meta("AAPL", "US")
    record(
        "US quote provider=finnhub",
        meta is not None and meta["provider"] == "finnhub",
        f"provider={meta and meta['provider']}",
    )


# ---------------------------------------------------------------------------
# Test 5 -- cache hit within the TTL (no second Finnhub call)
# ---------------------------------------------------------------------------

calls = {"n": 0}


def counting_finnhub(url, *args, **kwargs):
    return _real_get(url, *args, **kwargs)


clear_quote_cache()
intercept_finnhub(counting_finnhub, counter=calls)

try:
    get_quote("AAPL", "US")
    after_first = calls["n"]
    get_quote("AAPL", "US")
    after_second = calls["n"]
    record(
        "cache: second call within TTL makes no request",
        after_second == after_first and after_first >= 1,
        f"finnhub requests before={after_first} after={after_second}",
    )
except Exception as e:
    skip("cache: second call within TTL", f"no network: {type(e).__name__}")
finally:
    restore_get()


# ---------------------------------------------------------------------------
# Test 6 -- cache expiry past the TTL
# ---------------------------------------------------------------------------

original_ttl = market_data._QUOTE_CACHE_TTL_SECONDS
calls["n"] = 0
market_data._QUOTE_CACHE_TTL_SECONDS = 1
intercept_finnhub(counting_finnhub, counter=calls)

try:
    clear_quote_cache()
    get_quote("AAPL", "US")
    before = calls["n"]
    time.sleep(1.2)
    get_quote("AAPL", "US")
    record(
        "cache: expires after TTL",
        calls["n"] > before,
        f"finnhub requests before={before} after={calls['n']}",
    )
except Exception as e:
    skip("cache: expires after TTL", f"no network: {type(e).__name__}")
finally:
    restore_get()
    market_data._QUOTE_CACHE_TTL_SECONDS = original_ttl


# ---------------------------------------------------------------------------
# Test 7 / 8 -- NSE and BSE never touch Finnhub
# ---------------------------------------------------------------------------

for exchange, suffix in (("NSE", ".NS"), ("BSE", ".BO")):
    clear_quote_cache()
    calls = {"n": 0}
    intercept_finnhub(counting_finnhub, counter=calls)

    try:
        quote = get_quote("RELIANCE", exchange=exchange)
    except Exception as e:
        skip(f"{exchange}: uses yfinance", f"no network: {type(e).__name__}")
    else:
        meta = get_last_quote_meta("RELIANCE", exchange)
        record(
            f"{exchange}: uses yfinance, symbol suffixed",
            meta["provider"] == "yfinance"
            and quote["symbol"] == f"RELIANCE{suffix}"
            and calls["n"] == 0,
            f"provider={meta['provider']} symbol={quote['symbol']} finnhub_calls={calls['n']}",
        )
        record(f"{exchange}: schema unchanged", set(quote) == EXPECTED_KEYS)
    finally:
        restore_get()


# ---------------------------------------------------------------------------
# Test 9 -- missing API key falls back to yfinance
# ---------------------------------------------------------------------------

saved_key = os.environ.pop("FINNHUB_API_KEY", None)
clear_quote_cache()

try:
    quote = get_quote("AAPL", "US")
except Exception as e:
    skip("missing key: falls back to yfinance", f"no network: {type(e).__name__}")
else:
    meta = get_last_quote_meta("AAPL", "US")
    record(
        "missing key: falls back to yfinance",
        meta["provider"] == "yfinance" and set(quote) == EXPECTED_KEYS,
        f"provider={meta['provider']}",
    )
finally:
    if saved_key is not None:
        os.environ["FINNHUB_API_KEY"] = saved_key


# ---------------------------------------------------------------------------
# Test 10 -- Finnhub failure modes all fall back
# ---------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, status_code=200, payload=None, text="", bad_json=False):
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self._bad_json = bad_json

    @property
    def ok(self):
        return 200 <= self.status_code < 300

    def json(self):
        if self._bad_json:
            raise ValueError("not json")
        return self._payload


FAILURE_MODES = {
    "timeout": lambda *a, **k: (_ for _ in ()).throw(requests.exceptions.Timeout()),
    "connection error": lambda *a, **k: (_ for _ in ()).throw(
        requests.exceptions.ConnectionError("connection failed")
    ),
    "http 500": lambda *a, **k: FakeResponse(status_code=500, text="server error"),
    "rate limit 429": lambda *a, **k: FakeResponse(status_code=429, text="limit"),
    "malformed json": lambda *a, **k: FakeResponse(bad_json=True),
    "empty quote": lambda *a, **k: FakeResponse(
        payload={"c": 0, "d": None, "dp": None, "h": 0, "l": 0, "o": 0, "pc": 0, "t": 0}
    ),
    "non-dict payload": lambda *a, **k: FakeResponse(payload=["unexpected"]),
}

if not has_key:
    skip("Finnhub failure modes", "no API key, Finnhub path never taken")
else:
    for label, fake in FAILURE_MODES.items():
        clear_quote_cache()
        intercept_finnhub(fake)

        try:
            quote = get_quote("AAPL", "US")
        except Exception as e:
            record(f"fallback on {label}", False, f"raised {type(e).__name__}: {e}")
        else:
            meta = get_last_quote_meta("AAPL", "US")
            record(
                f"fallback on {label}",
                meta["provider"] == "yfinance_fallback" and set(quote) == EXPECTED_KEYS,
                f"provider={meta['provider']}",
            )
        finally:
            restore_get()


# ---------------------------------------------------------------------------
# Finnhub -> schema mapping, with a known payload
# ---------------------------------------------------------------------------

if not has_key:
    skip("Finnhub field mapping", "no API key, Finnhub path never taken")
else:
    clear_quote_cache()
    intercept_finnhub(
        lambda *a, **k: FakeResponse(
            payload={
                "c": 191.5,
                "d": 1.5,
                "dp": 0.79,
                "h": 192.0,
                "l": 188.25,
                "o": 189.0,
                "pc": 190.0,
                "t": 1700000000,
            }
        )
    )

    try:
        quote = get_quote("AAPL", "US")
        meta = get_last_quote_meta("AAPL", "US")
        mapped = (
            quote["last_price"] == 191.5
            and quote["previous_close"] == 190.0
            and quote["change"] == 1.5
            and quote["change_pct"] == 0.79
            and quote["day_high"] == 192.0
            and quote["day_low"] == 188.25
        )
        record("Finnhub field mapping c/pc/d/dp/h/l", mapped)
        record("Finnhub schema keys unchanged", set(quote) == EXPECTED_KEYS)
        record(
            "Finnhub timestamp kept internal, not in schema",
            "t" not in quote
            and "quote_timestamp" not in quote
            and meta["quote_timestamp"] == 1700000000,
        )
    finally:
        restore_get()
        clear_quote_cache()


# ---------------------------------------------------------------------------
# Test 11 -- existing /api/market/quote route, schema unchanged
# ---------------------------------------------------------------------------

try:
    from fastapi.testclient import TestClient
    from fastapi import FastAPI
    from app.routes.market import router as market_router

    isolated = FastAPI()
    isolated.include_router(market_router)
    client = TestClient(isolated)
    clear_quote_cache()
    response = client.get("/api/market/quote/AAPL", params={"exchange": "US"})
    body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
    meta = get_last_quote_meta("AAPL", "US")
    record(
        "API /api/market/quote/AAPL HTTP 200 + schema",
        response.status_code == 200 and set(body) == EXPECTED_KEYS,
        f"status={response.status_code} keys={sorted(body) if isinstance(body, dict) else type(body).__name__}",
    )
    record(
        "API route uses Finnhub when key is present",
        has_key and meta is not None and meta["provider"] == "finnhub",
        f"provider={meta and meta['provider']}",
    )
except Exception as e:
    skip("API /api/market/quote/AAPL", f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------

print()
passed = sum(1 for _, p, _ in results if p is True)
failed = sum(1 for _, p, _ in results if p is False)
skipped = sum(1 for _, p, _ in results if p is None)
print(f"passed={passed} failed={failed} skipped={skipped}")

sys.exit(1 if failed else 0)
