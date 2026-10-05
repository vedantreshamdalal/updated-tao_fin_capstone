"""
Market data access layer.

Provider routing for current quotes:

    US        -> Finnhub REST /quote, falling back to yfinance
    NSE/BSE   -> yfinance (Finnhub does not cover these)

History and financial ratios stay on yfinance.

Callers only ever see the quote schema defined by get_quote(); which
provider served a given quote is an implementation detail of this
module. Quotes are cached in-memory for 5 seconds, so a returned
quote is fresh to within roughly 5 seconds plus network/API latency.

Finnhub's /quote carries no volume, market cap or currency, so those
three are sourced from yfinance on a separate 60-second cache. Price
fields therefore track the 5-second quote TTL while those three can
trail by up to a minute.
"""

import os
import threading
import time
from pathlib import Path
from typing import Dict, Literal, Optional

import requests
import yfinance as yf
from dotenv import load_dotenv

# Always load backend/.env, regardless of the process working directory.
# Does not override variables already in the environment.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

Exchange = Literal["US", "NSE", "BSE"]

_SUFFIX = {
    "US": "",
    "NSE": ".NS",
    "BSE": ".BO",
}

_FINNHUB_QUOTE_URL = "https://finnhub.io/api/v1/quote"
_FINNHUB_TIMEOUT_SECONDS = 5

_QUOTE_CACHE_TTL_SECONDS = 5

# Finnhub's /quote endpoint carries no volume, market cap or currency,
# so those three are sourced from yfinance and cached for longer than
# the quote itself. 60s keeps the schema fully populated at one extra
# call per symbol per minute; the price fields still come from Finnhub
# and are never more than _QUOTE_CACHE_TTL_SECONDS old.
_SUPPLEMENT_TTL_SECONDS = 60

# (resolved_symbol, exchange) -> (stored_at_monotonic, quote_dict)
_quote_cache: Dict[tuple, tuple] = {}

# (resolved_symbol, exchange) -> provider metadata for the last fetch.
# Kept out of the quote dict so the public schema stays unchanged.
_quote_meta: Dict[tuple, Dict] = {}

# (resolved_symbol, exchange) -> (stored_at_monotonic, {"volume":, "market_cap":, "currency":})
_yf_supplement: Dict[tuple, tuple] = {}

_cache_lock = threading.Lock()


def _resolve_symbol(symbol: str, exchange: Exchange) -> str:
    symbol = symbol.upper().strip()
    suffix = _SUFFIX.get(exchange, "")

    if suffix and not symbol.endswith(suffix):
        symbol = f"{symbol}{suffix}"

    return symbol


def _finnhub_api_key() -> Optional[str]:
    """Read at call time so tests can override the environment."""
    key = os.getenv("FINNHUB_API_KEY")
    return key.strip() if key and key.strip() else None


# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------

def _fast_info_value(fast_info, *keys):
    """
    Read a field from yfinance's FastInfo.

    FastInfo.get() resolves only camelCase keys ("marketCap"), while
    attribute access resolves only snake_case ("market_cap"), so a
    plain .get("market_cap") silently yields None. Try both spellings
    and both access styles before giving up.
    """

    for key in keys:
        try:
            value = fast_info.get(key)
        except Exception:
            value = None

        if value is not None:
            return value

    for key in keys:
        try:
            value = getattr(fast_info, key)
        except Exception:
            continue

        if value is not None:
            return value

    return None


def _get_yf_supplement(resolved: str, exchange: Exchange) -> Dict:
    """
    volume / market_cap / currency for a Finnhub-served quote.

    Finnhub's /quote does not carry these, so they come from a single
    yfinance fast_info call, cached for _SUPPLEMENT_TTL_SECONDS. A
    failure here must never fail the quote: we fall back to the last
    known values, or to empty, rather than fabricating anything.
    """

    key = (resolved, exchange)

    with _cache_lock:
        cached = _yf_supplement.get(key)

        if cached is not None and (time.monotonic() - cached[0]) < _SUPPLEMENT_TTL_SECONDS:
            return cached[1]

    try:
        fast_info = yf.Ticker(resolved).fast_info

        supplement = {
            "volume": _fast_info_value(fast_info, "last_volume", "lastVolume", "volume"),
            "market_cap": _fast_info_value(fast_info, "market_cap", "marketCap"),
            "currency": _fast_info_value(fast_info, "currency"),
        }
    except Exception as e:
        print(
            f"[market_data] Could not refresh volume/market_cap for "
            f"{resolved} from yfinance: {type(e).__name__}"
        )
        return cached[1] if cached is not None else {}

    with _cache_lock:
        _yf_supplement[key] = (time.monotonic(), supplement)

    print(f"[market_data] Refreshed volume/market_cap for {resolved} from yfinance")

    return supplement


def _get_finnhub_quote(resolved: str, exchange: Exchange) -> Optional[Dict]:
    """
    Fetch a current quote from Finnhub and adapt it into the shared
    quote schema. Returns None on any failure so the caller can fall
    back to yfinance -- a market data outage should degrade, not raise.

    The API token is passed as a request parameter and is never logged.
    """

    api_key = _finnhub_api_key()

    if api_key is None:
        return None

    print(f"[market_data] Finnhub quote requested for {resolved}")

    try:
        response = requests.get(
            _FINNHUB_QUOTE_URL,
            params={"symbol": resolved, "token": api_key},
            timeout=_FINNHUB_TIMEOUT_SECONDS,
        )
    except requests.exceptions.Timeout:
        print(f"[market_data] Finnhub timed out for {resolved}")
        return None
    except requests.exceptions.RequestException as e:
        # Deliberately report only the exception class: a RequestException's
        # string form can embed the request URL, which carries the token.
        print(f"[market_data] Finnhub request failed for {resolved}: {type(e).__name__}")
        return None

    if response.status_code == 429:
        print(f"[market_data] Finnhub rate-limited {resolved}")
        return None

    if not response.ok:
        print(f"[market_data] Finnhub HTTP {response.status_code} for {resolved}")
        return None

    try:
        payload = response.json()
    except ValueError:
        print(f"[market_data] Finnhub returned non-JSON for {resolved}")
        return None

    if not isinstance(payload, dict):
        print(f"[market_data] Finnhub returned unexpected payload for {resolved}")
        return None

    def num(key: str) -> Optional[float]:
        value = payload.get(key)
        return float(value) if isinstance(value, (int, float)) else None

    last_price = num("c")
    previous_close = num("pc")

    # Finnhub answers unknown or unsupported symbols with an all-zero
    # body rather than an error status, so treat that as a miss.
    if not last_price and not previous_close:
        print(f"[market_data] Finnhub returned an empty quote for {resolved}")
        return None

    change = num("d")
    change_pct = num("dp")

    if change is None and last_price is not None and previous_close:
        change = round(last_price - previous_close, 4)

    if change_pct is None and change is not None and previous_close:
        change_pct = round((change / previous_close) * 100, 4)

    # Quote timestamp (epoch seconds). Used for the freshness check
    # below; not surfaced in the public schema.
    quote_timestamp = num("t")

    supplement = _get_yf_supplement(resolved, exchange)

    print(f"[market_data] Finnhub quote successful for {resolved}")
    print(f"[market_data] provider=finnhub symbol={resolved}")

    return {
        "quote": {
            "symbol": resolved,
            "exchange": exchange,
            "last_price": last_price,
            "previous_close": previous_close,
            "change": change,
            "change_pct": change_pct,
            "day_high": num("h"),
            "day_low": num("l"),
            # Absent from Finnhub's /quote payload -- sourced from
            # yfinance above, never invented.
            "volume": supplement.get("volume"),
            "market_cap": supplement.get("market_cap"),
            # US listings quote in USD; prefer the observed value.
            "currency": supplement.get("currency") or "USD",
        },
        "quote_timestamp": quote_timestamp,
    }


def _get_yfinance_quote(resolved: str, exchange: Exchange) -> Dict:
    """The original yfinance quote path, unchanged in behaviour."""

    ticker = yf.Ticker(resolved)

    try:
        fast_info = ticker.fast_info
    except Exception as e:
        raise RuntimeError(
            f"Could not fetch quote for '{resolved}': {e}"
        ) from e

    last_price = _fast_info_value(fast_info, "last_price", "lastPrice")
    previous_close = _fast_info_value(fast_info, "previous_close", "previousClose")

    change = None
    change_pct = None

    if last_price is not None and previous_close:
        change = round(last_price - previous_close, 4)
        change_pct = round((change / previous_close) * 100, 4)

    volume = _fast_info_value(fast_info, "last_volume", "lastVolume", "volume")
    market_cap = _fast_info_value(fast_info, "market_cap", "marketCap")
    currency = _fast_info_value(fast_info, "currency")

    # Remember the fields Finnhub's /quote cannot provide, so a later
    # Finnhub-served quote can reuse them without its own lookup.
    with _cache_lock:
        _yf_supplement[(resolved, exchange)] = (
            time.monotonic(),
            {"volume": volume, "market_cap": market_cap, "currency": currency},
        )

    return {
        "symbol": resolved,
        "exchange": exchange,
        "last_price": last_price,
        "previous_close": previous_close,
        "change": change,
        "change_pct": change_pct,
        "day_high": _fast_info_value(fast_info, "day_high", "dayHigh"),
        "day_low": _fast_info_value(fast_info, "day_low", "dayLow"),
        "volume": volume,
        "market_cap": market_cap,
        "currency": currency,
    }


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def get_quote(symbol: str, exchange: Exchange = "US") -> Dict:
    """
    Current quote for a symbol.

    US symbols are served by Finnhub when FINNHUB_API_KEY is set,
    falling back to yfinance on any failure. NSE/BSE always use
    yfinance. Results are cached for 5 seconds per (symbol, exchange).
    """

    resolved = _resolve_symbol(symbol, exchange)
    cache_key = (resolved, exchange)

    with _cache_lock:
        cached = _quote_cache.get(cache_key)

        if cached is not None and (time.monotonic() - cached[0]) < _QUOTE_CACHE_TTL_SECONDS:
            print(f"[market_data] Returning cached quote for {resolved}")
            return dict(cached[1])

    quote = None
    provider = None
    quote_timestamp = None

    if exchange == "US" and _finnhub_api_key() is not None:
        finnhub_result = _get_finnhub_quote(resolved, exchange)

        if finnhub_result is not None:
            quote = finnhub_result["quote"]
            quote_timestamp = finnhub_result["quote_timestamp"]
            provider = "finnhub"
        else:
            print(
                f"[market_data] Finnhub failed for {resolved}; "
                f"falling back to yfinance"
            )

    if quote is None:
        # Reached for NSE/BSE, a missing API key, or any Finnhub failure.
        quote = _get_yfinance_quote(resolved, exchange)

        provider = (
            "yfinance_fallback"
            if exchange == "US" and _finnhub_api_key() is not None
            else "yfinance"
        )

        print(f"[market_data] provider={provider} symbol={resolved}")

    # Only successful fetches are cached; failures raise out of
    # _get_yfinance_quote before reaching this point.
    with _cache_lock:
        _quote_cache[cache_key] = (time.monotonic(), dict(quote))

        _quote_meta[cache_key] = {
            "provider": provider,
            "quote_timestamp": quote_timestamp,
            "fetched_at": time.time(),
        }

    return quote


def get_last_quote_meta(symbol: str, exchange: Exchange = "US") -> Optional[Dict]:
    """
    Which provider served the most recent quote for this symbol, plus
    Finnhub's own quote timestamp when available.

    Diagnostics only -- deliberately separate from get_quote() so the
    quote schema that the API routes and company_service depend on
    stays exactly as it was.
    """

    return _quote_meta.get((_resolve_symbol(symbol, exchange), exchange))


def clear_quote_cache() -> None:
    """Drop all cached quotes. Intended for tests."""

    with _cache_lock:
        _quote_cache.clear()


def get_financial_ratios(symbol: str, exchange: Exchange = "US") -> Dict:
    """
    Fallback source for named margin ratios when the indexed SEC
    filings don't contain the underlying line items. yfinance's
    .info exposes these as trailing-twelve-month (TTM) figures, which
    is NOT the same thing as a specific fiscal year's audited figure
    -- callers must label this distinction clearly to the user.
    """

    resolved = _resolve_symbol(symbol, exchange)

    try:
        info = yf.Ticker(resolved).info
    except Exception as e:
        raise RuntimeError(f"Could not fetch financial ratios for '{resolved}': {e}") from e

    def pct(key: str):
        v = info.get(key)
        return round(v * 100, 2) if v is not None else None

    return {
        "symbol": resolved,
        "operating_margin_pct": pct("operatingMargins"),
        "gross_margin_pct": pct("grossMargins"),
        "profit_margin_pct": pct("profitMargins"),
        "period": "trailing_twelve_months",
        "note": (
            "Trailing-twelve-month figures from Yahoo Finance, not tied "
            "to a specific fiscal year filing."
        ),
    }


def get_history(symbol: str, exchange: Exchange = "US", period: str = "1mo") -> Dict:
    """period examples: '1d','5d','1mo','3mo','6mo','1y','ytd','max'"""

    resolved = _resolve_symbol(symbol, exchange)
    ticker = yf.Ticker(resolved)

    hist = ticker.history(period=period)

    if hist.empty:
        raise RuntimeError(f"No historical data found for '{resolved}'")

    return {
        "symbol": resolved,
        "exchange": exchange,
        "period": period,
        "points": [
            {
                "date": str(index.date()),
                "open": round(float(row["Open"]), 4),
                "high": round(float(row["High"]), 4),
                "low": round(float(row["Low"]), 4),
                "close": round(float(row["Close"]), 4),
                "volume": int(row["Volume"]),
            }
            for index, row in hist.iterrows()
        ],
    }
