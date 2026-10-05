"""
Company data sync: pulls SEC EDGAR filing metadata + text, and
yfinance quote/history data, for a US ticker, and persists all of it
via the DB models in app.db.models.

    upsert_company(db, ticker)   -> resolves ticker -> CIK, stores name/sector/etc.
    sync_filings(db, ticker)     -> downloads + parses 10-K/10-Q filings, records them
    sync_market_data(db, ticker) -> stores latest quote + recent daily bars

SEC's ticker -> CIK mapping is cached to disk (data/cache/company_tickers.json)
since it's a single ~1MB file covering every US public company, and SEC
asks that you not refetch it on every request.
"""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import requests
import yfinance as yf
from dotenv import load_dotenv
from sqlalchemy.orm import Session

from app.db.models import Company, MarketQuote, PriceHistoryBar, SECFiling
from app.services.market_data import get_history, get_quote
from app.services.sec_client import SECClient
from app.services.sec_parser import parse_filing_html

load_dotenv()

_CACHE_DIR = Path("data/cache")
_SEC_DIR = Path("data/sec")
_TICKER_MAP_PATH = _CACHE_DIR / "company_tickers.json"
_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------
# Ticker -> CIK resolution
# --------------------------------------------------------------------------

def _load_ticker_cik_map() -> dict[str, str]:
    """
    Load SEC's ticker -> CIK mapping, using a local cache if present.
    Returns {"AAPL": "0000320193", ...} (CIK zero-padded to 10 digits).
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if _TICKER_MAP_PATH.exists():
        raw = json.loads(_TICKER_MAP_PATH.read_text(encoding="utf-8"))
    else:
        user_agent = os.getenv("SEC_USER_AGENT")
        if not user_agent:
            raise ValueError("SEC_USER_AGENT is not configured in .env")

        response = requests.get(
            _TICKER_MAP_URL,
            headers={"User-Agent": user_agent},
            timeout=30,
        )
        response.raise_for_status()
        raw = response.json()

        _TICKER_MAP_PATH.write_text(json.dumps(raw), encoding="utf-8")

    # raw is like {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
    return {
        entry["ticker"].upper(): str(entry["cik_str"]).zfill(10)
        for entry in raw.values()
    }


def resolve_cik(ticker: str) -> Optional[str]:
    """Return the zero-padded CIK for a ticker, or None if not found."""
    mapping = _load_ticker_cik_map()
    return mapping.get(ticker.upper())


_QUESTION_STOPWORDS = frozenset({
    "A", "AN", "AND", "ARE", "DID", "DOES", "FOR", "FROM", "HAD", "HAS",
    "HOW", "IS", "ITS", "OF", "ON", "OR", "THE", "THIS", "THAT", "WAS",
    "WHAT", "WHEN", "WHY", "WITH", "YEAR", "FY",
})

_TOKEN_RE = re.compile(r"\b[A-Za-z]{1,10}\b")


def find_ticker_in_question(question: str) -> Optional[str]:
    """
    Return a US ticker mentioned in the question, validated against
    the SEC ticker map. Common English words are ignored. Returns
    None when no listed ticker or company name is present.
    """

    if not question or not question.strip():
        return None

    mapping = _load_ticker_cik_map()
    unknown_explicit = None

    matches = list(_TOKEN_RE.finditer(question))
    matches.sort(key=lambda m: len(m.group()), reverse=True)

    for match in matches:
        token = match.group()
        ticker = token.upper()
        if ticker in _QUESTION_STOPWORDS:
            continue
        if ticker in mapping:
            # Lowercase English words such as "price" / "stock" collide
            # with real tickers. Only treat a map hit as a ticker when
            # the user wrote it like a symbol (all caps).
            if token.isupper():
                return ticker
            continue
        # All-caps tokens are treated as an explicit ticker attempt so
        # INVALIDXYZ becomes 404 rather than "no ticker found".
        if token.isupper() and 2 <= len(token) <= 10 and unknown_explicit is None:
            unknown_explicit = ticker

    if unknown_explicit:
        return unknown_explicit

    # Company-name fallback: "Apple's revenue" -> AAPL
    raw = json.loads(_TICKER_MAP_PATH.read_text(encoding="utf-8")) if _TICKER_MAP_PATH.exists() else {}
    question_lower = question.lower()
    best = None
    best_len = 0

    for entry in raw.values():
        title = (entry.get("title") or "").strip()
        ticker = (entry.get("ticker") or "").upper()
        if not title or ticker not in mapping:
            continue

        title_lower = title.lower()
        first_word = re.sub(r"[^a-z0-9]+", "", title.split()[0].lower())
        if first_word.upper() in _QUESTION_STOPWORDS or len(first_word) < 4:
            continue
        if first_word in {
            "stock", "price", "share", "current", "market", "group",
            "holdings", "company", "corp", "incorporated", "financial",
        }:
            continue

        if re.search(r"\b" + re.escape(first_word) + r"\b", question_lower):
            if len(first_word) > best_len:
                best = ticker
                best_len = len(first_word)

        if title_lower in question_lower and len(title_lower) > best_len:
            best = ticker
            best_len = len(title_lower)

    return best or unknown_explicit


def needs_filing_sync(db: Session, ticker: str) -> bool:
    """True when the company is missing or has no parsed SEC filings."""

    company = (
        db.query(Company)
        .filter(Company.ticker == ticker.upper().strip())
        .one_or_none()
    )
    if company is None:
        return True
    return not any(filing.local_text_path for filing in company.filings)


# --------------------------------------------------------------------------
# Company upsert
# --------------------------------------------------------------------------

def upsert_company(db: Session, ticker: str) -> Company:
    """
    Resolve CIK + fetch basic profile info (name, sector, industry,
    currency) from yfinance, and upsert into the companies table.
    """
    ticker = ticker.upper().strip()

    company = db.query(Company).filter(Company.ticker == ticker).one_or_none()
    if company is None:
        company = Company(ticker=ticker, exchange="US")
        db.add(company)

    company.cik = resolve_cik(ticker)

    try:
        info = yf.Ticker(ticker).info
    except Exception:
        info = {}

    company.name = info.get("longName") or info.get("shortName") or company.name
    company.sector = info.get("sector")
    company.industry = info.get("industry")
    company.currency = info.get("currency")
    company.last_synced_at = _now()

    db.commit()
    db.refresh(company)
    return company


# --------------------------------------------------------------------------
# SEC filings sync
# --------------------------------------------------------------------------

def sync_filings(
    db: Session,
    ticker: str,
    forms: Iterable[str] = ("10-K", "10-Q"),
    limit_per_form: int = 1,
) -> list[SECFiling]:
    """
    Download + parse the most recent filing(s) of each requested form
    type for this company, record their metadata in the DB, and save
    both raw HTML and cleaned text to data/sec/ (matching filenames,
    e.g. AAPL_10-K_000032019324000123.html / .txt) so the existing
    RAG-indexing loop in main.py can pick them up unchanged.
    """
    ticker = ticker.upper().strip()
    company = db.query(Company).filter(Company.ticker == ticker).one_or_none()
    if company is None:
        company = upsert_company(db, ticker)

    if not company.cik:
        raise ValueError(f"No SEC CIK found for ticker '{ticker}'")

    _SEC_DIR.mkdir(parents=True, exist_ok=True)

    sec_client = SECClient()
    submissions = sec_client.get_company_submissions(company.cik)
    recent = submissions["filings"]["recent"]

    forms = set(forms)
    per_form_count: dict[str, int] = {f: 0 for f in forms}
    saved: list[SECFiling] = []

    for i in range(len(recent["form"])):
        form = recent["form"][i]

        if form not in forms or per_form_count[form] >= limit_per_form:
            continue

        accession = recent["accessionNumber"][i]

        already = (
            db.query(SECFiling)
            .filter(SECFiling.accession_number == accession)
            .one_or_none()
        )
        if already is not None:
            per_form_count[form] += 1
            saved.append(already)
            continue

        document = recent["primaryDocument"][i]
        filing_url = sec_client.build_filing_url(company.cik, accession, document)

        html = sec_client.download_filing(company.cik, accession, document)
        text = parse_filing_html(html)

        base = f"{ticker}_{form}_{accession.replace('-', '')}"
        html_path = _SEC_DIR / f"{base}.html"
        text_path = _SEC_DIR / f"{base}.txt"
        html_path.write_text(html, encoding="utf-8")
        text_path.write_text(text, encoding="utf-8")

        filing = SECFiling(
            company_id=company.id,
            form=form,
            accession_number=accession,
            filing_date=recent["filingDate"][i],
            report_date=recent["reportDate"][i],
            filing_url=filing_url,
            local_html_path=str(html_path),
            local_text_path=str(text_path),
            downloaded_at=_now(),
        )
        db.add(filing)
        db.commit()
        db.refresh(filing)

        saved.append(filing)
        per_form_count[form] += 1

    return saved


# --------------------------------------------------------------------------
# Market data sync
# --------------------------------------------------------------------------

def sync_market_data(db: Session, ticker: str, history_period: str = "6mo") -> MarketQuote:
    """
    Fetch the latest yfinance quote + recent daily bars for this
    company and upsert them into market_quotes / price_history.
    """
    ticker = ticker.upper().strip()
    company = db.query(Company).filter(Company.ticker == ticker).one_or_none()
    if company is None:
        company = upsert_company(db, ticker)

    quote_data = get_quote(ticker, "US")

    quote = db.query(MarketQuote).filter(MarketQuote.company_id == company.id).one_or_none()
    if quote is None:
        quote = MarketQuote(company_id=company.id)
        db.add(quote)

    quote.last_price = quote_data["last_price"]
    quote.previous_close = quote_data["previous_close"]
    quote.change = quote_data["change"]
    quote.change_pct = quote_data["change_pct"]
    quote.day_high = quote_data["day_high"]
    quote.day_low = quote_data["day_low"]
    quote.volume = quote_data["volume"]
    quote.market_cap = quote_data["market_cap"]
    quote.fetched_at = _now()

    history_data = get_history(ticker, "US", period=history_period)
    for point in history_data["points"]:
        bar = (
            db.query(PriceHistoryBar)
            .filter(
                PriceHistoryBar.company_id == company.id,
                PriceHistoryBar.date == point["date"],
            )
            .one_or_none()
        )
        if bar is None:
            bar = PriceHistoryBar(company_id=company.id, date=point["date"])
            db.add(bar)

        bar.open = point["open"]
        bar.high = point["high"]
        bar.low = point["low"]
        bar.close = point["close"]
        bar.volume = point["volume"]

    db.commit()
    db.refresh(quote)
    return quote


# --------------------------------------------------------------------------
# Convenience: do everything for a ticker in one call
# --------------------------------------------------------------------------

def full_sync(db: Session, ticker: str) -> dict:
    """Upsert company profile + SEC filings + market data in one call."""
    company = upsert_company(db, ticker)
    filings = sync_filings(db, ticker)
    quote = sync_market_data(db, ticker)

    return {
        "company": company,
        "filings": filings,
        "quote": quote,
    }