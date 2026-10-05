from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Company
from app.services import company_service

router = APIRouter(prefix="/api/companies", tags=["Companies"])


def _company_summary(company: Company) -> dict:
    return {
        "ticker": company.ticker,
        "cik": company.cik,
        "name": company.name,
        "sector": company.sector,
        "industry": company.industry,
        "currency": company.currency,
        "last_synced_at": company.last_synced_at.isoformat() if company.last_synced_at else None,
        "num_filings": len(company.filings),
        "has_quote": company.quote is not None,
    }


@router.get("")
def list_companies(db: Session = Depends(get_db)):
    companies = db.query(Company).order_by(Company.ticker).all()
    return {"companies": [_company_summary(c) for c in companies]}


@router.get("/{ticker}")
def get_company(ticker: str, db: Session = Depends(get_db)):
    company = db.query(Company).filter(Company.ticker == ticker.upper()).one_or_none()
    if company is None:
        raise HTTPException(status_code=404, detail=f"'{ticker}' has not been synced yet")

    quote = company.quote
    return {
        **_company_summary(company),
        "filings": [
            {
                "form": f.form,
                "filing_date": f.filing_date,
                "report_date": f.report_date,
                "filing_url": f.filing_url,
                "indexed": f.local_text_path is not None,
            }
            for f in company.filings
        ],
        "quote": (
            {
                "last_price": quote.last_price,
                "previous_close": quote.previous_close,
                "change": quote.change,
                "change_pct": quote.change_pct,
                "day_high": quote.day_high,
                "day_low": quote.day_low,
                "volume": quote.volume,
                "market_cap": quote.market_cap,
                "fetched_at": quote.fetched_at.isoformat(),
            }
            if quote
            else None
        ),
    }


@router.post("/{ticker}/sync")
def sync_company(ticker: str, db: Session = Depends(get_db)):
    """
    Full sync for a US ticker: resolve CIK, download + parse recent
    10-K/10-Q filings, and fetch the latest yfinance quote + history.
    Restart the server afterwards to pick up new filings in the RAG index
    (there's no hot-reload of the index yet).
    """
    try:
        result = company_service.full_sync(db, ticker)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    return {
        "company": _company_summary(result["company"]),
        "filings_synced": len(result["filings"]),
    }