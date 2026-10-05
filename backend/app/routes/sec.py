from fastapi import APIRouter, HTTPException

from app.services.sec_client import SECClient


router = APIRouter(
    prefix="/api/sec",
    tags=["SEC"]
)


sec_client = SECClient()


@router.get("/company/{cik}")
def get_company(cik: str):

    try:

        data = sec_client.get_company_submissions(cik)

        return {
            "name": data["name"],
            "cik": data["cik"],
            "tickers": data["tickers"],
            "exchanges": data["exchanges"]
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.get("/company/{cik}/filings")
def get_filings(cik: str):

    try:

        filings = sec_client.get_annual_reports(cik)

        return {
            "cik": cik,
            "filings": filings
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )