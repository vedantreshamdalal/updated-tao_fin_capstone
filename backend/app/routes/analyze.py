import threading

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.db.database import SessionLocal
from app.services import company_service
from app.services.market_snapshot import build_market_snapshot
from app.services.rag_index import build_tao_pipeline

router = APIRouter(
    prefix="/api/analyze",
    tags=["Analyze"]
)

# Serializes on-demand sync + index rebuild so two first-time
# requests for the same ticker cannot create duplicate work/rows.
_ensure_lock = threading.Lock()


class AnalyzeRequest(BaseModel):
    question: str


def _ensure_pipeline(request: Request, ticker: str):
    """
    Make sure the named company has filings and is in the RAG index.

    Only synchronizes when the company record is missing or has no
    parsed filings. Other failures (SEC, embeddings) stay as 503.
    Unknown tickers are rejected before any row is created.
    """

    cik = company_service.resolve_cik(ticker)
    if cik is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown ticker '{ticker}'",
        )

    db = SessionLocal()
    try:
        with _ensure_lock:
            if company_service.needs_filing_sync(db, ticker):
                try:
                    company_service.full_sync(db, ticker)
                except ValueError as exc:
                    raise HTTPException(status_code=404, detail=str(exc)) from exc
                except Exception as exc:
                    raise HTTPException(
                        status_code=503,
                        detail=f"Could not synchronize {ticker}: {exc}",
                    ) from exc

            pipeline = getattr(request.app.state, "tao_pipeline", None)
            indexed = getattr(request.app.state, "indexed_tickers", set()) or set()

            if pipeline is None or ticker not in indexed:
                pipeline, indexed = build_tao_pipeline()
                request.app.state.tao_pipeline = pipeline
                request.app.state.indexed_tickers = indexed

            if pipeline is None:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"Company '{ticker}' was synchronized but the "
                        "RAG index could not be built from its filings."
                    ),
                )

            return pipeline
    finally:
        db.close()


@router.post("")
def analyze(request: Request, body: AnalyzeRequest):
    if not body.question or not body.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    try:
        ticker = company_service.find_ticker_in_question(body.question)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if ticker is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Could not determine a US company ticker from the question. "
                "Include a listed ticker such as AAPL."
            ),
        )

    pipeline = _ensure_pipeline(request, ticker)
    snapshot = build_market_snapshot(ticker)

    try:
        return pipeline.run(body.question, market_snapshot=snapshot)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
