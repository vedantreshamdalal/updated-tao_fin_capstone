# backend/app/routes/market.py

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Literal

from app.services.finnhub_websocket import get_stream_manager
from app.services.market_data import get_history, get_quote

router = APIRouter(
    prefix="/api/market",
    tags=["Market"]
)

Exchange = Literal["US", "NSE", "BSE"]


class StreamSymbolRequest(BaseModel):
    symbol: str = Field(..., min_length=1)


@router.get("/quote/{symbol}")
def quote(symbol: str, exchange: Exchange = Query("US")):
    try:
        return get_quote(symbol, exchange)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/history/{symbol}")
def history(symbol: str, exchange: Exchange = Query("US"), period: str = Query("1mo")):
    try:
        return get_history(symbol, exchange, period)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/stream/status")
def stream_status():
    """Connection, subscriptions, and freshness — never the API key."""
    return get_stream_manager().status()


@router.get("/stream/{symbol}")
def stream_latest(symbol: str):
    manager = get_stream_manager()
    latest = manager.get_latest(symbol)
    if latest is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No live trade yet for '{symbol.upper()}'. "
                "Subscribe first, or wait for a trade (none arrive while the market is closed)."
            ),
        )
    return latest


@router.post("/stream/subscribe")
def stream_subscribe(body: StreamSymbolRequest):
    return get_stream_manager().subscribe(body.symbol)


@router.post("/stream/unsubscribe")
def stream_unsubscribe(body: StreamSymbolRequest):
    return get_stream_manager().unsubscribe(body.symbol)