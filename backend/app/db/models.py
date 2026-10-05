"""
Database schema.

Company            -- one row per ticker (US companies for now)
SECFiling          -- SEC EDGAR filing metadata + where its parsed text lives on disk
MarketQuote        -- latest yfinance quote snapshot (one row per company, upserted)
PriceHistoryBar    -- yfinance daily OHLCV bars (one row per company+date)
"""

from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Company(Base):
    __tablename__ = "companies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(16), unique=True, index=True)
    cik: Mapped[str | None] = mapped_column(String(10), nullable=True, index=True)
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    exchange: Mapped[str] = mapped_column(String(8), default="US")
    sector: Mapped[str | None] = mapped_column(String(128), nullable=True)
    industry: Mapped[str | None] = mapped_column(String(128), nullable=True)
    currency: Mapped[str | None] = mapped_column(String(8), nullable=True)
    last_synced_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    filings: Mapped[list["SECFiling"]] = relationship(
        back_populates="company", cascade="all, delete-orphan"
    )
    quote: Mapped["MarketQuote | None"] = relationship(
        back_populates="company", cascade="all, delete-orphan", uselist=False
    )
    price_history: Mapped[list["PriceHistoryBar"]] = relationship(
        back_populates="company", cascade="all, delete-orphan"
    )


class SECFiling(Base):
    __tablename__ = "sec_filings"
    __table_args__ = (UniqueConstraint("accession_number", name="uq_accession_number"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"), index=True)

    form: Mapped[str] = mapped_column(String(16))  # "10-K", "10-Q", "8-K"
    accession_number: Mapped[str] = mapped_column(String(32))
    filing_date: Mapped[str] = mapped_column(String(16))
    report_date: Mapped[str | None] = mapped_column(String(16), nullable=True)
    filing_url: Mapped[str] = mapped_column(String(512))

    # Where the raw HTML / cleaned text live on disk, once downloaded.
    # Kept as files (not DB blobs) so the existing RAG indexing loop
    # in main.py can keep reading them directly.
    local_html_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    local_text_path: Mapped[str | None] = mapped_column(String(512), nullable=True)

    downloaded_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    company: Mapped["Company"] = relationship(back_populates="filings")


class MarketQuote(Base):
    """Latest yfinance quote snapshot. One row per company -- upserted on sync."""

    __tablename__ = "market_quotes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(
        ForeignKey("companies.id"), unique=True, index=True
    )

    last_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    previous_close: Mapped[float | None] = mapped_column(Float, nullable=True)
    change: Mapped[float | None] = mapped_column(Float, nullable=True)
    change_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    day_high: Mapped[float | None] = mapped_column(Float, nullable=True)
    day_low: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume: Mapped[float | None] = mapped_column(Float, nullable=True)
    market_cap: Mapped[float | None] = mapped_column(Float, nullable=True)

    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    company: Mapped["Company"] = relationship(back_populates="quote")


class PriceHistoryBar(Base):
    """One daily OHLCV bar. Unique per (company, date)."""

    __tablename__ = "price_history"
    __table_args__ = (UniqueConstraint("company_id", "date", name="uq_company_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"), index=True)

    date: Mapped[str] = mapped_column(String(10))  # "YYYY-MM-DD"
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[int] = mapped_column(Integer)

    company: Mapped["Company"] = relationship(back_populates="price_history")