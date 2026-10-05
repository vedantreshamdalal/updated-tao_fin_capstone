"""
TAO Pipeline

Orchestrates:

    question
        ↓
    query analyzer
        ↓
    deterministic financial resolver OR hybrid retrieval
        ↓
    reasoning LLM (only when necessary)
        ↓
    verifier
        ↓
    TAO controller
        ↓
    final answer

Named financial ratios such as operating margin, gross margin,
net margin and profit margin are resolved deterministically from
the indexed SEC tables.

IMPORTANT:
For fiscal-year questions, metrics_resolver.py uses annual 10-K
filings rather than 10-Q year-to-date figures.

For deterministic metrics, Ollama is skipped completely because
there is no reason to ask an LLM to reproduce arithmetic that the
system can calculate exactly.
"""

from __future__ import annotations

import re
import time
from typing import Dict, Optional

from app.services import metrics_resolver
from app.services.market_data import get_financial_ratios
from app.services.query_analyzer import analyze_query
from app.services.rag_pipeline import RAGPipeline
from app.services.tao_controller import (
    Action,
    TAOState,
    decide,
)
from app.services.verifier import verify


MAX_TOTAL_ITERATIONS = 5

YFINANCE_FALLBACK_CONFIDENCE_THRESHOLD = 0.5

_CURRENT_PRICE_PHRASES = (
    "current price",
    "stock price",
    "share price",
    "last price",
    "live price",
    "market price",
    "trading at",
    "last trade",
)


def _wants_current_price(question: str) -> bool:
    """True for live/spot price questions, not fiscal-year lookups."""

    if re.search(r"\b(?:19|20)\d{2}\b", question):
        return False
    q = question.lower()
    return any(phrase in q for phrase in _CURRENT_PRICE_PHRASES)


class TAOPipeline:

    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        company_index: Optional[Dict[str, str]] = None,
        allow_external_fallback: bool = True,
    ):
        self.rag_pipeline = rag_pipeline

        self.company_index = company_index or {}

        # Controls whether Yahoo Finance/external market data
        # can be used as a fallback.
        #
        # Normal application:
        #     True
        #
        # FinanceBench:
        #     False
        self.allow_external_fallback = allow_external_fallback

    def _guess_ticker(
        self,
        question: str,
    ) -> Optional[str]:

        q = question.lower()

        for ticker, name in self.company_index.items():

            ticker_lower = ticker.lower()

            if ticker_lower in q.split():
                return ticker

            if name:
                first_word = name.split()[0].lower()

                if first_word in q:
                    return ticker

        return None

    def _try_yfinance_fallback(
        self,
        question: str,
        metric: str,
    ) -> Optional[Dict]:

        ratio_key = (
            metrics_resolver
            .METRIC_TO_YFINANCE_KEY
            .get(metric)
        )

        ticker = self._guess_ticker(
            question
        )

        if ratio_key is None or ticker is None:
            return None

        try:
            ratios = get_financial_ratios(
                ticker
            )
        except Exception:
            return None

        value = ratios.get(
            ratio_key
        )

        if value is None:
            return None

        answer = (
            f"The indexed SEC filings don't contain the specific "
            f"line items needed to compute {ticker}'s {metric} "
            f"for an exact fiscal year, so this figure comes from "
            f"live market data instead: {ticker}'s {metric} is "
            f"approximately {value}% (Yahoo Finance, trailing "
            f"twelve months). This is not the same as an audited "
            f"fiscal-year figure from the 10-K/10-Q and may differ "
            f"slightly."
        )

        return {
            "answer": answer,
            "evidence": [
                {
                    "score": None,
                    "source": (
                        "Yahoo Finance (yfinance) - live market "
                        "data, not SEC filing"
                    ),
                    "filing": None,
                    "text_preview": (
                        f"{metric} (TTM) for "
                        f"{ticker}: {value}%"
                    ),
                }
            ],
            "verification": {
                "checks": {
                    "numerical": True,
                    "evidence": False,
                    "logical": True,
                    "financial_consistency": True,
                },
                "confidence": 0.75,
                "failed_checks": [
                    "evidence"
                ],
                "details": {
                    "source": "yfinance_fallback",
                    "ratios": ratios,
                },
            },
        }

    def _build_deterministic_answer(
        self,
        computed_metric: Dict,
    ) -> Dict:

        metric = computed_metric["metric"]
        year = computed_metric["year"]

        numerator_label = computed_metric[
            "numerator_label"
        ]

        numerator_value = computed_metric[
            "numerator_value"
        ]

        denominator_label = computed_metric[
            "denominator_label"
        ]

        denominator_value = computed_metric[
            "denominator_value"
        ]

        computed_value = computed_metric[
            "computed_value_pct"
        ]

        answer = (
            f"{metric.capitalize()} for fiscal year "
            f"{year} was {computed_value}%. "
            f"It was calculated as "
            f"{numerator_value:,.0f} divided by "
            f"{denominator_value:,.0f}, multiplied by 100."
        )

        # Keep the exact two source chunks as the evidence.
        evidence = []

        seen_filings = set()

        for chunk in computed_metric[
            "source_chunks"
        ]:

            metadata = chunk.get(
                "metadata",
                {}
            )

            filing = metadata.get(
                "filing",
                "Unknown"
            )

            # Avoid displaying the same chunk twice.
            key = (
                filing,
                chunk.get("text", "")
            )

            if key in seen_filings:
                continue

            seen_filings.add(key)

            evidence.append(
                {
                    "score": 1000.0,
                    "source": metadata.get(
                        "source",
                        "SEC EDGAR"
                    ),
                    "filing": filing,
                    "text_preview": chunk.get(
                        "text",
                        ""
                    )[:300],
                }
            )

        verification = {
            "checks": {
                "numerical": True,
                "evidence": True,
                "logical": True,
                "financial_consistency": True,
            },
            "confidence": 1.0,
            "failed_checks": [],
            "details": {
                "method": "deterministic_metrics_resolver",
                "metric": metric,
                "year": year,
                "numerator": {
                    "label": numerator_label,
                    "value": numerator_value,
                },
                "denominator": {
                    "label": denominator_label,
                    "value": denominator_value,
                },
                "formula": (
                    "numerator / denominator * 100"
                ),
            },
        }

        return {
            "answer": answer,
            "evidence": evidence,
            "verification": verification,
        }

    def _build_spot_price_answer(
        self,
        snapshot: Dict,
        analysis: Dict,
        state: TAOState,
        start: float,
    ) -> Dict:
        """Deterministic last-trade / quote answer. Does not call the LLM."""

        symbol = snapshot.get("symbol") or "the company"
        price = snapshot["price"]
        source = snapshot.get("source") or "market data"
        volume = snapshot.get("volume")

        source_label = {
            "finnhub_websocket": "live Finnhub trade stream",
            "finnhub_rest": "Finnhub REST quote",
            "yfinance": "Yahoo Finance",
        }.get(source, source)

        volume_bit = f" Volume of the last trade was {volume}." if volume is not None else ""
        answer = (
            f"{symbol}'s last price is {price} "
            f"({source_label}).{volume_bit}"
        )

        evidence = [
            {
                "score": None,
                "source": source_label,
                "filing": None,
                "page": None,
                "document": None,
                "text_preview": (
                    f"{symbol} price={price} source={source} "
                    f"timestamp={snapshot.get('timestamp')}"
                ),
            }
        ]

        verification = {
            "checks": {
                "numerical": True,
                "evidence": True,
                "logical": True,
                "financial_consistency": True,
            },
            "confidence": 1.0,
            "failed_checks": [],
            "details": {
                "method": "market_snapshot",
                "source": source,
                "is_stale": snapshot.get("is_stale"),
                "age_seconds": snapshot.get("age_seconds"),
                "timestamp": snapshot.get("timestamp"),
            },
        }

        return {
            "answer": answer,
            "evidence": evidence,
            "verification": verification,
            "query_analysis": analysis,
            "tao": state.to_dict(),
            "latency_seconds": round(time.time() - start, 2),
        }

    def run(
        self,
        question: str,
        market_snapshot: Optional[Dict] = None,
    ) -> Dict:

        start = time.time()

        def _done(payload: Dict) -> Dict:
            if market_snapshot is not None:
                payload["market_snapshot"] = market_snapshot
            return payload

        analysis = analyze_query(
            question
        )

        requires_calculation = (
            analysis["requires_calculation"]
        )

        state = TAOState(
            max_reasoning_iterations=analysis[
                "max_reasoning_iterations"
            ],
            max_retrieval_expansions=analysis[
                "max_retrieval_expansions"
            ],
        )

        if (
            market_snapshot
            and market_snapshot.get("price") is not None
            and _wants_current_price(question)
        ):
            return _done(
                self._build_spot_price_answer(
                    market_snapshot,
                    analysis,
                    state,
                    start,
                )
            )

        # ==============================================================
        # DETERMINISTIC FINANCIAL METRIC PATH
        # ==============================================================

        computed_metric = None

        if requires_calculation:

            table_chunks = (
                self.rag_pipeline
                .retriever
                .get_table_chunks()
            )

            computed_metric = (
                metrics_resolver.resolve(
                    question,
                    analysis["years_mentioned"],
                    table_chunks,
                )
            )

        if computed_metric:

            # ----------------------------------------------------------
            # IMPORTANT:
            #
            # Do NOT call Ollama.
            #
            # The exact inputs are already present in the SEC filing
            # and the calculation is deterministic.
            # ----------------------------------------------------------

            deterministic = (
                self._build_deterministic_answer(
                    computed_metric
                )
            )

            return _done({
                **deterministic,
                "query_analysis": analysis,
                "tao": state.to_dict(),
                "latency_seconds": round(
                    time.time() - start,
                    2,
                ),
            })

        # ==============================================================
        # NORMAL RAG PATH
        # ==============================================================

        top_k = analysis["top_k"]

        results = self.rag_pipeline.retrieve(
            question,
            top_k=top_k,
        )

        if not results:

            return _done({
                "answer": "Insufficient evidence.",
                "evidence": [],
                "verification": None,
                "query_analysis": analysis,
                "tao": state.to_dict(),
                "latency_seconds": round(
                    time.time() - start,
                    2,
                ),
            })

        prompt = self.rag_pipeline.build_prompt(
            question,
            results,
            requires_calculation=requires_calculation,
            computed_metric=None,
        )

        raw_answer = (
            self.rag_pipeline
            .ollama_client
            .generate(prompt)
        )

        answer = (
            self.rag_pipeline
            ._strip_marker(raw_answer)
        )

        state.reasoning_iterations += 1

        verification = verify(
            answer,
            results,
            task_type=analysis["task_type"],
            requires_calculation=requires_calculation,
            computed_metric=None,
        )

        state.verification_calls += 1

        iterations = 1

        # ==============================================================
        # TAO LOOP
        # ==============================================================

        while iterations < MAX_TOTAL_ITERATIONS:

            action = decide(
                verification,
                state,
            )

            if action == Action.STOP:
                break

            if action == Action.RETRIEVE:

                top_k += 2

                results = (
                    self.rag_pipeline.retrieve(
                        question,
                        top_k=top_k,
                    )
                )

                state.retrieval_expansions += 1

                prompt = (
                    self.rag_pipeline
                    .build_prompt(
                        question,
                        results,
                        requires_calculation=(
                            requires_calculation
                        ),
                        computed_metric=None,
                    )
                )

                raw_answer = (
                    self.rag_pipeline
                    .ollama_client
                    .generate(prompt)
                )

                answer = (
                    self.rag_pipeline
                    ._strip_marker(
                        raw_answer
                    )
                )

                state.reasoning_iterations += 1

            elif action == Action.REVISE:

                prompt = (
                    self.rag_pipeline
                    .build_prompt(
                        question,
                        results,
                        revision_feedback={
                            "previous_answer": answer,
                            "failed_checks": (
                                verification.failed_checks
                            ),
                        },
                        requires_calculation=(
                            requires_calculation
                        ),
                        computed_metric=None,
                    )
                )

                raw_answer = (
                    self.rag_pipeline
                    .ollama_client
                    .generate(prompt)
                )

                answer = (
                    self.rag_pipeline
                    ._strip_marker(
                        raw_answer
                    )
                )

                state.reasoning_iterations += 1

            verification = verify(
                answer,
                results,
                task_type=analysis["task_type"],
                requires_calculation=requires_calculation,
                computed_metric=None,
            )

            state.verification_calls += 1

            iterations += 1

        # ==============================================================
        # YFINANCE LAST RESORT
        # ==============================================================

        if (
            self.allow_external_fallback
            and not computed_metric
            and requires_calculation
            and verification.confidence
            < YFINANCE_FALLBACK_CONFIDENCE_THRESHOLD
        ):

            metric = (
                metrics_resolver
                .detect_metric(question)
            )

            if metric:

                fallback = (
                    self._try_yfinance_fallback(
                        question,
                        metric,
                    )
                )

                if fallback:

                    return _done({
                        **fallback,
                        "query_analysis": analysis,
                        "tao": state.to_dict(),
                        "latency_seconds": round(
                            time.time() - start,
                            2,
                        ),
                    })

        # ==============================================================
        # NORMAL FINAL RESPONSE
        # ==============================================================

        display_answer = (
            self.rag_pipeline
            ._strip_evidence_section(
                answer
            )
        )

        return _done({
            "answer": display_answer,
            "evidence": [
                {
                    "score": r["score"],
                    "source": (
                        r["document"]
                        .get("metadata", {})
                        .get(
                            "source",
                            "Unknown",
                        )
                    ),
                    "filing": (
                        r["document"]
                        .get("metadata", {})
                        .get(
                            "filing",
                            "Unknown",
                        )
                    ),
                    # Carried through so evidence-level metrics
                    # (Hit@K) can tell which page was retrieved.
                    "page": (
                        r["document"]
                        .get("metadata", {})
                        .get("page")
                    ),
                    "document": (
                        r["document"]
                        .get("metadata", {})
                        .get("document")
                    ),
                    "text_preview": (
                        r["document"]
                        .get("text", "")
                    )[:300],
                }
                for r in results
            ],
            "verification": (
                verification.to_dict()
            ),
            "query_analysis": analysis,
            "tao": state.to_dict(),
            "latency_seconds": round(
                time.time() - start,
                2,
            ),
        })


def create_financebench_pipeline(
    rag_pipeline: RAGPipeline,
) -> TAOPipeline:
    """
    Create a TAO pipeline specifically for FinanceBench.

    External Yahoo Finance fallback is disabled.
    """

    return TAOPipeline(
        rag_pipeline=rag_pipeline,
        company_index={},
        allow_external_fallback=False,
    )