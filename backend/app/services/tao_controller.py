# backend/app/services/tao_controller.py

"""
TAO Controller

Given the current verification result and the iteration budget the
Query Analyzer set for this question, decide the next action:

    STOP     - confidence is high enough, or budget is exhausted
    REVISE   - evidence is fine but the reasoning/answer itself failed
               a check (numerical / logical / financial) -> re-prompt
               the same LLM call with the specific failure named
    RETRIEVE - evidence itself looks insufficient -> widen top_k and
               re-run retrieval before reasoning again

Also computes the "compute saved vs a fixed budget" metric shown in
the UI: we compare actual (reasoning + retrieval + verification)
calls made against a FIXED_BUDGET that a naive non-adaptive system
would always spend on every question regardless of difficulty.
"""

from enum import Enum
from typing import Dict


# What a naive, non-adaptive pipeline would always do per question,
# regardless of how easy or hard it is. Used only for the "compute
# saved" comparison metric.
FIXED_BUDGET_CALLS = 6  # e.g. always: 3 reasoning passes + 2 retrievals + 1 verify

STOP_CONFIDENCE_THRESHOLD = 0.90


class Action(str, Enum):
    STOP = "stop"
    REVISE = "revise"
    RETRIEVE = "retrieve"


class TAOState:
    def __init__(self, max_reasoning_iterations: int, max_retrieval_expansions: int):
        self.max_reasoning_iterations = max_reasoning_iterations
        self.max_retrieval_expansions = max_retrieval_expansions

        self.reasoning_iterations = 0
        self.retrieval_expansions = 0
        self.verification_calls = 0

    def total_calls(self) -> int:
        return (
            self.reasoning_iterations
            + self.retrieval_expansions
            + self.verification_calls
        )

    def compute_saved_pct(self) -> float:
        saved = 1 - (self.total_calls() / FIXED_BUDGET_CALLS)
        return round(max(saved, 0.0) * 100, 1)

    def to_dict(self) -> Dict:
        return {
            "reasoning_iterations": self.reasoning_iterations,
            "retrieval_expansions": self.retrieval_expansions,
            "verification_calls": self.verification_calls,
            "compute_saved_pct": self.compute_saved_pct(),
        }


def decide(
    verification,  # VerificationResult
    state: TAOState,
) -> Action:

    if verification.confidence >= STOP_CONFIDENCE_THRESHOLD:
        return Action.STOP

    budget_exhausted = (
        state.reasoning_iterations >= state.max_reasoning_iterations
        and state.retrieval_expansions >= state.max_retrieval_expansions
    )

    if budget_exhausted:
        return Action.STOP

    # Evidence itself looks insufficient -> widen retrieval first,
    # since revising reasoning over the same weak evidence won't help.
    if "evidence" in verification.failed_checks and state.retrieval_expansions < state.max_retrieval_expansions:
        return Action.RETRIEVE

    # Otherwise, if we still have reasoning budget, ask the model to
    # revise (numerical / logical / financial_consistency failures).
    if state.reasoning_iterations < state.max_reasoning_iterations:
        return Action.REVISE

    # Reasoning budget spent but retrieval budget remains -- try
    # broader evidence as a last resort.
    if state.retrieval_expansions < state.max_retrieval_expansions:
        return Action.RETRIEVE

    return Action.STOP