# backend/app/services/query_analyzer.py

"""
Query Analyzer

Cheap, regex/keyword-based classification of the incoming question.
No LLM call here on purpose -- this runs before any expensive work
and its whole job is to set sensible starting budgets for the TAO
controller (how much top_k, how many reasoning/verification rounds
are worth spending on THIS question).

Difficulty levels:
    simple   - single fact lookup ("what was X's revenue in 2025")
    moderate - one calculation or a two-item comparison
    complex  - multi-step reasoning, trend/"why" explanation,
               multi-year or multi-metric comparison

Task types:
    lookup       - direct fact retrieval
    calculation  - requires arithmetic on retrieved numbers
    comparison   - compares across years/entities/metrics
    explanation  - "why/how" -- needs reasoning across evidence
"""

import re
from typing import Dict


_CALCULATION_WORDS = [
    "margin", "ratio", "growth rate", "growth", "yoy", "y/y",
    "cagr", "percentage", "percent", "%", "increase", "decrease",
    "change in", "difference between"
]

_COMPARISON_WORDS = [
    " vs ", " versus ", "compare", "compared to", "relative to",
    "higher than", "lower than", "between"
]

_EXPLANATION_WORDS = [
    "why", "how did", "what caused", "what drove", "explain",
    "reason for", "due to"
]

_YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")


def analyze_query(question: str) -> Dict:
    q = question.lower()

    # _YEAR_PATTERN has a capturing group (for validation only), so
    # pull the full 4-digit year matches with a separate non-capturing regex.
    year_matches = re.findall(r"\b(?:19|20)\d{2}\b", question)

    has_calculation = any(w in q for w in _CALCULATION_WORDS)
    has_comparison = any(w in q for w in _COMPARISON_WORDS) or len(set(year_matches)) >= 2
    has_explanation = any(w in q for w in _EXPLANATION_WORDS)

    if has_explanation or (has_comparison and has_calculation):
        difficulty = "complex"
    elif has_calculation or has_comparison:
        difficulty = "moderate"
    else:
        difficulty = "simple"

    if has_explanation:
        task_type = "explanation"
    elif has_comparison:
        task_type = "comparison"
    elif has_calculation:
        task_type = "calculation"
    else:
        task_type = "lookup"

    budgets = {
        "simple":   {"top_k": 3, "max_reasoning_iterations": 1, "max_retrieval_expansions": 1},
        "moderate": {"top_k": 4, "max_reasoning_iterations": 2, "max_retrieval_expansions": 1},
        "complex":  {"top_k": 5, "max_reasoning_iterations": 3, "max_retrieval_expansions": 2},
    }[difficulty]

    return {
        "difficulty": difficulty,
        "task_type": task_type,
        "requires_calculation": has_calculation,
        "years_mentioned": sorted(set(year_matches)),
        **budgets,
    }