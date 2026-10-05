"""
Metrics Resolver

Deterministically resolves named financial ratios from indexed SEC
filing table chunks.

Important distinction:
    - 10-K = annual fiscal-year filing
    - 10-Q = quarterly / year-to-date filing

For questions asking for a "fiscal year" or "most recent fiscal year",
we MUST prefer completed 10-K filings and must NOT accidentally use a
2026 10-Q year-to-date number as if it were Apple's fiscal-year 2026.

For explicit fiscal years, e.g. "fiscal year 2025", we also prefer the
10-K for that year when available.

This avoids sending already-computable financial ratios through the LLM.
"""

import re
from typing import Dict, List, Optional, Tuple

from app.services.calculator import CalculatorError, margin


_PAIR_RE = re.compile(r"(\d{4})=(-?\(?[\d.]+\)?)")
_LINE_RE = re.compile(r"^([^:]+):\s*(.+)$")


# metric name -> (numerator label candidates, denominator label candidates)
METRIC_DEFINITIONS: Dict[str, Tuple[List[str], List[str]]] = {
    "operating margin": (
        ["operating income"],
        ["total net sales", "net sales"],
    ),
    "gross margin": (
        ["gross margin", "gross profit"],
        ["total net sales", "net sales"],
    ),
    "net margin": (
        ["net income"],
        ["total net sales", "net sales"],
    ),
    "profit margin": (
        ["net income"],
        ["total net sales", "net sales"],
    ),
}


_METRIC_KEYS_BY_LENGTH = sorted(
    METRIC_DEFINITIONS,
    key=len,
    reverse=True,
)


METRIC_TO_YFINANCE_KEY = {
    "operating margin": "operating_margin_pct",
    "gross margin": "gross_margin_pct",
    "net margin": "profit_margin_pct",
    "profit margin": "profit_margin_pct",
}


def _parse_value(raw: str) -> float:
    raw = raw.strip()

    negative = (
        raw.startswith("(")
        and raw.endswith(")")
    )

    raw = raw.strip("()")

    value = float(raw)

    return -value if negative else value


def detect_metric(question: str) -> Optional[str]:
    q = question.lower()

    for key in _METRIC_KEYS_BY_LENGTH:
        if key in q:
            return key

    return None


def _is_annual_filing(chunk: Dict) -> bool:
    """
    Return True when a chunk belongs to a 10-K annual filing.

    We inspect metadata["form"] first. As a fallback, we inspect the
    filename because some older indexed chunks may not have the form
    metadata populated.
    """

    metadata = chunk.get("metadata", {})

    form = str(metadata.get("form", "")).upper().strip()

    if form == "10-K":
        return True

    filing = str(
        metadata.get("filing", "")
    ).upper()

    return "10-K" in filing or "10K" in filing


def _is_quarterly_filing(chunk: Dict) -> bool:
    """
    Return True when a chunk belongs to a 10-Q filing.
    """

    metadata = chunk.get("metadata", {})

    form = str(metadata.get("form", "")).upper().strip()

    if form == "10-Q":
        return True

    filing = str(
        metadata.get("filing", "")
    ).upper()

    return "10-Q" in filing or "10Q" in filing


def _filing_key(chunk: Dict) -> str:
    """
    Identify the filing a table chunk came from.

    This prevents the resolver from accidentally combining the
    numerator from one filing with the denominator from another.
    """

    metadata = chunk.get("metadata", {})

    return str(
        metadata.get("filing", "")
    )


def _extract_line_items(
    label_candidates: List[str],
    table_chunks: List[Dict],
) -> List[Tuple[int, Dict[int, float], Dict]]:
    """
    Find all matching line items across all table chunks.

    Returns:
        [
            (match_score, year_to_value, source_chunk),
            ...
        ]

    Higher score means a better line-item match.
    """

    matches = []

    for chunk in table_chunks:
        text = chunk.get("text", "")

        for line in text.splitlines():

            match = _LINE_RE.match(line)

            if not match:
                continue

            label = match.group(1).strip().lower()

            if not any(
                candidate in label
                for candidate in label_candidates
            ):
                continue

            pairs = _PAIR_RE.findall(
                match.group(2)
            )

            if not pairs:
                continue

            values = {
                int(year): _parse_value(value)
                for year, value in pairs
            }

            if label in label_candidates:
                score = 3

            elif (
                "total" in label
                and " - " not in label
            ):
                score = 2

            elif " - " not in label:
                score = 1

            else:
                score = 0

            matches.append(
                (
                    score,
                    values,
                    chunk,
                )
            )

    return matches


def _best_match_for_filing(
    matches: List[Tuple[int, Dict[int, float], Dict]],
    filing_key: str,
) -> Optional[Tuple[int, Dict[int, float], Dict]]:
    """
    Select the best line-item match belonging to a specific filing.
    """

    candidates = [
        match
        for match in matches
        if _filing_key(match[2]) == filing_key
    ]

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return candidates[0]


def find_line_item(
    label_candidates: List[str],
    table_chunks: List[Dict],
    preferred_filings: Optional[set[str]] = None,
) -> Optional[Tuple[Dict[int, float], Dict]]:
    """
    Find the best line item.

    If preferred_filings is supplied, only chunks from those filings
    are considered.

    Otherwise, annual 10-K chunks are preferred over 10-Q chunks.
    """

    matches = _extract_line_items(
        label_candidates,
        table_chunks,
    )

    if not matches:
        return None

    # First preference: explicitly preferred filings.
    if preferred_filings:
        preferred_matches = [
            match
            for match in matches
            if _filing_key(match[2]) in preferred_filings
        ]

        if preferred_matches:
            preferred_matches.sort(
                key=lambda item: item[0],
                reverse=True,
            )

            _, values, chunk = preferred_matches[0]

            return values, chunk

    # Second preference: annual 10-K filings.
    annual_matches = [
        match
        for match in matches
        if _is_annual_filing(match[2])
    ]

    if annual_matches:
        annual_matches.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        _, values, chunk = annual_matches[0]

        return values, chunk

    # Last resort: use whatever is available.
    matches.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    _, values, chunk = matches[0]

    return values, chunk


def _find_best_matching_pair(
    numerator_matches: List[Tuple[int, Dict[int, float], Dict]],
    denominator_matches: List[Tuple[int, Dict[int, float], Dict]],
    requested_years: List[int],
) -> Optional[Dict]:
    """
    Find numerator + denominator belonging to the SAME filing.

    Annual 10-K pairs are strongly preferred.

    This is important because a corpus may contain:
        AAPL 2025 10-K
        AAPL 2026 10-Q

    We must never combine values from those different periods.
    """

    candidates = []

    numerator_by_filing = {}

    for score, values, chunk in numerator_matches:
        key = _filing_key(chunk)

        numerator_by_filing.setdefault(
            key,
            []
        ).append(
            (score, values, chunk)
        )

    denominator_by_filing = {}

    for score, values, chunk in denominator_matches:
        key = _filing_key(chunk)

        denominator_by_filing.setdefault(
            key,
            []
        ).append(
            (score, values, chunk)
        )

    common_filings = set(
        numerator_by_filing
    ) & set(
        denominator_by_filing
    )

    for filing in common_filings:

        for numerator in numerator_by_filing[filing]:
            for denominator in denominator_by_filing[filing]:

                numerator_score, numerator_values, numerator_chunk = numerator
                denominator_score, denominator_values, denominator_chunk = denominator

                common_years = sorted(
                    set(numerator_values)
                    & set(denominator_values),
                    reverse=True,
                )

                if not common_years:
                    continue

                # Prefer explicitly requested years.
                requested_common = [
                    year
                    for year in requested_years
                    if year in common_years
                ]

                if requested_common:
                    year = requested_common[0]
                    requested_bonus = 100
                else:
                    year = common_years[0]
                    requested_bonus = 0

                # Annual filing gets a very large preference.
                annual_bonus = (
                    1000
                    if _is_annual_filing(numerator_chunk)
                    else 0
                )

                # 10-Q gets no annual bonus.
                # This is intentionally NOT treated as a fiscal-year
                # result when an annual filing is available.
                score = (
                    annual_bonus
                    + requested_bonus
                    + numerator_score
                    + denominator_score
                )

                candidates.append(
                    {
                        "score": score,
                        "year": year,
                        "numerator_value": numerator_values[year],
                        "denominator_value": denominator_values[year],
                        "numerator_chunk": numerator_chunk,
                        "denominator_chunk": denominator_chunk,
                    }
                )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return candidates[0]


def resolve(
    question: str,
    years_mentioned: List[str],
    table_chunks: List[Dict],
) -> Optional[Dict]:
    """
    Resolve a named financial ratio deterministically.

    Rules:

    1. If a fiscal year is explicitly requested, prefer that year's
       annual 10-K.
    2. If "most recent fiscal year" is requested, prefer the most
       recent completed 10-K year.
    3. Never treat a 10-Q year-to-date value as a completed fiscal
       year when a 10-K is available.
    4. Numerator and denominator must come from the SAME filing.
    """

    metric = detect_metric(question)

    if metric is None:
        return None

    numerator_labels, denominator_labels = METRIC_DEFINITIONS[
        metric
    ]

    numerator_matches = _extract_line_items(
        numerator_labels,
        table_chunks,
    )

    denominator_matches = _extract_line_items(
        denominator_labels,
        table_chunks,
    )

    if not numerator_matches or not denominator_matches:
        return None

    requested_years = []

    for year in years_mentioned:
        try:
            requested_years.append(int(year))
        except (TypeError, ValueError):
            continue

    question_lower = question.lower()

    asks_most_recent_fiscal_year = (
        "most recent fiscal year" in question_lower
        or "latest fiscal year" in question_lower
        or "most recent fiscal" in question_lower
        or "latest fiscal" in question_lower
    )

    # ------------------------------------------------------------------
    # IMPORTANT:
    #
    # For a fiscal-year question, restrict candidates to annual 10-K
    # filings whenever we have them.
    # ------------------------------------------------------------------

    annual_numerator_matches = [
        match
        for match in numerator_matches
        if _is_annual_filing(match[2])
    ]

    annual_denominator_matches = [
        match
        for match in denominator_matches
        if _is_annual_filing(match[2])
    ]

    if annual_numerator_matches and annual_denominator_matches:
        numerator_matches = annual_numerator_matches
        denominator_matches = annual_denominator_matches

    # Find numerator + denominator from the same filing.
    pair = _find_best_matching_pair(
        numerator_matches,
        denominator_matches,
        requested_years,
    )

    if pair is None:
        return None

    year = pair["year"]

    # If this was specifically "most recent fiscal year", we only want
    # the most recent completed 10-K year. Because we filtered to annual
    # filings above, this will be 2025 rather than a 2026 10-Q.
    if asks_most_recent_fiscal_year:
        annual_years = []

        for _, values, chunk in numerator_matches:
            if not _is_annual_filing(chunk):
                continue

            for candidate_year in values:
                annual_years.append(candidate_year)

        for _, values, chunk in denominator_matches:
            if not _is_annual_filing(chunk):
                continue

            for candidate_year in values:
                annual_years.append(candidate_year)

        if annual_years:
            common_annual_years = []

            for candidate_year in set(annual_years):
                numerator_has_year = any(
                    candidate_year in values
                    for _, values, _ in numerator_matches
                )

                denominator_has_year = any(
                    candidate_year in values
                    for _, values, _ in denominator_matches
                )

                if numerator_has_year and denominator_has_year:
                    common_annual_years.append(
                        candidate_year
                    )

            if common_annual_years:
                year = max(common_annual_years)

                # Re-select the exact pair for that year.
                matching_pairs = []

                for (
                    numerator_score,
                    numerator_values,
                    numerator_chunk,
                ) in numerator_matches:

                    if year not in numerator_values:
                        continue

                    filing = _filing_key(
                        numerator_chunk
                    )

                    for (
                        denominator_score,
                        denominator_values,
                        denominator_chunk,
                    ) in denominator_matches:

                        if (
                            _filing_key(denominator_chunk)
                            != filing
                        ):
                            continue

                        if year not in denominator_values:
                            continue

                        matching_pairs.append(
                            (
                                numerator_score
                                + denominator_score,
                                numerator_values[year],
                                denominator_values[year],
                                numerator_chunk,
                                denominator_chunk,
                            )
                        )

                if matching_pairs:
                    matching_pairs.sort(
                        key=lambda x: x[0],
                        reverse=True,
                    )

                    (
                        _,
                        numerator_value,
                        denominator_value,
                        numerator_chunk,
                        denominator_chunk,
                    ) = matching_pairs[0]

                else:
                    return None

            else:
                return None

        else:
            return None

    else:
        numerator_value = pair["numerator_value"]
        denominator_value = pair["denominator_value"]
        numerator_chunk = pair["numerator_chunk"]
        denominator_chunk = pair["denominator_chunk"]

    try:
        computed = margin(
            numerator_value,
            denominator_value,
        )
    except CalculatorError:
        return None

    return {
        "metric": metric,
        "year": year,
        "numerator_label": numerator_labels[0],
        "numerator_value": numerator_value,
        "denominator_label": denominator_labels[0],
        "denominator_value": denominator_value,
        "computed_value_pct": round(
            computed,
            2,
        ),
        "source_chunks": [
            numerator_chunk,
            denominator_chunk,
        ],
        "is_most_recent_available": asks_most_recent_fiscal_year,
    }