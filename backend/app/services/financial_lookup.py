"""
Deterministic financial lookup.

Used for simple lookup questions where the requested value should
already exist verbatim in an SEC filing table.

Important distinction:

    "What was Apple's total net sales in fiscal year 2025?"

must prefer Apple's 2025 10-K over a 2026 10-Q containing a
year-to-date 2025 number.

For fiscal-year questions, 10-K is therefore strongly preferred.
For quarter / YTD questions, 10-Q can be selected instead.
"""

import re
from typing import Dict, List, Optional


_LINE_RE = re.compile(r"^([^:]+):\s*(.+)$")
_PAIR_RE = re.compile(r"(\d{4})=(-?\(?[\d,.]+\)?)")


def _parse_value(raw: str) -> float:
    raw = raw.strip()

    negative = raw.startswith("(") and raw.endswith(")")

    raw = raw.strip("()")
    raw = raw.replace(",", "")

    value = float(raw)

    return -value if negative else value


def _extract_year_values(text: str) -> Dict[int, float]:
    """
    Extract values from table-parser lines such as:

        Total net sales: 2025=416161, 2024=391035

    Returns:
        {2025: 416161.0, 2024: 391035.0}
    """

    values = {}

    for year, raw_value in _PAIR_RE.findall(text):
        try:
            values[int(year)] = _parse_value(raw_value)
        except ValueError:
            continue

    return values


def _normalise_label(label: str) -> str:
    """
    Normalize whitespace and case so that:

        "Total Net Sales"
        "total net sales"

    compare equally.
    """

    return " ".join(label.lower().split())


def _is_fiscal_year_question(question: str) -> bool:
    q = question.lower()

    fiscal_markers = [
        "fiscal year",
        "fiscal-year",
        "fy ",
        "fy202",
        "fy 202",
    ]

    return any(marker in q for marker in fiscal_markers)


def _requested_year(question: str) -> Optional[int]:
    """
    Extract the most likely requested fiscal/calendar year.
    """

    years = re.findall(r"\b(20\d{2})\b", question)

    if not years:
        return None

    return int(years[0])


def _company_matches(
    question: str,
    metadata: Dict,
    company_index: Optional[Dict[str, str]],
) -> bool:
    """
    If we know the company, make sure the table belongs to that company.

    If company_index is empty, don't reject anything.
    """

    if not company_index:
        return True

    q = question.lower()

    company = str(metadata.get("company", "")).lower()

    if company and company in q:
        return True

    for ticker, name in company_index.items():
        ticker_lower = str(ticker).lower()
        name_lower = str(name).lower()

        if ticker_lower in q.split():
            return company == ticker_lower

        if name_lower and name_lower in q:
            return company == ticker_lower

        # Also handle "Apple's", "Apple", etc.
        first_name = name_lower.split()[0] if name_lower else ""

        if first_name and first_name in q:
            return company == ticker_lower

    # If we can't confidently identify a company in the question,
    # don't filter the corpus based on company.
    return True


def _score_candidate(
    question: str,
    document: Dict,
    label: str,
    year: int,
    company_index: Optional[Dict[str, str]],
) -> Optional[int]:
    """
    Score an exact financial-table match.

    The most important rule:

        fiscal-year question + 10-K = very strong preference

    This prevents a 10-Q YTD number from beating the actual fiscal-year
    figure simply because the 10-Q happens to be newer.
    """

    metadata = document.get("metadata", {})

    normalized_label = _normalise_label(label)

    if normalized_label != "total net sales":
        return None

    text = document.get("text", "")
    values = _extract_year_values(text)

    if year not in values:
        return None

    if not _company_matches(question, metadata, company_index):
        return None

    score = 0

    form = str(metadata.get("form", "")).upper()
    filing = str(metadata.get("filing", "")).upper()

    fiscal_question = _is_fiscal_year_question(question)

    # Exact row label.
    score += 100

    # Requested year is present.
    score += 100

    # Fiscal-year questions MUST strongly prefer 10-K.
    if fiscal_question:
        if form == "10-K":
            score += 1000
        elif form == "10-Q":
            score -= 500

    # If filing name itself identifies a 10-K, give it extra preference.
    if "10-K" in filing:
        score += 200

    # For non-fiscal questions, don't automatically reject 10-Q.
    if not fiscal_question and form == "10-Q":
        score += 50

    # Prefer SEC source.
    if str(metadata.get("source", "")).upper() == "SEC EDGAR":
        score += 20

    return score


def lookup_total_net_sales(
    question: str,
    table_chunks: List[Dict],
    company_index: Optional[Dict[str, str]] = None,
) -> Optional[Dict]:
    """
    Deterministically find "Total net sales" for a requested year.

    Returns None if the requested value cannot be found.

    Example return:

        {
            "metric": "total net sales",
            "year": 2025,
            "value": 416161,
            "unit": "millions",
            "document": {...},
            "source": "SEC EDGAR",
            "filing": "...",
            "form": "10-K",
        }
    """

    year = _requested_year(question)

    if year is None:
        return None

    candidates = []

    for document in table_chunks:
        text = document.get("text", "")

        for line in text.splitlines():
            match = _LINE_RE.match(line)

            if not match:
                continue

            label = match.group(1).strip()
            values_text = match.group(2)

            score = _score_candidate(
                question=question,
                document=document,
                label=label,
                year=year,
                company_index=company_index,
            )

            if score is None:
                continue

            values = _extract_year_values(values_text)

            if year not in values:
                continue

            candidates.append(
                {
                    "score": score,
                    "value": values[year],
                    "year": year,
                    "document": document,
                    "label": label,
                }
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda candidate: candidate["score"],
        reverse=True,
    )

    best = candidates[0]
    document = best["document"]
    metadata = document.get("metadata", {})

    return {
        "metric": "total net sales",
        "year": best["year"],
        "value": best["value"],
        "unit": "millions",
        "document": document,
        "source": metadata.get("source", "Unknown"),
        "filing": metadata.get("filing", "Unknown"),
        "form": metadata.get("form", "Unknown"),
    }