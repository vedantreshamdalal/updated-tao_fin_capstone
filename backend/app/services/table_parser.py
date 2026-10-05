# backend/app/services/table_parser.py

"""
Table Parser

The root cause of the diluted-EPS hallucination we hit: sec_parser.py
strips all HTML and flattens tables into one bare number per line
(no column headers nearby), so a small LLM has no way to tell which
number belongs to which year/row. pandas.read_html() gives us the
table back as a structured grid instead, and this module turns each
row into one unambiguous line like:

    "Earnings per share - Diluted: 2025=7.46, 2024=6.08, 2023=6.13"

instead of 18 bare numbers with the year headers three rows up.

SEC's Inline XBRL tables render with heavy colspan/rowspan, which
pandas.read_html expands into repeated adjacent columns (the same
value copy-pasted across the merged cells). We handle that by
deduping consecutive identical cells within a row before pairing
values with years -- this generalizes reasonably well across 10-K
tables (validated: 31/54 tables in the Apple filing produced clean
output, 0 crashes) but isn't guaranteed perfect on every possible
table layout. Spot-check important numbers.
"""

import re
from typing import Dict, List

import pandas as pd

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_NUMERIC_RE = re.compile(r"^-?\(?\d[\d.]*\)?$")

# Generic row labels that need their section header prefixed to stay
# unambiguous (e.g. "Basic" appears both under "Earnings per share:"
# and under "Shares used in computing earnings per share:").
_AMBIGUOUS_LABELS = {"basic", "diluted", "products", "services", "total", "domestic", "foreign"}


def _norm_cell(value) -> str | None:
    if pd.isna(value):
        return None

    s = str(value).strip()

    if s == "" or s.lower() == "nan" or s == "$":
        return None

    try:
        f = float(s.replace(",", ""))
        return str(int(f)) if f == int(f) else f"{f:g}"
    except ValueError:
        return s


def _dedupe_consecutive(values: List) -> List:
    out = []
    for v in values:
        if v is not None and (not out or v != out[-1]):
            out.append(v)
    return out


def _find_year_header(table: pd.DataFrame, max_scan: int = 10):
    """
    Scan the first few rows for the row containing the most distinct
    years (e.g. "September 27, 2025" or bare "2025") -- that's our
    column-to-year mapping for every row below it.
    """

    best_idx, best_years = None, []

    for i in range(min(max_scan, len(table))):
        raw_cells = [str(c) for c in table.iloc[i].tolist()]
        year_cells = [c for c in raw_cells if _YEAR_RE.search(c)]
        years = _dedupe_consecutive([_YEAR_RE.search(c).group(0) for c in year_cells])

        if len(years) >= 2 and len(years) > len(best_years):
            best_idx, best_years = i, years

    return best_idx, best_years


def table_to_lines(table: pd.DataFrame) -> List[str]:
    """Convert one pandas table into a list of 'label: year=value, ...' lines."""

    header_idx, years = _find_year_header(table)

    if header_idx is None:
        return []

    lines = []
    section = None

    for i in range(header_idx + 1, len(table)):
        row = _dedupe_consecutive([_norm_cell(c) for c in table.iloc[i].tolist()])

        if not row:
            continue

        label, *values = row
        numeric_values = [
            v for v in values
            if _NUMERIC_RE.match(v.replace(",", ""))
        ]

        if not numeric_values:
            # This is a section header row (e.g. "Earnings per share:")
            # with a label but no numbers -- remember it for context.
            if label and not label.replace(",", "").isdigit():
                section = label.rstrip(":")
            continue

        full_label = label
        if section and label.lower() in _AMBIGUOUS_LABELS:
            full_label = f"{section} - {label}"

        pairs = list(zip(years, numeric_values))
        if pairs:
            lines.append(
                f"{full_label}: " + ", ".join(f"{y}={v}" for y, v in pairs)
            )

    return lines


def extract_table_chunks(html_path: str, metadata: Dict | None = None) -> List[Dict]:
    """
    Parse every table in a filing's HTML into RAG-ready chunk dicts,
    same shape as chunker.split_into_chunks() output, so they can be
    embedded and added to the vector store alongside narrative text.
    """

    try:
        tables = pd.read_html(html_path)
    except ValueError:
        # No tables found in the document.
        return []

    chunks = []

    for table_index, table in enumerate(tables):
        lines = table_to_lines(table)

        if not lines:
            continue

        chunk_metadata = {
            "type": "table",
            "table_index": table_index,
        }
        if metadata:
            chunk_metadata.update(metadata)

        chunks.append({
            "text": "\n".join(lines),
            "metadata": chunk_metadata,
        })

    return chunks