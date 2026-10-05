import re
from typing import Any, Dict, List


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_text(text: Any) -> str:

    if text is None:
        return ""

    text = str(text).lower()

    text = text.replace(
        "$",
        ""
    )

    text = text.replace(
        ",",
        ""
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# NUMBER EXTRACTION
# =========================================================

def extract_numbers(
    text: Any
) -> List[float]:

    if text is None:
        return []

    text = str(text)

    matches = re.findall(
        r"-?\d+(?:,\d{3})*(?:\.\d+)?",
        text
    )

    numbers = []

    for value in matches:

        try:

            numbers.append(
                float(
                    value.replace(
                        ",",
                        ""
                    )
                )
            )

        except ValueError:
            pass

    return numbers


# =========================================================
# NUMERICAL MATCH
# =========================================================

def numerical_match(
    prediction: Any,
    gold: Any,
    tolerance: float = 0.01
) -> bool:

    predicted_numbers = extract_numbers(
        prediction
    )

    gold_numbers = extract_numbers(
        gold
    )

    if not predicted_numbers:
        return False

    if not gold_numbers:
        return False

    for gold_number in gold_numbers:

        found = False

        for predicted_number in predicted_numbers:

            difference = abs(
                predicted_number
                - gold_number
            )

            denominator = max(
                abs(gold_number),
                1.0
            )

            relative_error = (
                difference
                / denominator
            )

            if relative_error <= tolerance:

                found = True
                break

        if not found:
            return False

    return True


# =========================================================
# ANSWER MATCHING
# =========================================================

def answer_matches(
    prediction: Any,
    gold: Any
) -> bool:

    prediction_normalized = (
        normalize_text(prediction)
    )

    gold_normalized = (
        normalize_text(gold)
    )

    if not prediction_normalized:
        return False

    if not gold_normalized:
        return False

    # Exact match
    if (
        prediction_normalized
        == gold_normalized
    ):
        return True

    # Gold answer appears in model answer
    if gold_normalized in prediction_normalized:
        return True

    # Numerical match
    if numerical_match(
        prediction,
        gold
    ):
        return True

    return False


# =========================================================
# PAGE EXTRACTION
# =========================================================

def extract_page(
    evidence: Any
):

    if not isinstance(
        evidence,
        dict
    ):
        return None

    for key in (
        "page",
        "page_number",
        "page_num",
        "evidence_page_num",
    ):

        value = evidence.get(
            key
        )

        if value is not None:

            try:
                return int(value)

            except (
                TypeError,
                ValueError
            ):
                return None

    metadata = evidence.get(
        "metadata"
    )

    if isinstance(
        metadata,
        dict
    ):

        for key in (
            "page",
            "page_number",
            "page_num",
            "evidence_page_num",
        ):

            value = metadata.get(
                key
            )

            if value is not None:

                try:
                    return int(value)

                except (
                    TypeError,
                    ValueError
                ):
                    return None

    return None


# =========================================================
# GOLD EVIDENCE PAGES
# =========================================================

def get_gold_pages(
    benchmark_item: Dict
) -> List[int]:

    evidence = (
        benchmark_item.get(
            "evidence"
        )
        or []
    )

    if not isinstance(
        evidence,
        list
    ):
        evidence = [
            evidence
        ]

    pages = []

    for item in evidence:

        page = extract_page(
            item
        )

        if page is not None:

            pages.append(
                page
            )

    return sorted(
        set(pages)
    )


# =========================================================
# RETRIEVED PAGES
# =========================================================

def get_retrieved_pages(
    retrieved_evidence: List[Dict]
) -> List[int]:

    pages = []

    for evidence in retrieved_evidence:

        page = extract_page(
            evidence
        )

        if page is not None:

            pages.append(
                page
            )

    return pages


# =========================================================
# HIT@K
# =========================================================

def hit_at_k(
    retrieved_evidence: List[Dict],
    gold_pages: List[int],
    k: int
) -> bool:

    if not gold_pages:
        return False

    retrieved_pages = (
        get_retrieved_pages(
            retrieved_evidence[:k]
        )
    )

    return any(
        page in gold_pages
        for page in retrieved_pages
    )


# =========================================================
# EVALUATE ONE QUESTION
# =========================================================

def evaluate_question(
    benchmark_item: Dict,
    model_result: Dict
) -> Dict:

    gold_answer = (
        benchmark_item.get(
            "answer",
            ""
        )
    )

    prediction = (
        model_result.get(
            "answer",
            ""
        )
    )

    retrieved_evidence = (
        model_result.get(
            "evidence",
            []
        )
        or []
    )

    gold_pages = get_gold_pages(
        benchmark_item
    )

    return {

        "financebench_id":
            benchmark_item.get(
                "financebench_id"
            ),

        "question":
            benchmark_item.get(
                "question"
            ),

        "company":
            benchmark_item.get(
                "company"
            ),

        "document":
            benchmark_item.get(
                "doc_name"
            ),

        "question_type":
            benchmark_item.get(
                "question_type"
            ),

        "question_reasoning":
            benchmark_item.get(
                "question_reasoning"
            ),

        "gold_answer":
            gold_answer,

        "predicted_answer":
            prediction,

        "answer_correct":
            answer_matches(
                prediction,
                gold_answer
            ),

        "gold_pages":
            gold_pages,

        "hit_at_1":
            hit_at_k(
                retrieved_evidence,
                gold_pages,
                1
            ),

        "hit_at_3":
            hit_at_k(
                retrieved_evidence,
                gold_pages,
                3
            ),

        "hit_at_5":
            hit_at_k(
                retrieved_evidence,
                gold_pages,
                5
            ),

        "latency_seconds":
            model_result.get(
                "latency_seconds"
            ),

        "verification":
            model_result.get(
                "verification"
            ),

        "retrieved_evidence":
            retrieved_evidence,

        "error":
            model_result.get(
                "error"
            )
    }


# =========================================================
# SUMMARY
# =========================================================

def calculate_summary(
    results: List[Dict]
) -> Dict:

    total = len(
        results
    )

    if total == 0:

        return {
            "num_questions": 0
        }

    correct = sum(
        bool(
            result[
                "answer_correct"
            ]
        )
        for result in results
    )

    hit1 = sum(
        bool(
            result[
                "hit_at_1"
            ]
        )
        for result in results
    )

    hit3 = sum(
        bool(
            result[
                "hit_at_3"
            ]
        )
        for result in results
    )

    hit5 = sum(
        bool(
            result[
                "hit_at_5"
            ]
        )
        for result in results
    )

    latencies = [
        result[
            "latency_seconds"
        ]
        for result in results
        if isinstance(
            result[
                "latency_seconds"
            ],
            (int, float)
        )
    ]

    summary = {

        "num_questions":
            total,

        "answer_accuracy":
            correct / total,

        "evidence_hit_at_1":
            hit1 / total,

        "evidence_hit_at_3":
            hit3 / total,

        "evidence_hit_at_5":
            hit5 / total,
    }

    if latencies:

        summary[
            "average_latency_seconds"
        ] = sum(
            latencies
        ) / len(
            latencies
        )

    # -----------------------------------------------------
    # Accuracy by question type
    # -----------------------------------------------------

    by_type = {}

    for result in results:

        question_type = (
            result.get(
                "question_type"
            )
            or "unknown"
        )

        by_type.setdefault(
            question_type,
            []
        )

        by_type[
            question_type
        ].append(
            result
        )

    summary[
        "accuracy_by_question_type"
    ] = {}

    for question_type, items in by_type.items():

        summary[
            "accuracy_by_question_type"
        ][
            question_type
        ] = {

            "count":
                len(items),

            "accuracy":
                sum(
                    x[
                        "answer_correct"
                    ]
                    for x in items
                )
                / len(items),

            "hit_at_5":
                sum(
                    x[
                        "hit_at_5"
                    ]
                    for x in items
                )
                / len(items)
        }

    return summary