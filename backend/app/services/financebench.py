import json
from pathlib import Path
from typing import Dict, List, Optional


class FinanceBench:

    def __init__(self, data_dir: Optional[str] = None):

        if data_dir is None:

            project_root = (
                Path(__file__)
                .resolve()
                .parents[3]
            )

            data_dir = (
                project_root
                / "data"
                / "financebench"
            )

        else:

            data_dir = Path(data_dir)

        self.data_dir = data_dir

        self.questions_file = (
            self.data_dir
            / "financebench_open_source.jsonl"
        )

        self.documents_file = (
            self.data_dir
            / "financebench_document_information.jsonl"
        )

        self.questions = []
        self.documents = {}

        self._load()

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------

    def _load(self):

        if not self.questions_file.exists():

            raise FileNotFoundError(
                f"FinanceBench questions file "
                f"not found: {self.questions_file}"
            )

        with open(
            self.questions_file,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                self.questions.append(
                    json.loads(line)
                )

        if self.documents_file.exists():

            with open(
                self.documents_file,
                "r",
                encoding="utf-8"
            ) as f:

                for line in f:

                    line = line.strip()

                    if not line:
                        continue

                    document = json.loads(line)

                    doc_name = document.get(
                        "doc_name"
                    )

                    if doc_name:
                        self.documents[
                            doc_name
                        ] = document

    # ---------------------------------------------------------
    # Dataset information
    # ---------------------------------------------------------

    def size(self) -> int:

        return len(
            self.questions
        )

    # ---------------------------------------------------------
    # Get all questions
    # ---------------------------------------------------------

    def get_questions(self):

        return self.questions

    # ---------------------------------------------------------
    # Get one question
    # ---------------------------------------------------------

    def get_question(
        self,
        financebench_id: str
    ) -> Optional[Dict]:

        for item in self.questions:

            if (
                str(
                    item.get(
                        "financebench_id"
                    )
                )
                ==
                str(financebench_id)
            ):

                return item

        return None

    # ---------------------------------------------------------
    # Get document metadata
    # ---------------------------------------------------------

    def get_document(
        self,
        doc_name: str
    ) -> Optional[Dict]:

        return self.documents.get(
            doc_name
        )

    # ---------------------------------------------------------
    # Join question + document metadata
    # ---------------------------------------------------------

    def get_enriched_questions(
        self
    ) -> List[Dict]:

        enriched = []

        for question in self.questions:

            item = dict(
                question
            )

            doc_name = question.get(
                "doc_name"
            )

            document = self.get_document(
                doc_name
            )

            if document:

                item.update({
                    "document_type":
                        document.get(
                            "doc_type"
                        ),

                    "document_period":
                        document.get(
                            "doc_period"
                        ),

                    "company":
                        document.get(
                            "company"
                        ),

                    "document_link":
                        document.get(
                            "doc_link"
                        ),

                    "sector":
                        document.get(
                            "comany_sector_gics"
                        )
                })

            enriched.append(item)

        return enriched

    # ---------------------------------------------------------
    # Filter by company
    # ---------------------------------------------------------

    def by_company(
        self,
        company: str
    ) -> List[Dict]:

        company_lower = (
            company.lower()
        )

        return [
            item
            for item in self.get_enriched_questions()
            if str(
                item.get("company", "")
            ).lower()
            ==
            company_lower
        ]

    # ---------------------------------------------------------
    # Filter by question type
    # ---------------------------------------------------------

    def by_question_type(
        self,
        question_type: str
    ) -> List[Dict]:

        return [
            item
            for item in self.questions
            if item.get(
                "question_type"
            )
            ==
            question_type
        ]

    # ---------------------------------------------------------
    # Gold answer
    # ---------------------------------------------------------

    def get_gold_answer(
        self,
        financebench_id: str
    ) -> Optional[str]:

        item = self.get_question(
            financebench_id
        )

        if not item:
            return None

        return item.get(
            "answer"
        )

    # ---------------------------------------------------------
    # Evidence
    # ---------------------------------------------------------

    def get_evidence(
        self,
        financebench_id: str
    ):

        item = self.get_question(
            financebench_id
        )

        if not item:
            return None

        return item.get(
            "evidence"
        )

    # ---------------------------------------------------------
    # Convert into evaluation format
    # ---------------------------------------------------------

    def evaluation_examples(self):

        examples = []

        for item in self.get_enriched_questions():

            examples.append({

                "id":
                    item.get(
                        "financebench_id"
                    ),

                "question":
                    item.get(
                        "question"
                    ),

                "gold_answer":
                    item.get(
                        "answer"
                    ),

                "justification":
                    item.get(
                        "justification"
                    ),

                "evidence":
                    item.get(
                        "evidence"
                    ),

                "company":
                    item.get(
                        "company"
                    ),

                "document":
                    item.get(
                        "doc_name"
                    ),

                "document_type":
                    item.get(
                        "document_type"
                    ),

                "document_period":
                    item.get(
                        "document_period"
                    )
            })

        return examples