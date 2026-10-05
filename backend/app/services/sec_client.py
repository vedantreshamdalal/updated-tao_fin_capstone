import os

import requests
from dotenv import load_dotenv

load_dotenv()


class SECClient:

    BASE_URL = "https://data.sec.gov"

    def __init__(self):
        self._session = None

    @property
    def session(self):
        # Built on first request rather than in __init__, so that a
        # missing SEC_USER_AGENT only fails the SEC endpoints instead
        # of preventing the whole app from importing -- routes/sec.py
        # constructs this client at module import time.
        if self._session is None:
            user_agent = os.getenv("SEC_USER_AGENT")

            if not user_agent:
                raise ValueError(
                    "SEC_USER_AGENT is not configured in .env"
                )

            self._session = requests.Session()

            self._session.headers.update({
                "User-Agent": user_agent,
                "Accept-Encoding": "gzip, deflate",
            })

        return self._session

    def get_company_submissions(self, cik: str):
        cik = str(cik).zfill(10)

        url = (
            f"{self.BASE_URL}/submissions/"
            f"CIK{cik}.json"
        )

        response = self.session.get(url)
        response.raise_for_status()

        return response.json()

    def get_annual_reports(self, cik: str):
        data = self.get_company_submissions(cik)

        recent = data["filings"]["recent"]

        filings = []

        for i in range(len(recent["form"])):

            if recent["form"][i] != "10-K":
                continue

            accession = recent["accessionNumber"][i]
            document = recent["primaryDocument"][i]

            filings.append({
                "form": recent["form"][i],
                "filing_date": recent["filingDate"][i],
                "report_date": recent["reportDate"][i],
                "accession_number": accession,
                "primary_document": document,
                "filing_url": self.build_filing_url(
                    cik,
                    accession,
                    document
                )
            })

        return filings

    def build_filing_url(
        self,
        cik: str,
        accession_number: str,
        primary_document: str
    ):
        cik_number = str(int(cik))

        accession_clean = accession_number.replace(
            "-",
            ""
        )

        return (
            "https://www.sec.gov/Archives/edgar/data/"
            f"{cik_number}/"
            f"{accession_clean}/"
            f"{primary_document}"
        )

    def download_filing(
        self,
        cik: str,
        accession_number: str,
        primary_document: str
    ):
        url = self.build_filing_url(
            cik,
            accession_number,
            primary_document
        )

        response = self.session.get(url)
        response.raise_for_status()

        return response.text