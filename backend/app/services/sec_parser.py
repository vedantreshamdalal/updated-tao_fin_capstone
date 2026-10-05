
from bs4 import BeautifulSoup


def parse_filing_html(html: str) -> str:
    """
    Convert an SEC Inline XBRL filing into readable text.
    """

    soup = BeautifulSoup(
        html,
        "lxml-xml"
    )

    # Remove XBRL metadata and non-readable elements.
    for element in soup.find_all([
        "header",
        "hidden",
        "script",
        "table",
        "style",
        "ix:header",
        "ix:hidden"
    ]):
        element.decompose()

    # Extract text while preserving logical separation.
    text = soup.get_text(
        separator="\n",
        strip=True
    )

    # Clean blank lines and whitespace.
    lines = []

    for line in text.splitlines():

        line = " ".join(line.split())

        if not line:
            continue

        lines.append(line)

    return "\n".join(lines)

