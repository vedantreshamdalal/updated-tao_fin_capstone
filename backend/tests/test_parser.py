
from pathlib import Path

from app.services.sec_parser import parse_filing_html


# Find the downloaded 10-K
sec_folder = Path("data/sec")

html_files = list(
    sec_folder.glob("*.html")
)

if not html_files:
    raise FileNotFoundError(
        "No SEC HTML filing found in data/sec/"
    )


html_file = html_files[0]

print("Reading:")
print(html_file)


# Read HTML
html = html_file.read_text(
    encoding="utf-8"
)


# Parse HTML
text = parse_filing_html(html)


# Save clean text
output_file = (
    sec_folder
    / f"{html_file.stem}.txt"
)

output_file.write_text(
    text,
    encoding="utf-8"
)


print()
print("Clean text saved to:")
print(output_file)

print()
print("Characters:")
print(len(text))

print()
print("First 3000 characters:")
print(text[:3000])
