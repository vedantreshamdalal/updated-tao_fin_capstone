
from pathlib import Path

from app.services.sec_client import SECClient


sec = SECClient()

# Get Apple's 10-K filings
filings = sec.get_annual_reports("320193")

# Use the most recent 10-K
first_filing = filings[0]

print("Downloading:")
print(first_filing["filing_url"])

# Download the actual HTML filing
html = sec.download_filing(
    "320193",
    first_filing["accession_number"],
    first_filing["primary_document"]
)

print()
print("Downloaded characters:")
print(len(html))

# Create data/sec directory
output_dir = Path("data/sec")
output_dir.mkdir(
    parents=True,
    exist_ok=True
)

# Create output filename
output_file = (
    output_dir
    / f"apple_10k_{first_filing['filing_date']}.html"
)

# Save HTML
output_file.write_text(
    html,
    encoding="utf-8"
)

print()
print("Saved to:")
print(output_file)

print()
print("File size:")
print(output_file.stat().st_size, "bytes")

