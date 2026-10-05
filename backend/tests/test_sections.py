
from pathlib import Path


sec_folder = Path("data/sec")

text_files = list(
    sec_folder.glob("*.txt")
)

if not text_files:
    raise FileNotFoundError(
        "No parsed SEC text file found."
    )


text_file = text_files[0]

text = text_file.read_text(
    encoding="utf-8"
)


keywords = [
    "CONSOLIDATED STATEMENTS OF OPERATIONS",
    "CONSOLIDATED BALANCE SHEETS",
    "CONSOLIDATED STATEMENTS OF CASH FLOWS",
    "NET SALES",
    "NET INCOME"
]


for keyword in keywords:

    print()
    print("=" * 80)
    print(f"SEARCHING FOR: {keyword}")
    print("=" * 80)

    position = text.upper().find(keyword)

    if position == -1:
        print("Not found.")
        continue

    start = max(0, position - 500)
    end = min(len(text), position + 3000)

    print(text[start:end])