
from pathlib import Path

from app.services.chunker import split_into_chunks


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


chunks = split_into_chunks(
    text,
    chunk_size=1500,
    overlap=200
)


print("Total chunks:", len(chunks))

print()
print("=" * 80)
print("FIRST CHUNK")
print("=" * 80)

print(chunks[0]["text"])

print()
print("=" * 80)
print("SECOND CHUNK")
print("=" * 80)

print(chunks[1]["text"])
