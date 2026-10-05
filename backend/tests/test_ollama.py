import requests


OLLAMA_URL = "http://localhost:11434/api/generate"

payload = {
    "model": "qwen2.5:7b",
    "prompt": (
        "confirm you are working locally"
    ),
    "stream": False
}


response = requests.post(
    OLLAMA_URL,
    json=payload,
    timeout=120
)

response.raise_for_status()

data = response.json()

print()
print("=" * 80)
print("OLLAMA RESPONSE")
print("=" * 80)

print(data["response"])