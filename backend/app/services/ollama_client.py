# backend/app/services/ollama_client.py

import os

import requests
from dotenv import load_dotenv

load_dotenv()


class OllamaClient:
    """
    Thin client around a local Ollama server.

    Defaults are chosen for low-RAM / no-dedicated-GPU machines:
    - qwen2.5:1.5b-instruct is ~1GB quantized and runs fine on CPU.
    - num_ctx is sized to comfortably fit a 2-chunk RAG prompt
      (roughly 3.5-4.5k tokens for chunk_size=1200 words/chunk),
      which was the actual cause of the previous crashes: the
      context window (2048) was SMALLER than the prompt being sent,
      while the model (qwen2.5:7b) was simultaneously too large for
      the host's RAM/VRAM. Fix both independently via env vars.
    """

    BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Small model to fall back to if the primary model fails to load
    # (e.g. OOM). Keep this genuinely small.
    FALLBACK_MODEL = os.getenv("OLLAMA_FALLBACK_MODEL", "qwen2.5:1.5b-instruct")

    def __init__(
        self,
        model: str | None = None,
        timeout: int = 300,
        num_ctx: int | None = None,
        allow_fallback: bool = True
    ):
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b-instruct")
        self.timeout = timeout
        self.num_ctx = num_ctx or int(os.getenv("OLLAMA_NUM_CTX", "2048"))
        self.allow_fallback = allow_fallback

    def _call(self, model: str, prompt: str) -> requests.Response:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_ctx": self.num_ctx
            }
        }

        return requests.post(
            f"{self.BASE_URL}/api/generate",
            json=payload,
            timeout=self.timeout
        )

    def generate(
        self,
        prompt: str
    ) -> str:

        try:
            response = self._call(self.model, prompt)

            # A model that isn't pulled yet, or one that fails to load
            # (OOM / CUDA host allocation errors) comes back as a 4xx/5xx
            # with a message in response.text. Try the small fallback
            # model once before giving up.
            if not response.ok and self.allow_fallback and self.model != self.FALLBACK_MODEL:
                print()
                print("=" * 80)
                print("OLLAMA ERROR - retrying with fallback model")
                print("=" * 80)
                print("Primary model:", self.model)
                print("Status:", response.status_code)
                print("Response:", response.text[:500])
                print("=" * 80)

                response = self._call(self.FALLBACK_MODEL, prompt)

            if not response.ok:
                print()
                print("=" * 80)
                print("OLLAMA ERROR")
                print("=" * 80)
                print("Status:", response.status_code)
                print("Response:", response.text)
                print("=" * 80)

            response.raise_for_status()

            data = response.json()

            if "response" not in data:
                raise RuntimeError(
                    f"Ollama response did not contain "
                    f"'response': {data}"
                )

            return data["response"].strip()

        except requests.exceptions.Timeout:
            raise RuntimeError(
                "Ollama request timed out. The model may need more time, "
                "or the prompt/num_ctx may be too large for this machine. "
                "Try a smaller OLLAMA_MODEL or reduce top_k in retrieval."
            )

        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Could not connect to Ollama. Make sure `ollama serve` "
                "is running (check with `ollama list`)."
            )