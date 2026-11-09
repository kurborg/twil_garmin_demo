"""Small client for the local Ollama server. No requests run on import."""

import os

import numpy as np
import requests

URL = os.environ.get("TWIL_OLLAMA_URL", "http://localhost:11434").rstrip("/")
CHAT_MODEL = os.environ.get("TWIL_CHAT_MODEL", "twil-8b")
EMBED_MODEL = os.environ.get("TWIL_EMBED_MODEL", "nomic-embed-text")


def post(endpoint: str, payload: dict) -> dict:
    try:
        response = requests.post(f"{URL}{endpoint}", json=payload, timeout=(10, 600))
        response.raise_for_status()
    except requests.exceptions.SSLError as exc:
        raise RuntimeError(f"TLS validation failed while connecting to Ollama at {URL}.") from exc
    except requests.ConnectionError as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {URL}. Start Ollama, then run 'ollama list'."
        ) from exc
    except requests.Timeout as exc:
        raise RuntimeError("Ollama timed out. Try a shorter question or smaller context.") from exc
    except requests.HTTPError as exc:
        detail = response.text[:400]
        raise RuntimeError(
            f"Ollama returned HTTP {response.status_code}: {detail}. "
            "Check 'ollama list'; create twil-8b or pull nomic-embed-text if missing."
        ) from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"Cannot call Ollama at {URL}: {exc}") from exc
    try:
        result = response.json()
    except ValueError as exc:
        raise RuntimeError("Ollama returned invalid JSON. Check that the URL points to an Ollama server.") from exc
    if not isinstance(result, dict):
        raise RuntimeError("Ollama returned an unexpected response; expected a JSON object.")
    if "error" in result:
        raise RuntimeError(f"Ollama: {result['error']}")
    return result


def generate(prompt: str, model: str = CHAT_MODEL) -> str:
    result = post("/api/generate", {
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.2},
    })
    if not isinstance(result.get("response"), str):
        raise RuntimeError("Ollama returned no text response.")
    return result["response"]


def embed(texts: list[str], *, query: bool = False) -> np.ndarray:
    prefix = "search_query: " if query else "search_document: "
    inputs = [prefix + text for text in texts] if EMBED_MODEL.startswith("nomic") else texts
    result = post("/api/embed", {"model": EMBED_MODEL, "input": inputs, "truncate": False})
    try:
        vectors = np.asarray(result["embeddings"], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Ollama returned invalid embeddings.") from exc
    if vectors.ndim != 2 or len(vectors) != len(texts) or not np.isfinite(vectors).all():
        raise RuntimeError("Ollama returned invalid embeddings.")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if (norms == 0).any():
        raise RuntimeError("Ollama returned an empty embedding.")
    return vectors / norms
