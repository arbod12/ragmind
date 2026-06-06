"""
LLM client.

Notice this file defines a tiny ABSTRACT interface (LLMClient) and one
concrete implementation (GeminiClient). Why bother, when we only use Gemini?
Because the rest of the engine only ever talks to the abstract `LLMClient`.
If she later swaps to Claude or a local model, she writes one new class and
changes one line in the factory, and nothing in retriever/grader/generator
changes. This is the "model-agnostic" design decision, and being able to
articulate *why* you drew the boundary there is exactly the architectural
judgment that separates a junior who copies tutorials from one who designs.

Security note worth saying out loud in an interview: the API key is read
from the environment (os.environ), never written in code, never committed.
Same discipline as the website's serverless function.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import os
import json
import urllib.request
import urllib.error


class LLMClient(ABC):
    """The only surface the rest of the app depends on."""

    @abstractmethod
    def complete(self, system: str, user: str, temperature: float,
                 max_tokens: int) -> str:
        ...


class GeminiClient(LLMClient):
    """Calls Google's Generative Language API with no SDK, just stdlib HTTP.

    Using urllib instead of the google-genai package is a deliberate teaching
    choice: she can see exactly what HTTP request goes out, what JSON shape
    Gemini expects, and how the response is parsed. No magic. In production
    you might use the SDK for retries/streaming, but for understanding (and
    for a dependency-light deploy) raw HTTP is clearer and she can defend
    every byte of the request.
    """

    BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.key = os.getenv("GEMINI_API_KEY", "")

    def complete(self, system: str, user: str, temperature: float,
                 max_tokens: int) -> str:
        if not self.key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Export it in your shell or add it "
                "to Streamlit secrets before running."
            )

        url = f"{self.BASE}/{self.model_name}:generateContent?key={self.key}"
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="ignore")[:300]
            raise RuntimeError(f"Gemini API error {e.code}: {detail}")

        # Defensive parsing: the API can return a candidate with no text if it
        # was blocked or truncated. We fail loudly rather than silently
        # returning "" which would look like a hallucination downstream.
        candidates = body.get("candidates", [])
        if not candidates:
            raise RuntimeError(f"Gemini returned no candidates: {json.dumps(body)[:300]}")
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts).strip()
        if not text:
            raise RuntimeError("Gemini returned an empty completion.")
        return text


def make_llm(model_name: str) -> LLMClient:
    """Factory. The one place that knows which concrete client to build.
    To add Claude support later: write ClaudeClient(LLMClient) and branch
    here on model_name. Nothing else in the codebase changes."""
    return GeminiClient(model_name)
