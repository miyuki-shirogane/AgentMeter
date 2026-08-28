"""OpenAI-compatible judge provider.

Speaks plain HTTP to the standard ``/chat/completions`` endpoint, so it
works with OpenAI, Ollama, vLLM, LM Studio, and any other OpenAI-compatible
service by pointing ``base_url`` at it. Structured output is not assumed to
be supported; JSON is instead extracted tolerantly from the reply text.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from pydantic import ValidationError

from agentmeter.judge.base import JudgeError, JudgeProvider
from agentmeter.judge.result import JudgeResult


def _extract_json(text: str) -> dict[str, Any]:
    """Extract a JSON object from a judge reply.

    Handles bare JSON, JSON wrapped in markdown code fences, and JSON
    surrounded by prose. Raises :class:`JudgeError` when no object can be
    found or parsed.
    """
    candidate = text.strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    if candidate.startswith("```"):
        lines = candidate.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            pass

    raise JudgeError("could not extract valid JSON from judge response")


class OpenAIJudgeProvider(JudgeProvider):
    """Calls any OpenAI-compatible chat-completions endpoint as a judge.

    Args:
        model: model identifier, e.g. ``"gpt-4o-mini"`` or ``"llama3.1"``.
        base_url: API root; defaults to ``https://api.openai.com/v1``. For
            Ollama use e.g. ``http://localhost:11434/v1``.
        api_key: bearer token; defaults to the ``OPENAI_API_KEY`` env var.
        timeout: request timeout in seconds.
        temperature: sampling temperature; kept near 0 for stable judging.
        http_client: optional ``httpx.AsyncClient`` (useful in tests with
            ``httpx.MockTransport``). When omitted, a client is created and
            owned internally; close it with :meth:`aclose`.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        timeout: float = 30.0,
        temperature: float = 0.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        self._temperature = temperature
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(timeout=httpx.Timeout(timeout))

    async def judge(self, system_prompt: str, user_prompt: str) -> JudgeResult:
        payload = {
            "model": self._model,
            "temperature": self._temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

        try:
            response = await self._client.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            parsed = _extract_json(content)
            return JudgeResult.model_validate(parsed)
        except JudgeError:
            raise
        except httpx.TimeoutException as exc:
            raise JudgeError(f"judge API request timed out after {self._client.timeout}") from exc
        except httpx.HTTPStatusError as exc:
            raise JudgeError(f"judge API returned HTTP {exc.response.status_code}") from exc
        except httpx.HTTPError as exc:
            raise JudgeError(f"judge API request failed: {exc}") from exc
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise JudgeError(f"malformed judge API response: {exc}") from exc
        except ValidationError as exc:
            raise JudgeError(f"judge returned invalid structure: {exc}") from exc

    async def aclose(self) -> None:
        """Close the HTTP client when this provider created it."""
        if self._owns_client:
            await self._client.aclose()
