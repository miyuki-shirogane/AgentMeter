"""Tests for OpenAIJudgeProvider using httpx.MockTransport (no network)."""

import json

import httpx
import pytest

from agentmeter import JudgeError, JudgeResult, OpenAIJudgeProvider


def _provider(handler) -> OpenAIJudgeProvider:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenAIJudgeProvider(model="test-model", api_key="test-key", http_client=client)


def _ok_response(content: str) -> httpx.Response:
    body = {"choices": [{"message": {"content": content}}]}
    return httpx.Response(200, json=body)


async def test_judge_returns_structured_result():
    async def handler(request):
        return _ok_response(json.dumps({"passed": True, "score": 0.92, "reason": "good"}))

    provider = _provider(handler)
    result = await provider.judge("system", "user")
    assert isinstance(result, JudgeResult)
    assert result.passed is True
    assert result.score == 0.92
    assert result.reason == "good"


async def test_judge_sends_correct_request():
    captured = {}

    def handler(request):
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        captured["auth"] = request.headers.get("Authorization")
        return _ok_response('{"passed": true, "score": 1.0}')

    provider = _provider(handler)
    await provider.judge("sys", "usr")

    assert captured["url"].endswith("/v1/chat/completions")
    assert captured["auth"] == "Bearer test-key"
    assert captured["body"]["model"] == "test-model"
    assert captured["body"]["messages"][0] == {"role": "system", "content": "sys"}
    assert captured["body"]["messages"][1] == {"role": "user", "content": "usr"}
    assert captured["body"]["temperature"] == 0.0


async def test_judge_handles_markdown_fenced_json():
    content = '```json\n{"passed": false, "score": 0.1}\n```'

    async def handler(request):
        return _ok_response(content)

    result = await _provider(handler).judge("s", "u")
    assert result.passed is False
    assert result.score == 0.1


async def test_judge_handles_json_surrounded_by_prose():
    content = 'I evaluated this. {"passed": true, "score": 0.8, "reason": "ok"} Done.'

    async def handler(request):
        return _ok_response(content)

    result = await _provider(handler).judge("s", "u")
    assert result.passed is True
    assert result.score == 0.8


async def test_judge_invalid_json_raises_judge_error():
    async def handler(request):
        return _ok_response("not json at all")

    with pytest.raises(JudgeError, match="could not extract valid JSON"):
        await _provider(handler).judge("s", "u")


async def test_judge_schema_violation_raises_judge_error():
    async def handler(request):
        return _ok_response('{"passed": true, "score": 5.0}')

    with pytest.raises(JudgeError, match="invalid structure"):
        await _provider(handler).judge("s", "u")


async def test_judge_missing_choices_raises_judge_error():
    async def handler(request):
        return httpx.Response(200, json={"error": "boom"})

    with pytest.raises(JudgeError, match="malformed"):
        await _provider(handler).judge("s", "u")


async def test_judge_http_error_raises_judge_error():
    async def handler(request):
        return httpx.Response(500, json={"error": "server error"})

    with pytest.raises(JudgeError, match="HTTP 500"):
        await _provider(handler).judge("s", "u")


async def test_judge_timeout_raises_judge_error():
    def handler(request):
        raise httpx.ConnectTimeout("connection timed out", request=request)

    with pytest.raises(JudgeError, match="timed out"):
        await _provider(handler).judge("s", "u")
