"""The request-size middleware as a bare ASGI app. Its rejections render through
starlette, which comes with the serve extra, so these skip without it (httpx not needed)."""

import asyncio

import pytest

from augur.api import RequestSizeLimitMiddleware

pytest.importorskip("starlette", reason="serve extra not installed")


def _run_middleware(inner_app, headers, messages=(), scope_type="http"):
    sent = []
    pending = list(messages)

    async def receive():
        return pending.pop(0)

    async def send(message):
        sent.append(message)

    scope = {"type": scope_type, "headers": headers}
    asyncio.run(RequestSizeLimitMiddleware(inner_app)(scope, receive, send))
    return sent


@pytest.mark.parametrize("declared_length", [b"5000", b"9" * 5000],
                         ids=["just-over", "5000-digits"])
def test_size_limit_rejects_declared_oversize_before_reading(monkeypatch, declared_length):
    monkeypatch.setattr("augur.api.MAX_REQUEST_BYTES", 1000)

    async def inner_app(scope, receive, send):
        raise AssertionError("the app should never run")

    sent = _run_middleware(inner_app, [(b"content-length", declared_length)])
    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 413


@pytest.mark.parametrize("chunk_sizes, is_rejected", [([400, 400, 200], False),
                                                      ([400, 400, 201], True)])
def test_size_limit_counts_streamed_bytes(monkeypatch, chunk_sizes, is_rejected):
    monkeypatch.setattr("augur.api.MAX_REQUEST_BYTES", 1000)
    messages = [{"type": "http.request", "body": b"x" * size, "more_body": i < len(chunk_sizes) - 1}
                for i, size in enumerate(chunk_sizes)]

    async def draining_app(scope, receive, send):
        while (await receive())["more_body"]:
            pass

    sent = _run_middleware(draining_app, [], messages)
    assert [m["status"] for m in sent if m["type"] == "http.response.start"] == (
        [413] if is_rejected else [])


@pytest.mark.parametrize("scope_type", ["lifespan", "websocket"])
def test_size_limit_only_applies_to_http(monkeypatch, scope_type):
    monkeypatch.setattr("augur.api.MAX_REQUEST_BYTES", 1000)
    calls = []

    async def inner_app(scope, receive, send):
        calls.append(scope["type"])

    sent = _run_middleware(inner_app, [(b"content-length", b"5000")], scope_type=scope_type)
    assert calls == [scope_type]
    assert sent == []
