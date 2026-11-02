from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from codenix.client import APIError, ChatClient
from codenix.config import Config


def make_config() -> Config:
    return Config(
        base_url="https://api.example.com/v1",
        api_key="test-key",
        model="test-model",
        timeout=5.0,
    )


def make_client(handler: Any) -> tuple[ChatClient, httpx.Client]:
    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    return ChatClient(make_config(), http_client=http_client), http_client


def test_chat_completion_sends_openai_compatible_request() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": "你好"}}
                ]
            },
        )

    client, http_client = make_client(handler)
    messages = [{"role": "user", "content": "hi"}]

    with http_client:
        data = client.chat_completion(messages)

    assert data["choices"][0]["message"]["content"] == "你好"

    request = captured["request"]
    assert str(request.url) == "https://api.example.com/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer test-key"
    assert request.headers["Content-Type"] == "application/json"
    assert json.loads(request.content) == {
        "model": "test-model",
        "messages": messages,
    }


def test_optional_parameters_are_forwarded() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}]},
        )

    tools = [{"type": "function", "function": {"name": "run_shell"}}]
    client, http_client = make_client(handler)

    with http_client:
        client.chat_completion(
            [{"role": "user", "content": "hi"}],
            tools=tools,
            tool_choice="auto",
            temperature=0.2,
            max_tokens=128,
            extra_body={"stream": False},
        )

    assert captured["body"]["tools"] == tools
    assert captured["body"]["tool_choice"] == "auto"
    assert captured["body"]["temperature"] == 0.2
    assert captured["body"]["max_tokens"] == 128
    assert captured["body"]["stream"] is False


def test_http_error_includes_status_and_api_message() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": {"message": "invalid api key"}},
        )

    client, http_client = make_client(handler)

    with http_client, pytest.raises(APIError) as excinfo:
        client.chat_completion([{"role": "user", "content": "hi"}])

    assert excinfo.value.status_code == 401
    assert "invalid api key" in str(excinfo.value)


def test_network_error_is_wrapped_as_api_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client, http_client = make_client(handler)

    with http_client, pytest.raises(APIError, match="connection refused"):
        client.chat_completion([{"role": "user", "content": "hi"}])


def test_malformed_response_is_rejected() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    client, http_client = make_client(handler)

    with http_client, pytest.raises(APIError, match="choices"):
        client.chat_completion([{"role": "user", "content": "hi"}])


def test_complete_returns_message_text() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "最终回答"}}]},
        )

    client, http_client = make_client(handler)

    with http_client:
        answer = client.complete([{"role": "user", "content": "hi"}])

    assert answer == "最终回答"
