"""OpenAI 兼容 Chat Completions API 的最小 HTTP 客户端。

这里刻意不使用 OpenAI SDK：请求体、请求头和响应结构都直接由本项目控制，
便于理解 Tool Calling 和后续 Agent Loop 的底层行为。
"""

from __future__ import annotations

import json
from typing import Any, Self

import httpx

from .config import Config


class APIError(RuntimeError):
    """模型 API 调用失败。"""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response_body: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class ChatClient:
    """封装一次 Chat Completions 请求。

    传入 ``http_client`` 时复用调用方提供的 ``httpx.Client``（测试中常用），
    否则由本对象自行创建并负责关闭。
    """

    def __init__(
        self,
        config: Config,
        *,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.config = config
        self._client = http_client or httpx.Client(timeout=config.timeout)
        self._owns_client = http_client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """发送消息并返回完整的响应 JSON。

        可选参数只在显式提供时才写入请求体，避免把 ``None`` 发给 API。
        """

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
        }
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if extra_body:
            payload.update(extra_body)

        url = self.config.chat_completions_url
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = self._client.post(url, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            raise APIError(f"请求 {url} 失败：{exc}") from exc

        if response.status_code >= 400:
            raise APIError(
                _error_message(response),
                status_code=response.status_code,
                response_body=response.text,
            )

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise APIError(
                "API 返回的内容不是合法 JSON",
                status_code=response.status_code,
                response_body=response.text,
            ) from exc

        _validate_response(data)
        return data

    def complete(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> str:
        """便捷方法：只返回第一条回复的文本内容。"""

        data = self.chat_completion(messages, **kwargs)
        content = data["choices"][0]["message"].get("content")
        return content or ""


def _error_message(response: httpx.Response) -> str:
    detail = ""
    try:
        body = response.json()
        error = body.get("error") if isinstance(body, dict) else None
        if isinstance(error, dict):
            detail = str(error.get("message", ""))
        elif error:
            detail = str(error)
        if not detail and isinstance(body, dict):
            detail = str(body.get("message", ""))
    except (json.JSONDecodeError, ValueError):
        detail = response.text[:500]

    detail = detail.strip() or "无错误信息"
    return f"API 请求失败（HTTP {response.status_code}）：{detail}"


def _validate_response(data: Any) -> None:
    if not isinstance(data, dict):
        raise APIError("API 响应格式错误：顶层应为 JSON 对象")

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise APIError("API 响应格式错误：choices 为空")

    first = choices[0]
    if not isinstance(first, dict) or not isinstance(first.get("message"), dict):
        raise APIError("API 响应格式错误：choices[0].message 缺失")
