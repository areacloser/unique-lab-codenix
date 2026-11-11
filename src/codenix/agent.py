"""Agent 核心层：对话、Tool Calling 循环与工具结果回传。"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from .history import ConversationHistory


class AgentError(RuntimeError):
    """Agent 无法继续执行或模型响应不符合预期。"""


class ChatCompletionClient(Protocol):
    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ...


class ToolExecutor(Protocol):
    def schemas(self) -> list[dict[str, Any]]:
        ...

    def execute(self, name: str, arguments: Mapping[str, Any]) -> str:
        ...


ToolCallCallback = Callable[[str, Mapping[str, Any]], None]
ToolResultCallback = Callable[[str, str], None]


class Agent:
    """管理一段对话，并在需要时执行模型请求的工具。"""

    def __init__(
        self,
        client: ChatCompletionClient,
        history: ConversationHistory | None = None,
        *,
        tool_registry: ToolExecutor | None = None,
        max_tool_rounds: int = 10,
        on_tool_call: ToolCallCallback | None = None,
        on_tool_result: ToolResultCallback | None = None,
    ) -> None:
        if max_tool_rounds < 1:
            raise ValueError("max_tool_rounds 必须大于 0")

        self.client: ChatCompletionClient = client
        self.history: ConversationHistory = (
            history if history is not None else ConversationHistory()
        )
        self.tool_registry: ToolExecutor | None = tool_registry
        self.max_tool_rounds = max_tool_rounds
        self.on_tool_call = on_tool_call
        self.on_tool_result = on_tool_result

    def send(self, user_input: str) -> str:
        text = user_input.strip()
        if not text:
            raise ValueError("用户输入不能为空")

        tools = self.tool_registry.schemas() if self.tool_registry else None
        tool_choice = "auto" if tools else None

        # 首次请求成功后才提交 user 消息，避免用户重试时产生重复上下文。
        messages = self.history.messages()
        messages.append({"role": "user", "content": text})
        message = self._request(messages, tools, tool_choice)
        self.history.add_user(text)
        self.history.add_assistant(message)

        rounds = 0
        while _has_tool_calls(message):
            if rounds >= self.max_tool_rounds:
                raise AgentError(
                    f"已达到最大工具调用轮数（{self.max_tool_rounds} 轮）"
                )
            rounds += 1
            self._execute_tool_calls(message)
            message = self._request(self.history.messages(), tools, tool_choice)
            self.history.add_assistant(message)

        content = message.get("content")
        return content if isinstance(content, str) else ""

    def _request(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | None,
    ) -> dict[str, Any]:
        if tools:
            data = self.client.chat_completion(
                messages,
                tools=tools,
                tool_choice=tool_choice,
            )
        else:
            data = self.client.chat_completion(messages)
        return _extract_message(data)

    def _execute_tool_calls(self, message: Mapping[str, Any]) -> None:
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            raise AgentError("assistant 消息中的 tool_calls 格式错误")

        for call in tool_calls:
            if not isinstance(call, dict):
                raise AgentError("tool_calls 中的元素必须是对象")

            call_id = call.get("id")
            if not isinstance(call_id, str) or not call_id:
                raise AgentError("tool_call 缺少 id，无法回传执行结果")

            function = call.get("function")
            if not isinstance(function, dict):
                self._record_tool_result(
                    call_id, "<unknown>", "错误：工具调用缺少 function 对象"
                )
                continue

            name = function.get("name")
            if not isinstance(name, str) or not name:
                self._record_tool_result(
                    call_id, "<unknown>", "错误：工具调用缺少 function.name"
                )
                continue

            arguments, parse_error = _parse_arguments(
                function.get("arguments", "{}")
            )
            if parse_error is not None:
                self._record_tool_result(
                    call_id, name, f"错误：{parse_error}"
                )
                continue
            if arguments is None:
                self._record_tool_result(
                    call_id, name, "错误：无法解析工具参数"
                )
                continue

            if self.on_tool_call is not None:
                self.on_tool_call(name, arguments)

            if self.tool_registry is None:
                result = "错误：Agent 未配置工具运行时"
            else:
                try:
                    result = self.tool_registry.execute(name, arguments)
                except BaseException as exc:
                    # 即使被 Ctrl-C 打断，也要补齐 tool 消息，保持历史合法。
                    self.history.add_tool(
                        call_id,
                        f"工具执行被中断：{type(exc).__name__}: {exc}",
                    )
                    raise

            self._record_tool_result(call_id, name, result)

    def _record_tool_result(
        self,
        call_id: str,
        name: str,
        result: str,
    ) -> None:
        self.history.add_tool(call_id, result)
        if self.on_tool_result is not None:
            self.on_tool_result(name, result)


def _extract_message(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise AgentError("API 响应格式错误：顶层应为 JSON 对象")

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise AgentError("API 响应格式错误：choices 为空")

    first = choices[0]
    if not isinstance(first, dict):
        raise AgentError("API 响应格式错误：choices[0] 不是对象")

    message = first.get("message")
    if not isinstance(message, dict):
        raise AgentError("API 响应格式错误：choices[0].message 缺失")
    return message


def _has_tool_calls(message: Mapping[str, Any]) -> bool:
    calls = message.get("tool_calls")
    if calls is None:
        return False
    if not isinstance(calls, list):
        raise AgentError("assistant 消息中的 tool_calls 格式错误")
    return bool(calls)


def _parse_arguments(
    raw: Any,
) -> tuple[dict[str, Any], None] | tuple[None, str]:
    if raw is None:
        raw = "{}"

    if isinstance(raw, Mapping):
        return dict(raw), None
    if not isinstance(raw, str):
        return None, "工具参数必须是 JSON 字符串或对象"

    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"工具参数不是合法 JSON：{exc}"

    if not isinstance(value, dict):
        return None, "工具参数必须是 JSON 对象"
    return value, None
