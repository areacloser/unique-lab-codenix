"""对话历史管理。

当前阶段维护内存中的 ``system`` / ``user`` / ``assistant`` / ``tool`` 消息。
持久化到 ``sessions/`` 会在 Hard 阶段实现。
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any


class ConversationHistory:
    """管理一段连续对话的消息历史。

    ``system`` prompt 与普通消息分开保存，因此 :meth:`clear` 清空上下文时
    不会丢掉助手的人设和规则。
    """

    def __init__(self, system_prompt: str | None = None) -> None:
        self._system_prompt: str = (system_prompt or "").strip()
        self._messages: list[dict[str, Any]] = []

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    def messages(self) -> list[dict[str, Any]]:
        """返回可直接发送给 API 的消息列表。

        返回值是深拷贝，调用方修改它不会影响内部历史。
        """

        messages: list[dict[str, Any]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.extend(deepcopy(self._messages))
        return messages

    def add_user(self, content: str) -> None:
        if not content.strip():
            raise ValueError("用户消息不能为空")
        self._messages.append({"role": "user", "content": content})

    def add_assistant(self, message: Mapping[str, Any]) -> None:
        """追加完整的 assistant 消息。

        保留整个字典而不只保存 ``content``，这样后续 Tool Calling 阶段
        才能把 ``tool_calls`` 原样回传给模型。
        """

        data = deepcopy(dict(message))
        data["role"] = "assistant"
        self._messages.append(data)

    def add_tool(self, tool_call_id: str, content: str) -> None:
        self._messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content,
            }
        )

    def clear(self) -> None:
        """清空对话，但保留 system prompt。"""

        self._messages.clear()

    def __len__(self) -> int:
        """普通消息数量，不包含 system prompt。"""

        return len(self._messages)
