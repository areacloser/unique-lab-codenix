"""对话历史与会话持久化。

``ConversationHistory`` 负责内存中的消息列表；``SessionStore`` 负责把完整的
对话（包括 assistant 的 tool_calls 和 tool 执行结果）保存为 ``sessions/*.json``。
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SESSION_VERSION = 1
MAX_SESSION_ID_LENGTH = 80


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

        保留整个字典而不只保存 ``content``，这样 Tool Calling 阶段才能把
        ``tool_calls`` 原样回传给模型。
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

    def to_dict(self) -> dict[str, Any]:
        """序列化为会话文件中的内容。"""

        return {
            "system_prompt": self._system_prompt,
            "messages": deepcopy(self._messages),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ConversationHistory:
        """从会话文件内容恢复历史。"""

        if not isinstance(data, Mapping):
            raise ValueError("会话数据必须是 JSON 对象")

        system_prompt = data.get("system_prompt", "")
        if system_prompt is None:
            system_prompt = ""
        if not isinstance(system_prompt, str):
            raise ValueError("system_prompt 必须是字符串")

        messages = data.get("messages", [])
        if not isinstance(messages, list):
            raise ValueError("messages 必须是数组")

        history = cls(system_prompt)
        for index, raw in enumerate(messages, 1):
            if not isinstance(raw, Mapping):
                raise ValueError(f"messages[{index}] 必须是对象")

            role = raw.get("role")
            if role == "system":
                content = raw.get("content")
                if not isinstance(content, str):
                    raise ValueError(f"messages[{index}].content 必须是字符串")
                if not history._system_prompt:
                    history._system_prompt = content.strip()
                continue

            if role == "user":
                content = raw.get("content")
                if not isinstance(content, str):
                    raise ValueError(f"messages[{index}].content 必须是字符串")
                history.add_user(content)
                continue

            if role == "assistant":
                content = raw.get("content")
                if content is not None and not isinstance(content, str):
                    raise ValueError(
                        f"messages[{index}].content 必须是字符串或 null"
                    )
                tool_calls = raw.get("tool_calls")
                if tool_calls is not None and not isinstance(tool_calls, list):
                    raise ValueError(f"messages[{index}].tool_calls 必须是数组")
                history.add_assistant(raw)
                continue

            if role == "tool":
                call_id = raw.get("tool_call_id")
                content = raw.get("content")
                if not isinstance(call_id, str) or not call_id:
                    raise ValueError(
                        f"messages[{index}].tool_call_id 必须是非空字符串"
                    )
                if not isinstance(content, str):
                    raise ValueError(f"messages[{index}].content 必须是字符串")
                history.add_tool(call_id, content)
                continue

            raise ValueError(f"messages[{index}] 的 role 非法：{role!r}")

        return history

    def __len__(self) -> int:
        """普通消息数量，不包含 system prompt。"""

        return len(self._messages)


class SessionError(ValueError):
    """会话保存或加载失败。"""


@dataclass(frozen=True)
class SessionInfo:
    """会话列表中的一条摘要。"""

    session_id: str
    path: Path
    created_at: str
    updated_at: str
    model: str
    message_count: int
    preview: str


@dataclass(frozen=True)
class LoadedSession:
    """从磁盘加载的完整会话。"""

    session_id: str
    path: Path
    history: ConversationHistory
    created_at: str
    updated_at: str
    model: str
    workspace: str


class SessionStore:
    """管理 ``sessions/`` 目录下的 JSON 会话文件。"""

    def __init__(self, sessions_dir: Path | str) -> None:
        self.sessions_dir = Path(sessions_dir).expanduser().resolve()
        try:
            self.sessions_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise SessionError(f"无法创建会话目录：{exc}") from exc
        if not self.sessions_dir.is_dir():
            raise SessionError(f"会话路径不是目录：{self.sessions_dir}")

    def save(
        self,
        history: ConversationHistory,
        session_id: str | None = None,
        *,
        model: str = "",
        workspace: str = "",
    ) -> SessionInfo:
        """保存会话；显式 session_id 已存在时会覆盖并保留 created_at。"""

        resolved_id = (
            self._new_session_id()
            if session_id is None or not session_id.strip()
            else self.normalize_id(session_id)
        )
        path = self._path(resolved_id)
        now = _utc_now()

        created_at = now
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                if (
                    isinstance(existing, dict)
                    and isinstance(existing.get("created_at"), str)
                ):
                    created_at = existing["created_at"]
            except (OSError, json.JSONDecodeError):
                pass

        data: dict[str, Any] = {
            "version": SESSION_VERSION,
            "id": resolved_id,
            "created_at": created_at,
            "updated_at": now,
            "model": model,
            "workspace": workspace,
        }
        data.update(history.to_dict())

        try:
            payload = json.dumps(data, ensure_ascii=False, indent=2)
        except TypeError as exc:
            raise SessionError("会话包含无法序列化的数据") from exc

        try:
            self._atomic_write(path, payload)
        except OSError as exc:
            raise SessionError(f"写入会话失败：{exc}") from exc

        return self._make_info(resolved_id, path, data, history)

    def load(self, session_id: str) -> LoadedSession:
        resolved_id = self.normalize_id(session_id)
        path = self._path(resolved_id)

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise SessionError(f"会话不存在：{resolved_id}") from exc
        except OSError as exc:
            raise SessionError(f"读取会话失败：{exc}") from exc
        except json.JSONDecodeError as exc:
            raise SessionError(f"会话文件不是合法 JSON：{path}") from exc

        if not isinstance(data, dict):
            raise SessionError("会话文件顶层必须是 JSON 对象")

        version = data.get("version", SESSION_VERSION)
        if version != SESSION_VERSION:
            raise SessionError(f"不支持的会话版本：{version!r}")

        try:
            history = ConversationHistory.from_dict(data)
        except ValueError as exc:
            raise SessionError(f"会话数据无效：{exc}") from exc

        return LoadedSession(
            session_id=resolved_id,
            path=path,
            history=history,
            created_at=_string_field(data, "created_at"),
            updated_at=_string_field(data, "updated_at"),
            model=_string_field(data, "model"),
            workspace=_string_field(data, "workspace"),
        )

    def list_sessions(self) -> list[SessionInfo]:
        """列出有效会话，按 updated_at 从新到旧排序。"""

        sessions: list[SessionInfo] = []
        try:
            paths = sorted(self.sessions_dir.glob("*.json"))
        except OSError:
            return sessions

        for path in paths:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    continue
                version = data.get("version", SESSION_VERSION)
                if version != SESSION_VERSION:
                    continue
                history = ConversationHistory.from_dict(data)
            except (OSError, json.JSONDecodeError, ValueError):
                continue

            sessions.append(self._make_info(path.stem, path, data, history))

        sessions.sort(key=lambda item: item.updated_at, reverse=True)
        return sessions

    @staticmethod
    def normalize_id(session_id: str) -> str:
        """校验并规范化会话名，阻止路径穿越。"""

        if not isinstance(session_id, str):
            raise SessionError("会话名必须是字符串")

        name = session_id.strip()
        if name.lower().endswith(".json"):
            name = name[:-5]
        if not name:
            raise SessionError("会话名不能为空")
        if ".." in name or "/" in name or "\\" in name:
            raise SessionError("会话名不能包含路径分隔符或 '..'")

        name = re.sub(r"[\x00-\x1f:*?\"<>|]+", "-", name)
        name = re.sub(r"\s+", "-", name).strip(".-_")
        if not name:
            raise SessionError("会话名不能为空")
        if len(name) > MAX_SESSION_ID_LENGTH:
            raise SessionError(
                f"会话名过长，最多 {MAX_SESSION_ID_LENGTH} 个字符"
            )
        return name

    def _path(self, session_id: str) -> Path:
        path = (self.sessions_dir / f"{session_id}.json").resolve()
        if not path.is_relative_to(self.sessions_dir):
            raise SessionError("非法的会话路径")
        return path

    def _new_session_id(self) -> str:
        base = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        candidate = base
        counter = 2
        while self._path(candidate).exists():
            candidate = f"{base}-{counter}"
            counter += 1
        return candidate

    def _atomic_write(self, path: Path, payload: str) -> None:
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{path.stem}-",
            suffix=".tmp",
            dir=self.sessions_dir,
        )
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                file.write(payload)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temp_path, path)
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    @staticmethod
    def _make_info(
        session_id: str,
        path: Path,
        data: Mapping[str, Any],
        history: ConversationHistory,
    ) -> SessionInfo:
        return SessionInfo(
            session_id=session_id,
            path=path,
            created_at=_string_field(data, "created_at"),
            updated_at=_string_field(data, "updated_at"),
            model=_string_field(data, "model"),
            message_count=len(history),
            preview=_preview(history.messages()),
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _string_field(data: Mapping[str, Any], name: str) -> str:
    value = data.get(name)
    return value if isinstance(value, str) else ""


def _preview(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        text = " ".join(content.split())
        if len(text) > 60:
            return text[:60] + "..."
        return text
    return ""
