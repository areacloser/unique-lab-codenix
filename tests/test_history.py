from __future__ import annotations

from pathlib import Path

import pytest

from codenix.history import ConversationHistory, SessionError, SessionStore


def test_system_prompt_is_always_first() -> None:
    history = ConversationHistory("请说中文")
    history.add_user("你好")

    messages = history.messages()

    assert messages[0] == {"role": "system", "content": "请说中文"}
    assert messages[1] == {"role": "user", "content": "你好"}


def test_messages_are_returned_as_deep_copies() -> None:
    history = ConversationHistory()
    history.add_assistant(
        {
            "content": None,
            "tool_calls": [{"id": "call_1", "type": "function"}],
        }
    )

    messages = history.messages()
    messages[0]["tool_calls"].append({"id": "call_2"})

    assert history.messages()[0]["tool_calls"] == [
        {"id": "call_1", "type": "function"}
    ]


def test_assistant_message_keeps_tool_calls_and_role() -> None:
    history = ConversationHistory()
    history.add_assistant(
        {
            "role": "user",
            "content": None,
            "tool_calls": [{"id": "call_1", "type": "function"}],
        }
    )

    message = history.messages()[0]

    assert message["role"] == "assistant"
    assert message["tool_calls"] == [{"id": "call_1", "type": "function"}]


def test_add_tool_result() -> None:
    history = ConversationHistory()
    history.add_tool("call_1", "12")

    assert history.messages()[-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "12",
    }


def test_clear_keeps_system_prompt() -> None:
    history = ConversationHistory("请说中文")
    history.add_user("你好")
    history.add_assistant({"content": "你好喵"})

    history.clear()

    assert len(history) == 0
    assert history.messages() == [{"role": "system", "content": "请说中文"}]


def make_session_history() -> ConversationHistory:
    history = ConversationHistory("系统提示")
    history.add_user("读取 a.txt")
    history.add_assistant(
        {
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "a.txt"}',
                    },
                }
            ],
        }
    )
    history.add_tool("call_1", "hello")
    history.add_assistant({"content": "读取完成"})
    return history


def test_history_serialization_roundtrip() -> None:
    history = make_session_history()

    restored = ConversationHistory.from_dict(history.to_dict())

    assert restored.system_prompt == "系统提示"
    assert restored.messages() == history.messages()


def test_save_and_load_session(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    history = make_session_history()

    info = store.save(history, "demo", model="test-model", workspace="/tmp/ws")
    loaded = store.load("demo")

    assert info.session_id == "demo"
    assert info.path.exists()
    assert loaded.session_id == "demo"
    assert loaded.history.messages() == history.messages()
    assert loaded.model == "test-model"
    assert loaded.workspace == "/tmp/ws"
    assert loaded.created_at
    assert loaded.updated_at


def test_save_without_name_uses_timestamp(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")

    info = store.save(make_session_history())

    assert info.session_id
    assert info.path.exists()
    assert len(info.session_id) > 0


def test_list_sessions_and_preview(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    store.save(make_session_history(), "b")
    store.save(make_session_history(), "a")

    sessions = store.list_sessions()

    assert {item.session_id for item in sessions} == {"a", "b"}
    assert all(item.message_count == 4 for item in sessions)
    assert all(item.preview == "读取 a.txt" for item in sessions)


def test_session_name_is_normalized(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")

    assert SessionStore.normalize_id(" my session ") == "my-session"
    assert SessionStore.normalize_id("demo.json") == "demo"
    assert store.save(make_session_history(), "my session").session_id == (
        "my-session"
    )


def test_session_name_rejects_path_traversal(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")

    with pytest.raises(SessionError):
        SessionStore.normalize_id("../evil")
    with pytest.raises(SessionError):
        store.load("../evil")


def test_invalid_session_file_is_rejected(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    store = SessionStore(sessions_dir)
    (sessions_dir / "bad.json").write_text("{", encoding="utf-8")

    with pytest.raises(SessionError, match="合法 JSON"):
        store.load("bad")

    assert store.list_sessions() == []


def test_overwrite_preserves_created_at(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    history = make_session_history()

    first = store.save(history, "demo")
    second = store.save(history, "demo")

    assert second.created_at == first.created_at
