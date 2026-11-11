from __future__ import annotations

from codenix.history import ConversationHistory


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
