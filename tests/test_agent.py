from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from codenix.agent import Agent, AgentError
from codenix.client import APIError
from codenix.history import ConversationHistory


def assistant_message(content: str) -> dict[str, Any]:
    return {"role": "assistant", "content": content}


def tool_call_message(*calls: tuple[str, str, str]) -> dict[str, Any]:
    """calls: (call_id, tool_name, arguments_json)"""

    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
            for call_id, name, arguments in calls
        ],
    }


class FakeClient:
    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self._messages = list(messages)
        self.calls: list[list[dict[str, Any]]] = []
        self.tool_options: list[tuple[Any, Any]] = []

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.calls.append(messages)
        self.tool_options.append((tools, tool_choice))
        return {"choices": [{"message": self._messages.pop(0)}]}


class FakeRegistry:
    def __init__(self, results: Mapping[str, str] | None = None) -> None:
        self.results = dict(results or {})
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {"name": "run_shell", "parameters": {}},
            }
        ]

    def execute(self, name: str, arguments: Mapping[str, Any]) -> str:
        self.calls.append((name, dict(arguments)))
        return self.results.get(name, "ok")


class FailingClient:
    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise APIError("connection failed")


def test_agent_keeps_context_across_turns() -> None:
    client = FakeClient(
        [assistant_message("18只羊喵"), assistant_message("7只黑羊喵")]
    )
    history = ConversationHistory("请说中文，并在每句末尾添加一个喵。")
    agent = Agent(client, history)

    first = agent.send("我有8只黑色羊，10只白色羊，请问我有几只羊？")
    second = agent.send("我有一只黑色羊被狼吃掉了，我还剩有几只黑色羊？")

    assert first == "18只羊喵"
    assert second == "7只黑羊喵"

    assert client.calls[0] == [
        {"role": "system", "content": "请说中文，并在每句末尾添加一个喵。"},
        {"role": "user", "content": "我有8只黑色羊，10只白色羊，请问我有几只羊？"},
    ]
    assert client.calls[1] == [
        {"role": "system", "content": "请说中文，并在每句末尾添加一个喵。"},
        {"role": "user", "content": "我有8只黑色羊，10只白色羊，请问我有几只羊？"},
        {"role": "assistant", "content": "18只羊喵"},
        {"role": "user", "content": "我有一只黑色羊被狼吃掉了，我还剩有几只黑色羊？"},
    ]

    assert history.messages() == [
        {"role": "system", "content": "请说中文，并在每句末尾添加一个喵。"},
        {"role": "user", "content": "我有8只黑色羊，10只白色羊，请问我有几只羊？"},
        {"role": "assistant", "content": "18只羊喵"},
        {"role": "user", "content": "我有一只黑色羊被狼吃掉了，我还剩有几只黑色羊？"},
        {"role": "assistant", "content": "7只黑羊喵"},
    ]


def test_agent_executes_tool_and_returns_final_answer() -> None:
    client = FakeClient(
        [
            tool_call_message(
                ("call_1", "run_shell", '{"command": "ls"}')
            ),
            assistant_message("目录里有 gcd.c"),
        ]
    )
    registry = FakeRegistry({"run_shell": "gcd.c"})
    history = ConversationHistory("系统提示")
    tool_calls: list[tuple[str, dict[str, Any]]] = []
    tool_results: list[tuple[str, str]] = []
    agent = Agent(
        client,
        history,
        tool_registry=registry,
        on_tool_call=lambda name, args: tool_calls.append((name, dict(args))),
        on_tool_result=lambda name, result: tool_results.append((name, result)),
    )

    assert agent.send("看看目录") == "目录里有 gcd.c"

    assert registry.calls == [("run_shell", {"command": "ls"})]
    assert tool_calls == [("run_shell", {"command": "ls"})]
    assert tool_results == [("run_shell", "gcd.c")]

    assert client.tool_options[0][1] == "auto"
    assert client.calls[1] == [
        {"role": "system", "content": "系统提示"},
        {"role": "user", "content": "看看目录"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "run_shell", "arguments": '{"command": "ls"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "gcd.c"},
    ]
    assert history.messages()[-1] == {
        "role": "assistant",
        "content": "目录里有 gcd.c",
    }


def test_agent_handles_multiple_tool_calls_in_one_response() -> None:
    client = FakeClient(
        [
            tool_call_message(
                ("call_1", "read_file", '{"path": "a.txt"}'),
                ("call_2", "read_file", '{"path": "b.txt"}'),
            ),
            assistant_message("完成"),
        ]
    )
    registry = FakeRegistry({"read_file": "内容"})
    agent = Agent(client, ConversationHistory("系统"), tool_registry=registry)

    assert agent.send("读取两个文件") == "完成"
    assert [call[0] for call in registry.calls] == ["read_file", "read_file"]
    assert agent.history.messages()[-3:] == [
        {"role": "tool", "tool_call_id": "call_1", "content": "内容"},
        {"role": "tool", "tool_call_id": "call_2", "content": "内容"},
        {"role": "assistant", "content": "完成"},
    ]


def test_invalid_tool_arguments_are_returned_to_model() -> None:
    client = FakeClient(
        [
            tool_call_message(("call_1", "run_shell", "not-json")),
            assistant_message("参数格式有问题"),
        ]
    )
    registry = FakeRegistry()
    agent = Agent(client, ConversationHistory("系统"), tool_registry=registry)

    assert agent.send("执行") == "参数格式有问题"
    assert registry.calls == []
    tool_message = agent.history.messages()[-2]
    assert tool_message["role"] == "tool"
    assert "不是合法 JSON" in tool_message["content"]


def test_max_tool_rounds_stops_the_loop() -> None:
    client = FakeClient(
        [
            tool_call_message(("call_1", "run_shell", '{"command": "ls"}')),
            tool_call_message(("call_2", "run_shell", '{"command": "ls"}')),
            tool_call_message(("call_3", "run_shell", '{"command": "ls"}')),
        ]
    )
    agent = Agent(
        client,
        ConversationHistory("系统"),
        tool_registry=FakeRegistry(),
        max_tool_rounds=2,
    )

    with pytest.raises(AgentError, match="最大工具调用轮数"):
        agent.send("循环")

    assert len(client.calls) == 3
    assert len(agent.history) == 6  # user + 3 个 assistant + 2 个 tool


def test_failed_request_does_not_change_history() -> None:
    history = ConversationHistory("系统提示")
    agent = Agent(FailingClient(), history)

    with pytest.raises(APIError, match="connection failed"):
        agent.send("你好")

    assert history.messages() == [{"role": "system", "content": "系统提示"}]


def test_empty_input_is_rejected() -> None:
    agent = Agent(FakeClient([]), ConversationHistory("系统提示"))

    with pytest.raises(ValueError, match="不能为空"):
        agent.send("   ")
