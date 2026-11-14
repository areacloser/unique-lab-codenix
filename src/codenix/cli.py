"""Codenix 命令行入口。

支持两种模式：

- 交互式多轮对话（默认）；
- 一次性提问（提供 prompt 参数或通过管道传入）。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from . import __version__
from .agent import Agent, AgentError
from .client import APIError, ChatClient
from .config import PROJECT_ROOT, Config, ConfigError, load_config
from .history import (
    ConversationHistory,
    SessionError,
    SessionInfo,
    SessionStore,
)
from .tools.registry import ToolContext, ToolError, ToolRegistry

SYSTEM_PROMPT_FILE = Path(__file__).resolve().parent / "prompts" / "system.md"
WORKSPACE_DIR = PROJECT_ROOT / "workspace"
SESSIONS_DIR = PROJECT_ROOT / "sessions"

HELP_TEXT = """可用命令：
  /help              显示帮助
  /clear             清空当前对话上下文（保留 system prompt）
  /history           查看当前会话消息
  /save [name]       保存当前会话；省略 name 时使用时间戳
  /load [name]       加载会话；省略 name 时列出可用会话
  /sessions          列出已保存的会话
  /model             显示当前模型
  /exit              退出（/quit 同义）"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codenix",
        description="基于 OpenAI 兼容 Chat Completions API 的终端 AI 编程助手。",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="一次性提问内容；省略且标准输入为终端时进入交互模式。",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="关闭流式输出，等待完整响应后一次性显示。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    prompt = args.prompt
    if prompt is None and not sys.stdin.isatty():
        # 支持：echo "问题" | codenix
        prompt = sys.stdin.read().strip() or None
    if prompt is not None and not prompt.strip():
        prompt = None

    if prompt is None and not sys.stdin.isatty():
        # 管道输入为空，且没有交互终端可用。
        parser.print_help()
        return 2

    try:
        config = load_config()
    except ConfigError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2

    try:
        workspace = WORKSPACE_DIR.resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        context = ToolContext(workspace=workspace, approve=_approve)
        registry = ToolRegistry.from_defaults(context)
    except (OSError, ToolError) as exc:
        print(f"工作目录或工具初始化失败：{exc}", file=sys.stderr)
        return 2

    try:
        session_store = SessionStore(SESSIONS_DIR)
    except SessionError as exc:
        print(f"会话目录初始化失败：{exc}", file=sys.stderr)
        return 2

    client = ChatClient(config)
    agent = Agent(
        client=client,
        history=ConversationHistory(_load_system_prompt(workspace)),
        tool_registry=registry,
        on_tool_call=_show_tool_call,
        on_tool_result=_show_tool_result,
    )

    try:
        with client:
            on_token = None if args.no_stream else _print_token
            if prompt is not None:
                return _run_once(agent, prompt, on_token)
            return _run_interactive(
                agent, config, session_store, workspace, on_token
            )
    except APIError as exc:
        print(f"API 错误：{exc}", file=sys.stderr)
        return 1


def _run_once(
    agent: Agent,
    prompt: str,
    on_token: Callable[[str], None] | None,
) -> int:
    try:
        answer = agent.send(prompt, on_token=on_token)
    except APIError as exc:
        print()
        print(f"API 错误：{exc}", file=sys.stderr)
        return 1
    except AgentError as exc:
        print()
        print(f"Agent 错误：{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print()
        print("已取消。", file=sys.stderr)
        return 130

    if on_token is None:
        print(answer)
    else:
        print()
    return 0


def _run_interactive(
    agent: Agent,
    config: Config,
    session_store: SessionStore,
    workspace: Path,
    on_token: Callable[[str], None] | None,
) -> int:
    print(f"Codenix {__version__} | 模型：{config.model}")
    print("输入 /help 查看命令，/exit 退出。\n")

    while True:
        try:
            line = input("你 > ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        text = line.strip()
        if not text:
            continue

        if text.startswith("/"):
            if _handle_command(
                text, agent, config, session_store, workspace
            ):
                return 0
            continue

        print("\nCodenix：")
        try:
            answer = agent.send(text, on_token=on_token)
        except APIError as exc:
            print()
            print(f"API 错误：{exc}", file=sys.stderr)
            continue
        except AgentError as exc:
            print()
            print(f"Agent 错误：{exc}", file=sys.stderr)
            continue
        except KeyboardInterrupt:
            print()
            print("已取消本轮请求。", file=sys.stderr)
            continue

        if on_token is None:
            print(answer)
        else:
            print()


def _handle_command(
    line: str,
    agent: Agent,
    config: Config,
    session_store: SessionStore,
    workspace: Path,
) -> bool:
    """处理 Slash Command。返回 True 表示退出程序。"""

    parts = line.split(maxsplit=1)
    command = parts[0].lower()
    argument = parts[1].strip() if len(parts) > 1 else ""

    if command in {"/exit", "/quit"}:
        print("再见。")
        return True
    if command == "/clear":
        agent.history.clear()
        print("已清空当前对话上下文。")
        return False
    if command == "/help":
        print(HELP_TEXT)
        return False
    if command == "/model":
        print(f"当前模型：{config.model}")
        return False
    if command == "/history":
        _print_history(agent.history)
        return False
    if command == "/save":
        _save_session(agent, config, session_store, workspace, argument)
        return False
    if command == "/load":
        _load_session(agent, config, session_store, argument)
        return False
    if command == "/sessions":
        _print_sessions(session_store.list_sessions())
        return False

    print(f"未知命令：{command}。输入 /help 查看可用命令。")
    return False


def _save_session(
    agent: Agent,
    config: Config,
    session_store: SessionStore,
    workspace: Path,
    name: str,
) -> None:
    try:
        info = session_store.save(
            agent.history,
            name or None,
            model=config.model,
            workspace=str(workspace),
        )
    except SessionError as exc:
        print(f"保存失败：{exc}")
        return

    print(f"会话已保存：{info.session_id}")
    print(f"文件：{info.path}")


def _load_session(
    agent: Agent,
    config: Config,
    session_store: SessionStore,
    name: str,
) -> None:
    if not name:
        _print_sessions(session_store.list_sessions())
        return

    try:
        loaded = session_store.load(name)
    except SessionError as exc:
        print(f"加载失败：{exc}")
        return

    agent.history = loaded.history
    updated = loaded.updated_at or "未知时间"
    print(
        f"已加载会话：{loaded.session_id}"
        f"（{len(loaded.history)} 条消息，更新于 {updated}）"
    )
    if loaded.model and loaded.model != config.model:
        print(
            f"注意：会话保存时使用的模型是 {loaded.model}，"
            f"当前模型是 {config.model}。"
        )


def _print_sessions(sessions: list[SessionInfo]) -> None:
    if not sessions:
        print("还没有保存的会话。使用 /save [name] 保存当前会话。")
        return

    print("已保存的会话：")
    for info in sessions:
        updated = info.updated_at or "未知时间"
        model = info.model or "-"
        preview = info.preview or "-"
        print(
            f"  {info.session_id}  {updated}  "
            f"{info.message_count} 条  {model}  {preview}"
        )


def _print_history(history: ConversationHistory) -> None:
    messages = history.messages()
    if not messages:
        print("当前会话为空。")
        return

    for index, message in enumerate(messages, 1):
        role = message.get("role", "unknown")
        if role == "assistant":
            print(
                f"[{index}] assistant: "
                f"{_preview_text(message.get('content'), 500)}"
            )
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                for call in tool_calls:
                    if not isinstance(call, dict):
                        continue
                    function = call.get("function")
                    if not isinstance(function, dict):
                        continue
                    name = function.get("name", "?")
                    arguments = function.get("arguments", "")
                    print(
                        f"    tool_call {name}: "
                        f"{_preview_text(arguments, 200)}"
                    )
            continue

        if role == "tool":
            call_id = message.get("tool_call_id", "?")
            print(
                f"[{index}] tool {call_id}: "
                f"{_preview_text(message.get('content'), 500)}"
            )
            continue

        print(
            f"[{index}] {role}: "
            f"{_preview_text(message.get('content'), 500)}"
        )


def _preview_text(value: object, limit: int) -> str:
    if not isinstance(value, str):
        return "-"
    text = value.strip().replace("\n", "\\n").replace("\r", "\\r")
    if not text:
        return "-"
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def _approve(description: str) -> bool:
    """Shell 和写文件前的交互式审批。"""

    if not sys.stdin.isatty():
        print(
            f"\n[需要批准]\n{description}\n"
            "当前不是交互终端，已自动拒绝。",
            file=sys.stderr,
        )
        return False

    print(f"\n[需要批准]\n{description}")
    try:
        answer = input("允许执行？[y/N] > ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in {"y", "yes"}


def _show_tool_call(name: str, arguments: Mapping[str, Any]) -> None:
    print(f"\n[工具调用] {name}")
    print(json.dumps(dict(arguments), ensure_ascii=False, indent=2))


def _show_tool_result(name: str, result: str) -> None:
    preview = result if len(result) <= 1500 else result[:1500] + "\n..."
    print(f"[工具结果] {name}\n{preview}\n")


def _print_token(token: str) -> None:
    sys.stdout.write(token)
    sys.stdout.flush()


def _load_system_prompt(workspace: Path) -> str:
    try:
        prompt = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        prompt = "你是 Codenix，一个运行在 Linux 终端中的 AI 编程助手。"
    return prompt.replace("{{workspace}}", str(workspace))
