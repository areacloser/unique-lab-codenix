"""Codenix 命令行入口。

Day 1 只提供一次性提问模式；交互式对话循环将在 Easy 阶段加入。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .client import APIError, ChatClient
from .config import ConfigError, load_config

SYSTEM_PROMPT_FILE = Path(__file__).resolve().parent / "prompts" / "system.md"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codenix",
        description="基于 OpenAI 兼容 Chat Completions API 的终端 AI 编程助手。",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="一次性提问内容；省略时显示帮助。",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.prompt:
        parser.print_help()
        return 2

    try:
        config = load_config()
    except ConfigError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2

    messages = [
        {"role": "system", "content": _load_system_prompt()},
        {"role": "user", "content": args.prompt},
    ]

    try:
        with ChatClient(config) as client:
            answer = client.complete(messages)
    except APIError as exc:
        print(f"API 错误：{exc}", file=sys.stderr)
        return 1

    print(answer)
    return 0


def _load_system_prompt() -> str:
    try:
        return SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return "你是 Codenix，一个运行在 Linux 终端中的 AI 编程助手。"
