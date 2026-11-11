"""Shell 命令执行工具。

命令在工作目录中通过 ``bash``/``sh`` 执行，执行前必须得到用户批准。
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Mapping
from typing import Any

from .base import (
    ToolContext,
    ToolDenied,
    ToolError,
    optional_float,
    require_string,
)

MAX_OUTPUT_CHARS = 20_000
MAX_TIMEOUT = 300.0
SHELL = shutil.which("bash") or shutil.which("sh")


def run_shell(arguments: Mapping[str, Any], context: ToolContext) -> str:
    command = require_string(arguments, "command")
    timeout = optional_float(arguments, "timeout", context.shell_timeout)

    if timeout <= 0:
        raise ToolError("timeout 必须大于 0")
    if timeout > MAX_TIMEOUT:
        raise ToolError(f"timeout 不能大于 {MAX_TIMEOUT:g} 秒")

    description = (
        f"工作目录：{context.workspace}\n"
        f"命令：\n{command}\n"
        f"超时：{timeout:g} 秒"
    )
    if not context.request_approval(description):
        raise ToolDenied("用户未批准执行该 Shell 命令")

    try:
        completed = subprocess.run(
            command,
            shell=True,
            executable=SHELL,
            cwd=context.workspace,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return _format_result(
            "命令执行超时",
            124,
            _as_text(exc.stdout),
            _as_text(exc.stderr),
        )

    return _format_result(
        "",
        completed.returncode,
        completed.stdout,
        completed.stderr,
    )


def _format_result(
    note: str,
    exit_code: int,
    stdout: str,
    stderr: str,
) -> str:
    parts = [f"exit_code: {exit_code}"]
    if note:
        parts.append(note)
    parts.append(f"stdout:\n{_truncate(stdout) or '(空)'}")
    parts.append(f"stderr:\n{_truncate(stderr) or '(空)'}")
    return "\n".join(parts)


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return (
        text[:MAX_OUTPUT_CHARS]
        + f"\n...（输出已截断，共 {len(text)} 字符）"
    )


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
