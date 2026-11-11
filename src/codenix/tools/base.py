"""工具运行时共享的类型、异常和参数校验。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ToolError(RuntimeError):
    """工具参数错误或执行失败。"""


class ToolDenied(ToolError):
    """用户拒绝执行该工具。"""


Approver = Callable[[str], bool]


@dataclass
class ToolContext:
    """工具执行时共享的环境与审批入口。"""

    workspace: Path
    approve: Approver | None = None
    shell_timeout: float = 30.0

    def __post_init__(self) -> None:
        self.workspace = self.workspace.resolve()
        if not self.workspace.is_dir():
            raise ToolError(f"工作目录不存在：{self.workspace}")

    def request_approval(self, description: str) -> bool:
        """没有审批回调时默认拒绝有副作用的操作。"""

        if self.approve is None:
            return False
        return bool(self.approve(description))

    def resolve_path(self, raw_path: str) -> Path:
        """把工具参数中的路径限制在工作目录内。"""

        if not raw_path.strip():
            raise ToolError("path 必须是非空字符串")

        candidate = (self.workspace / raw_path).resolve()
        if not candidate.is_relative_to(self.workspace):
            raise ToolError(f"路径越出工作目录：{raw_path}")
        return candidate

    def relative_path(self, path: Path) -> str:
        return path.relative_to(self.workspace).as_posix() or "."


ToolHandler = Callable[[Mapping[str, Any], ToolContext], str]


def require_string(arguments: Mapping[str, Any], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ToolError(f"参数 {name} 必须是非空字符串")
    return value


def optional_string(
    arguments: Mapping[str, Any],
    name: str,
    default: str,
) -> str:
    value = arguments.get(name, default)
    if not isinstance(value, str):
        raise ToolError(f"参数 {name} 必须是字符串")
    return value


def optional_bool(
    arguments: Mapping[str, Any],
    name: str,
    default: bool,
) -> bool:
    value = arguments.get(name, default)
    if not isinstance(value, bool):
        raise ToolError(f"参数 {name} 必须是布尔值")
    return value


def optional_int(
    arguments: Mapping[str, Any],
    name: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    value = arguments.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolError(f"参数 {name} 必须是整数")
    if minimum is not None and value < minimum:
        raise ToolError(f"参数 {name} 不能小于 {minimum}")
    if maximum is not None and value > maximum:
        raise ToolError(f"参数 {name} 不能大于 {maximum}")
    return value


def optional_float(
    arguments: Mapping[str, Any],
    name: str,
    default: float,
) -> float:
    value = arguments.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolError(f"参数 {name} 必须是数字")
    return float(value)
