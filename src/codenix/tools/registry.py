"""工具注册、Schema 加载与分发。

模型只看到 ``schemas/tools.json`` 中的 JSON Schema；真正执行的是本包中各模块
的 Python 函数。Agent Loop 通过统一的 :class:`ToolRegistry` 调用工具。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from ..config import PROJECT_ROOT
from .base import (
    ToolContext,
    ToolDenied,
    ToolError,
    ToolHandler,
)

DEFAULT_SCHEMAS_FILE = PROJECT_ROOT / "schemas" / "tools.json"


class ToolRegistry:
    """工具名到处理函数的映射，同时负责加载给模型看的 Schema。"""

    def __init__(self, context: ToolContext) -> None:
        self.context: ToolContext = context
        self._handlers: dict[str, ToolHandler] = {}
        self._schemas: dict[str, dict[str, Any]] = {}

    @classmethod
    def from_defaults(
        cls,
        context: ToolContext,
        schemas_file: Path = DEFAULT_SCHEMAS_FILE,
    ) -> ToolRegistry:
        # 延迟导入，避免 tools 子模块与 registry 之间形成循环依赖。
        from . import filesystem, shell

        registry = cls(context)
        registry.register("run_shell", shell.run_shell)
        registry.register("read_file", filesystem.read_file)
        registry.register("write_file", filesystem.write_file)
        registry.register("list_files", filesystem.list_files)
        registry.register("search_text", filesystem.search_text)
        registry.load_schemas(schemas_file)
        return registry

    def register(self, name: str, handler: ToolHandler) -> None:
        if not name:
            raise ValueError("工具名不能为空")
        if name in self._handlers:
            raise ValueError(f"工具重复注册：{name}")
        self._handlers[name] = handler

    def load_schemas(self, path: Path = DEFAULT_SCHEMAS_FILE) -> None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ToolError(f"工具 Schema 文件不存在：{path}") from exc
        except json.JSONDecodeError as exc:
            raise ToolError(f"工具 Schema 不是合法 JSON：{path}") from exc

        if not isinstance(data, list):
            raise ToolError("tools.json 顶层必须是数组")

        schemas: dict[str, dict[str, Any]] = {}
        for item in data:
            if not isinstance(item, dict):
                raise ToolError("tools.json 中的每一项都必须是对象")
            function = item.get("function")
            if not isinstance(function, dict):
                raise ToolError("工具定义缺少 function 对象")
            name = function.get("name")
            if not isinstance(name, str) or not name:
                raise ToolError("工具定义缺少 function.name")
            if name not in self._handlers:
                raise ToolError(f"Schema 中的工具没有对应实现：{name}")
            schemas[name] = item

        missing = set(self._handlers) - set(schemas)
        if missing:
            names = ", ".join(sorted(missing))
            raise ToolError(f"以下工具缺少 JSON Schema：{names}")

        self._schemas = schemas

    def schemas(self) -> list[dict[str, Any]]:
        """返回可以直接放入请求体 ``tools`` 字段的列表。"""

        return [deepcopy(self._schemas[name]) for name in self._handlers]

    def execute(self, name: str, arguments: Mapping[str, Any]) -> str:
        """执行工具，并把所有普通异常转换为可回传给模型的字符串。"""

        handler = self._handlers.get(name)
        if handler is None:
            return f"错误：未知工具 {name!r}"

        try:
            result = handler(arguments, self.context)
        except ToolDenied as exc:
            return f"已拒绝：{exc}"
        except ToolError as exc:
            return f"错误：{exc}"
        except Exception as exc:  # 工具失败不能导致 Agent 崩溃
            return f"工具执行异常：{type(exc).__name__}: {exc}"

        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False)


__all__ = [
    "DEFAULT_SCHEMAS_FILE",
    "ToolContext",
    "ToolDenied",
    "ToolError",
    "ToolRegistry",
]
