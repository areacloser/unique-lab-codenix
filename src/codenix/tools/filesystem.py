"""文件系统工具：读取、写入、列目录和文本搜索。

所有路径都相对于工作目录解析，并通过 ``resolve()`` 检查是否越界。
"""

from __future__ import annotations

import fnmatch
import os
import re
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from .base import (
    ToolContext,
    ToolDenied,
    ToolError,
    optional_bool,
    optional_int,
    optional_string,
    require_string,
)

MAX_READ_BYTES = 2 * 1024 * 1024
MAX_READ_LINES = 2000
MAX_LINE_CHARS = 2000
MAX_LIST_ENTRIES = 500
MAX_SEARCH_FILE_BYTES = 1 * 1024 * 1024
MAX_SEARCH_RESULTS = 200

IGNORED_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
}


def read_file(arguments: Mapping[str, Any], context: ToolContext) -> str:
    path = context.resolve_path(require_string(arguments, "path"))
    relative = context.relative_path(path)

    if not path.exists():
        raise ToolError(f"文件不存在：{relative}")
    if not path.is_file():
        raise ToolError(f"不是普通文件：{relative}")

    try:
        if path.stat().st_size > MAX_READ_BYTES:
            raise ToolError(f"文件过大，最多读取 {MAX_READ_BYTES} 字节")
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ToolError(f"读取文件失败：{exc}") from exc

    if "\x00" in text:
        raise ToolError("疑似二进制文件，拒绝读取")

    lines = text.splitlines()
    total = len(lines)
    start = optional_int(arguments, "start_line", 1, minimum=1)
    end = _end_line(arguments, start, total)

    if total == 0:
        return f"{relative}（空文件）"
    if start > total:
        return f"{relative} 只有 {total} 行，start_line={start} 超出范围"

    end = min(end, total, start + MAX_READ_LINES - 1)
    numbered = []
    for number in range(start, end + 1):
        line = lines[number - 1]
        if len(line) > MAX_LINE_CHARS:
            line = line[:MAX_LINE_CHARS] + " ..."
        numbered.append(f"{number:>6}\t{line}")

    header = f"{relative}（第 {start}-{end} 行，共 {total} 行）"
    if end < total:
        header += "；内容已截断"
    return "\n".join([header, *numbered])


def write_file(arguments: Mapping[str, Any], context: ToolContext) -> str:
    path = context.resolve_path(require_string(arguments, "path"))
    content = arguments.get("content")
    if not isinstance(content, str):
        raise ToolError("参数 content 必须是字符串")

    relative = context.relative_path(path)
    if path.exists() and path.is_dir():
        raise ToolError(f"目标是一个目录：{relative}")

    description = f"写入文件：{relative}\n字符数：{len(content)}"
    if not context.request_approval(description):
        raise ToolDenied("用户未批准写入文件")

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise ToolError(f"写入文件失败：{exc}") from exc

    return f"已写入 {relative}（{len(content)} 字符）"


def list_files(arguments: Mapping[str, Any], context: ToolContext) -> str:
    path = context.resolve_path(optional_string(arguments, "path", "."))
    if not path.exists():
        raise ToolError(f"路径不存在：{context.relative_path(path)}")

    if path.is_file():
        return f"路径：{context.relative_path(path)}\n{context.relative_path(path)}"

    recursive = optional_bool(arguments, "recursive", False)
    max_depth = optional_int(
        arguments, "max_depth", 2, minimum=1, maximum=10
    )
    entries = _list_entries(path, recursive=recursive, max_depth=max_depth)
    total = len(entries)
    truncated = total > MAX_LIST_ENTRIES
    visible = entries[:MAX_LIST_ENTRIES]

    header = f"目录：{context.relative_path(path)}（{total} 项）"
    if truncated:
        header += f"，仅显示前 {MAX_LIST_ENTRIES} 项"
    if not visible:
        return header + "\n(空)"

    result = [header, *visible]
    if truncated:
        result.append("...（列表已截断）")
    return "\n".join(result)


def search_text(arguments: Mapping[str, Any], context: ToolContext) -> str:
    pattern = require_string(arguments, "pattern")
    path = context.resolve_path(optional_string(arguments, "path", "."))
    file_glob = optional_string(arguments, "file_glob", "*")
    case_sensitive = optional_bool(arguments, "case_sensitive", False)
    max_results = optional_int(
        arguments, "max_results", 100, minimum=1, maximum=MAX_SEARCH_RESULTS
    )

    if not path.exists():
        raise ToolError(f"路径不存在：{context.relative_path(path)}")

    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        regex = re.compile(pattern, flags)
    except re.error as exc:
        raise ToolError(f"正则表达式无效：{exc}") from exc

    files: Iterable[Path] = (
        [path] if path.is_file() else _iter_files(path)
    )
    results: list[str] = []
    truncated = False

    for file in files:
        if path.is_file():
            relative = context.relative_path(file)
        else:
            relative = file.relative_to(path).as_posix()
            if not fnmatch.fnmatch(relative, file_glob):
                continue

        try:
            if not _is_inside_workspace(file, context):
                continue
            if file.stat().st_size > MAX_SEARCH_FILE_BYTES:
                continue
            raw = file.read_bytes()
        except OSError:
            continue

        if b"\x00" in raw[:8192]:
            continue

        text = raw.decode("utf-8", errors="replace")
        for line_number, line in enumerate(text.splitlines(), 1):
            if not regex.search(line):
                continue
            snippet = line.strip()
            if len(snippet) > 300:
                snippet = snippet[:300] + "..."
            results.append(f"{relative}:{line_number}: {snippet}")
            if len(results) >= max_results:
                truncated = True
                break

        if truncated:
            break

    header = f"搜索 {pattern!r}（{len(results)} 条结果）"
    if not results:
        return header + "\n未找到匹配"

    result = [header, *results]
    if truncated:
        result.append("...（结果已截断）")
    return "\n".join(result)


def _end_line(
    arguments: Mapping[str, Any],
    start: int,
    total: int,
) -> int:
    raw = arguments.get("end_line")
    if raw is None:
        return max(total, start)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ToolError("参数 end_line 必须是整数")
    if raw < start:
        raise ToolError("end_line 不能小于 start_line")
    return raw


def _list_entries(
    root: Path,
    *,
    recursive: bool,
    max_depth: int,
) -> list[str]:
    entries: list[str] = []

    if not recursive:
        for entry in sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name)):
            if entry.is_dir() and entry.name in IGNORED_DIRS:
                continue
            suffix = "/" if entry.is_dir() else ""
            entries.append(entry.name + suffix)
        return entries

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)
        relative_dir = current.relative_to(root)
        depth = len(relative_dir.parts)

        dirnames[:] = sorted(
            name
            for name in dirnames
            if name not in IGNORED_DIRS
            and not (current / name).is_symlink()
        )
        if depth >= max_depth:
            dirnames[:] = []
            continue

        for name in dirnames:
            entries.append((relative_dir / name).as_posix() + "/")
        for name in sorted(filenames):
            entries.append((relative_dir / name).as_posix())

    entries.sort()
    return entries


def _iter_files(root: Path) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)
        dirnames[:] = sorted(
            name
            for name in dirnames
            if name not in IGNORED_DIRS
            and not (current / name).is_symlink()
        )
        for name in sorted(filenames):
            yield current / name


def _is_inside_workspace(path: Path, context: ToolContext) -> bool:
    if not path.is_symlink():
        return True
    try:
        return path.resolve().is_relative_to(context.workspace)
    except OSError:
        return False
