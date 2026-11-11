from __future__ import annotations

from pathlib import Path

import pytest

from codenix.tools.registry import ToolContext, ToolRegistry


def make_context(workspace: Path, *, allow: bool = True) -> ToolContext:
    return ToolContext(workspace=workspace, approve=lambda _: allow)


def make_registry(workspace: Path, *, allow: bool = True) -> ToolRegistry:
    return ToolRegistry.from_defaults(make_context(workspace, allow=allow))


def test_default_registry_loads_all_schemas(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = make_registry(workspace)

    names = {
        item["function"]["name"]
        for item in registry.schemas()
    }

    assert names == {
        "run_shell",
        "read_file",
        "write_file",
        "list_files",
        "search_text",
    }


def test_read_file_returns_numbered_range(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "a.txt").write_text(
        "first\nsecond\nthird\n", encoding="utf-8"
    )
    registry = make_registry(workspace)

    result = registry.execute(
        "read_file",
        {"path": "a.txt", "start_line": 2, "end_line": 3},
    )

    assert "第 2-3 行，共 3 行" in result
    assert "     2\tsecond" in result
    assert "     3\tthird" in result
    assert "first" not in result


def test_read_file_rejects_path_traversal(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")
    registry = make_registry(workspace)

    result = registry.execute("read_file", {"path": "../secret.txt"})

    assert result.startswith("错误：")
    assert "越出工作目录" in result
    assert "secret" not in result


def test_read_file_rejects_symlink_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    try:
        (workspace / "link.txt").symlink_to(outside)
    except OSError:
        pytest.skip("当前环境无法创建符号链接")
    registry = make_registry(workspace)

    result = registry.execute("read_file", {"path": "link.txt"})

    assert "越出工作目录" in result


def test_write_file_requires_approval(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = make_registry(workspace, allow=False)

    result = registry.execute(
        "write_file",
        {"path": "out.txt", "content": "hello"},
    )

    assert "已拒绝" in result
    assert not (workspace / "out.txt").exists()


def test_write_file_creates_parent_and_content(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = make_registry(workspace)

    result = registry.execute(
        "write_file",
        {"path": "sub/out.txt", "content": "hello\nworld\n"},
    )

    assert "已写入 sub/out.txt" in result
    assert (workspace / "sub" / "out.txt").read_text(encoding="utf-8") == (
        "hello\nworld\n"
    )


def test_write_file_rejects_path_traversal(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = make_registry(workspace)

    result = registry.execute(
        "write_file",
        {"path": "../evil.txt", "content": "x"},
    )

    assert "越出工作目录" in result
    assert not (tmp_path / "evil.txt").exists()


def test_list_files_recursive(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "a.txt").write_text("a", encoding="utf-8")
    (workspace / "sub").mkdir()
    (workspace / "sub" / "b.c").write_text("b", encoding="utf-8")
    registry = make_registry(workspace)

    result = registry.execute(
        "list_files",
        {"path": ".", "recursive": True, "max_depth": 2},
    )

    assert "a.txt" in result
    assert "sub/" in result
    assert "sub/b.c" in result


def test_search_text_filters_by_glob(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "a.c").write_text(
        "int main(void) {\n    return 0;\n}\n", encoding="utf-8"
    )
    (workspace / "b.txt").write_text("main\n", encoding="utf-8")
    registry = make_registry(workspace)

    result = registry.execute(
        "search_text",
        {"pattern": "main", "file_glob": "*.c"},
    )

    assert "a.c:1:" in result
    assert "b.txt" not in result


def test_search_text_rejects_invalid_regex(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = make_registry(workspace)

    result = registry.execute("search_text", {"pattern": "["})

    assert result.startswith("错误：")
    assert "正则表达式无效" in result


def test_shell_runs_in_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = make_registry(workspace)

    result = registry.execute("run_shell", {"command": "pwd"})

    assert "exit_code: 0" in result
    assert str(workspace.resolve()) in result


def test_shell_requires_approval(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = make_registry(workspace, allow=False)

    result = registry.execute("run_shell", {"command": "touch marker"})

    assert "已拒绝" in result
    assert not (workspace / "marker").exists()


def test_shell_returns_nonzero_exit_code(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = make_registry(workspace)

    result = registry.execute("run_shell", {"command": "exit 3"})

    assert "exit_code: 3" in result


def test_shell_timeout(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = make_registry(workspace)

    result = registry.execute(
        "run_shell",
        {"command": "sleep 1", "timeout": 0.1},
    )

    assert "exit_code: 124" in result
    assert "超时" in result


def test_unknown_tool_returns_error(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = make_registry(workspace)

    assert "未知工具" in registry.execute("not_a_tool", {})
