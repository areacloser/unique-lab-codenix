from __future__ import annotations

import os
from pathlib import Path

import pytest

from codenix.config import DEFAULT_TIMEOUT, ConfigError, load_config

ENV_KEYS = ("BASE_URL", "API_KEY", "MODEL", "REQUEST_TIMEOUT")


@pytest.fixture(autouse=True)
def clean_config_env():
    """避免真实 .env 或外部环境变量影响测试。"""

    saved = {key: os.environ.get(key) for key in ENV_KEYS}
    for key in ENV_KEYS:
        os.environ.pop(key, None)
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def write_env_file(tmp_path: Path, **values: str) -> Path:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "".join(f"{key}={value}\n" for key, value in values.items()),
        encoding="utf-8",
    )
    return env_file


def test_load_config_from_env_file(tmp_path: Path) -> None:
    env_file = write_env_file(
        tmp_path,
        BASE_URL="https://openrouter.ai/api/v1/",
        API_KEY="test-key",
        MODEL="test-model",
    )

    config = load_config(env_file)

    assert config.base_url == "https://openrouter.ai/api/v1"
    assert config.api_key == "test-key"
    assert config.model == "test-model"
    assert config.timeout == DEFAULT_TIMEOUT
    assert (
        config.chat_completions_url
        == "https://openrouter.ai/api/v1/chat/completions"
    )


def test_process_env_wins_over_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = write_env_file(
        tmp_path,
        BASE_URL="https://from-file.example/v1",
        API_KEY="file-key",
        MODEL="file-model",
    )
    monkeypatch.setenv("MODEL", "from-process")

    config = load_config(env_file)

    assert config.model == "from-process"


def test_missing_variable_raises_config_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("MODEL", "test-model")

    with pytest.raises(ConfigError, match="API_KEY"):
        load_config(tmp_path / "missing.env")


def test_invalid_base_url_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BASE_URL", "example.com/v1")
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("MODEL", "test-model")

    with pytest.raises(ConfigError, match="BASE_URL"):
        load_config(tmp_path / "missing.env")


@pytest.mark.parametrize("raw_timeout", ["abc", "0", "-1"])
def test_invalid_timeout_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw_timeout: str
) -> None:
    monkeypatch.setenv("BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("MODEL", "test-model")
    monkeypatch.setenv("REQUEST_TIMEOUT", raw_timeout)

    with pytest.raises(ConfigError, match="REQUEST_TIMEOUT"):
        load_config(tmp_path / "missing.env")
