"""读取 Codenix 的运行时配置。

配置来源优先级：进程环境变量 > 项目根目录下的 ``.env`` 文件。
这样既方便本地开发，也方便在 CI 或容器中直接注入环境变量。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"

DEFAULT_TIMEOUT = 60.0


class ConfigError(ValueError):
    """配置缺失或取值非法。"""


@dataclass(frozen=True, slots=True)
class Config:
    """一次运行所需的模型 API 配置。"""

    base_url: str
    api_key: str
    model: str
    timeout: float = DEFAULT_TIMEOUT

    @property
    def chat_completions_url(self) -> str:
        """OpenAI 兼容的 Chat Completions 端点。"""

        return f"{self.base_url}/chat/completions"


def load_config(env_file: Path | str | None = None) -> Config:
    """加载 ``.env`` 并构造 :class:`Config`。

    ``env_file`` 不存在时不会报错，仍然会尝试读取进程环境变量。
    """

    path = Path(env_file) if env_file is not None else DEFAULT_ENV_FILE
    if path.is_file():
        # override=False：已经存在的环境变量优先于 .env 中的同名变量。
        load_dotenv(dotenv_path=path, override=False)

    base_url = _required("BASE_URL").rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        raise ConfigError("BASE_URL 必须以 http:// 或 https:// 开头")

    return Config(
        base_url=base_url,
        api_key=_required("API_KEY"),
        model=_required("MODEL"),
        timeout=_read_timeout(),
    )


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigError(
            f"缺少环境变量 {name}，请参考 .env.example 配置项目根目录下的 .env 文件"
        )
    return value


def _read_timeout() -> float:
    raw = os.getenv("REQUEST_TIMEOUT", "").strip()
    if not raw:
        return DEFAULT_TIMEOUT

    try:
        timeout = float(raw)
    except ValueError as exc:
        raise ConfigError("REQUEST_TIMEOUT 必须是数字（单位：秒）") from exc

    if timeout <= 0:
        raise ConfigError("REQUEST_TIMEOUT 必须大于 0")
    return timeout
