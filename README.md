# Codenix

一个基于 OpenAI 兼容 Chat Completions API 的终端 AI 编程助手。

## 当前进度

- [x] Day 1：`.env` 配置读取、Chat Completions API 客户端、项目骨架
- [x] Easy：交互式多轮对话、上下文管理、`/exit` 与 `/clear`
- [x] Normal：Tool Calling、Shell 与文件工具
- [ ] Hard：更多 Slash Commands、历史持久化、终端体验

## 环境准备

要求 Python 3.12+。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

复制环境变量模板并填写自己的配置：

```bash
cp .env.example .env
```

`.env` 中包含：

```dotenv
BASE_URL=https://openrouter.ai/api/v1
API_KEY=your_api_key_here
MODEL=your-model-id
# 可选，单位：秒
REQUEST_TIMEOUT=60
```

`.env` 已被 `.gitignore` 忽略，请勿提交真实的 API Key。

## 运行

### 交互式多轮对话

```bash
codenix
```

进入后会显示提示符 `你 >`，输入内容即可连续对话。可用命令：

| 命令      | 功能                              |
| ------- | ------------------------------- |
| `/help` | 显示帮助                            |
| `/clear` | 清空当前对话上下文（保留 system prompt） |
| `/model` | 显示当前模型                          |
| `/exit` | 退出（`/quit` 同义）                  |

### 一次性提问

```bash
codenix "用一句话介绍你自己"
```

也支持通过管道传入问题：

```bash
printf '用一句话介绍你自己\n' | codenix
```

如果尚未安装为命令，可以在所有命令前加上 `PYTHONPATH=src`，例如：

```bash
PYTHONPATH=src python -m codenix
```

## 工具与工作目录

模型可以调用以下工具：

| 工具            | 功能                     | 是否需要批准 |
| ------------- | ---------------------- | ------- |
| `run_shell`   | 在工作目录中执行 Shell 命令      | 是       |
| `read_file`   | 读取文件指定行范围，返回带行号内容      | 否       |
| `write_file`  | 写入完整文件内容，自动创建父目录       | 是       |
| `list_files`  | 列出文件和子目录               | 否       |
| `search_text` | 按正则表达式搜索文本             | 否       |

所有工具路径都被限制在 `workspace/` 中，`../` 和指向外部的符号链接会被拒绝。
Shell 命令和写文件操作会在终端中请求 `y/N` 批准；非交互环境下默认拒绝。

例如，可以要求 Codenix 完成 `workspace/gcd.c`：

```text
你 > 请阅读 workspace 中的 gcd.c，实现 gcd 函数，然后编译并运行验证。
```

## 测试

```bash
pytest
```

测试使用 `httpx.MockTransport` 模拟 API，不会发送真实请求，也不会产生费用。
