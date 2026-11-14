# Codenix

一个基于 OpenAI 兼容 Chat Completions API 的终端 AI 编程助手。

## 当前进度

- [x] Day 1：`.env` 配置读取、Chat Completions API 客户端、项目骨架
- [x] Easy：交互式多轮对话、上下文管理、`/exit` 与 `/clear`
- [x] Normal：Tool Calling、Shell 与文件工具
- [x] 流式输出：SSE 逐 token 显示，支持工具调用增量合并
- [x] 会话历史持久化：`/history`、`/save`、`/load`、`/sessions`
- [ ] Hard：Markdown 渲染、Web Search 等

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

| 命令         | 功能                              |
| ---------- | ------------------------------- |
| `/help`    | 显示帮助                            |
| `/clear`   | 清空当前对话上下文（保留 system prompt） |
| `/history` | 查看当前会话消息                        |
| `/save`    | 保存当前会话，使用时间戳作为名称               |
| `/save 名称` | 保存当前会话到指定名称                     |
| `/load`    | 列出已保存的会话                        |
| `/load 名称` | 加载指定会话                         |
| `/sessions` | 列出已保存的会话                       |
| `/model`   | 显示当前模型                          |
| `/exit`    | 退出（`/quit` 同义）                  |

### 一次性提问

```bash
codenix "用一句话介绍你自己"
```

默认使用流式输出；如果需要等待完整响应后一次性显示：

```bash
codenix --no-stream "用一句话介绍你自己"
```

也支持通过管道传入问题：

```bash
printf '用一句话介绍你自己\n' | codenix
```

流式输出通过 Chat Completions 的 SSE 接口实现，模型返回文本时会逐段写入终端。
工具调用同样支持流式参数，Agent 会自动把多个 `tool_calls` 增量拼接成完整参数后再执行。

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

## 会话持久化

会话以 JSON 文件保存在项目根目录的 `sessions/` 中，该目录已加入 `.gitignore`。

```text
sessions/
├── 2026-11-14_190000.json
└── gcd-debug.json
```

交互模式示例：

```text
你 > /save gcd-debug
会话已保存：gcd-debug

你 > /save
会话已保存：2026-11-14_190000

你 > /load gcd-debug
已加载会话：gcd-debug（4 条消息，更新于 2026-11-14T19:00:00+00:00）
```

会话文件包含：

- `system_prompt`
- 完整 `messages`，包括 assistant 的 `tool_calls` 和 `tool` 执行结果
- 保存时使用的模型和工作目录
- `created_at` / `updated_at` 时间戳

同名会话再次保存时会覆盖，但保留最初的 `created_at`。会话名不能包含 `/`、`\` 或 `..`，避免路径穿越。

## 测试

```bash
pytest
```

测试使用 `httpx.MockTransport` 模拟 API，不会发送真实请求，也不会产生费用。
