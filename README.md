# Codenix

一个基于 OpenAI 兼容 Chat Completions API 的终端 AI 编程助手。

## 当前进度

- [x] Day 1：`.env` 配置读取、Chat Completions API 客户端、项目骨架
- [ ] Easy：交互式多轮对话
- [ ] Normal：Tool Calling、Shell 与文件工具
- [ ] Hard：Slash Commands、历史持久化、终端体验

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

## 运行一次请求

```bash
codenix "用一句话介绍你自己"
```

如果尚未安装为命令，也可以直接运行：

```bash
PYTHONPATH=src python -m codenix "用一句话介绍你自己"
```

当前阶段使用一次性提问模式，交互式对话和工具调用会在后续迭代中加入。

## 测试

```bash
pytest
```

测试使用 `httpx.MockTransport` 模拟 API，不会发送真实请求，也不会产生费用。
