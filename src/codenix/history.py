"""对话历史管理。

后续在这里实现：

1. 追加 user / assistant / tool 消息；
2. 保留 assistant 的完整 ``tool_calls`` 字段；
3. 把会话以 JSON 保存到 ``sessions/``；
4. 从会话文件恢复上下文。

Day 1 暂不实现持久化。
"""
