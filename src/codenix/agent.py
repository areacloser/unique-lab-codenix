"""Agent 核心层。

后续在这里实现：

1. 维护 ``messages`` 对话历史；
2. 调用 :class:`codenix.client.ChatClient`；
3. 解析并执行 ``tool_calls``；
4. 把工具结果以 ``role="tool"`` 消息回传模型；
5. 限制最大工具循环轮数。

Day 1 暂不实现 Agent Loop。
"""
