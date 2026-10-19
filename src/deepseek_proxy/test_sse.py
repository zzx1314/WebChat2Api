"""
手动测试 SSE 解析 Pipeline (不依赖真实 API)
"""
import asyncio
import json
from deepseek_proxy.sse_parser import parse_sse_stream, parse_dsframe_stream, full_sse_pipeline


def _sse_event(payload: dict) -> bytes:
    return b"data: " + json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n\n"


# 模拟 DeepSeek SSE 响应
MOCK_SSE = (
    b'event: ready\ndata: {}\n\n'
    b'data: {"v":{"response":{"thinking_enabled":false,"fragments":[{"type":"ANSWER","content":"hello"}]}}}\n\n'
    b'data: {"p":"response/fragments/-1/content","o":"APPEND","v":" world"}\n\n'
    b'data: {"v":"! I am"}\n\n'
    b'data: {"v":" DeepSeek."}\n\n'
    b'data: {"p":"response/status","o":"APPEND","v":"FINISHED"}\n\n'
    b'event: finish\ndata: {}\n\n'
)

# 模拟 tool_calls 响应 — 内嵌 JSON 需要双重转义
# 实际 DeepSeek API: fragment content 内的 JSON 会被 JSON-encode
# 所以外层 JSON 的 v 值中, " 会变成 \"
MOCK_TOOL_CALLS = (
    b'event: ready\ndata: {}\n\n'
    b'data: {"v":{"response":{"thinking_enabled":false,"fragments":[{"type":"ANSWER","content":"Let me check that."}]}}}\n\n'
    b'data: {"v":"<tool_calls>[{\\\"name\\\":\\\"get_weather\\\",\\\"arguments\\\":{\\\"city\\\":\\\"Beijing\\\"}}]</tool_calls>"}\n\n'
    b'data: {"p":"response/status","o":"APPEND","v":"FINISHED"}\n\n'
    b'event: finish\ndata: {}\n\n'
)


async def test_sse_events():
    print("=== Layer 1: SSE Events ===")
    async def byte_stream():
        yield MOCK_SSE

    count = 0
    async for evt in parse_sse_stream(byte_stream()):
        print(f"  event={evt.event!r}, data={evt.data[:80]}")
        count += 1
    assert count >= 6, f"Expected >= 6 events, got {count}"
    print(f"  \u2713 {count} events parsed")


async def test_dsframes():
    print("\n=== Layer 2: DsFrames ===")
    async def byte_stream():
        yield MOCK_SSE

    frames = []
    async for frame in parse_dsframe_stream(parse_sse_stream(byte_stream())):
        print(f"  {frame.type.value}: {str(frame.value)[:60]}")
        frames.append(frame)
    assert len(frames) >= 4, f"Expected >= 4 frames, got {len(frames)}"
    print(f"  \u2713 {len(frames)} frames parsed")


async def test_full_pipeline():
    print("\n=== Layer 3: Full Pipeline (OpenAI Chunks) ===")
    async def byte_stream():
        yield MOCK_SSE

    chunks = []
    async for chunk in full_sse_pipeline(byte_stream(), model="test"):
        choice = chunk.get("choices", [{}])[0]
        delta = choice.get("delta", {})
        finish = choice.get("finish_reason")
        if delta.get("role"):
            print(f"  role: {delta['role']}")
        if delta.get("content"):
            print(f"  content: {delta['content']}")
        if delta.get("reasoning_content"):
            print(f"  reasoning: {delta['reasoning_content'][:50]}")
        if finish:
            print(f"  finish_reason: {finish}")
        chunks.append(chunk)
    assert len(chunks) >= 4, f"Expected >= 4 chunks, got {len(chunks)}"
    print(f"  \u2713 {len(chunks)} chunks produced")


async def test_tool_calls():
    print("\n=== Tool Call Detection ===")
    async def byte_stream():
        yield MOCK_TOOL_CALLS

    chunks = []
    tool_call_chunks = []
    content_chunks = []
    async for chunk in full_sse_pipeline(byte_stream(), model="test"):
        choice = chunk.get("choices", [{}])[0]
        delta = choice.get("delta", {})
        if delta.get("tool_calls"):
            tool_call_chunks.append(delta["tool_calls"])
            print(f"  tool_calls: {delta['tool_calls']}")
        if delta.get("content"):
            content_chunks.append(delta["content"])
            print(f"  content: {delta['content']}")
        chunks.append(chunk)

    assert len(tool_call_chunks) >= 1, f"Expected tool_calls chunk, got {len(tool_call_chunks)}"
    assert len(content_chunks) >= 1, f"Expected content chunks, got {len(content_chunks)}"
    # 验证 tool call 内容
    tc = tool_call_chunks[0][0]
    assert tc["function"]["name"] == "get_weather", f"Wrong function name: {tc['function']['name']}"
    print(f"  \u2713 Tool call correctly parsed: {tc['function']['name']}")


# 模拟真实 DeepSeek 流: <tool_calls> 标签与 JSON 被拆分到多个 content delta
MOCK_TOOL_CALLS_SPLIT = (
    b"event: ready\ndata: {}\n\n"
    + _sse_event({"v": {"response": {
        "thinking_enabled": False,
        "fragments": [{"type": "RESPONSE", "content": "Let me check. <"}],
    }}})
    + _sse_event({"p": "response/fragments/-1/content", "o": "APPEND", "v": "tool"})
    + _sse_event({"v": "_c"})
    + _sse_event({"v": "alls"})
    + _sse_event({"v": ">\n"})
    + _sse_event({"v": "["})
    + _sse_event({"v": "{\"name\":\"get_weather\",\"arguments\":"})
    + _sse_event({"v": "{\"city\":\"Beijing\"}}"})
    + _sse_event({"v": "]\n"})
    + _sse_event({"v": "</"})
    + _sse_event({"v": "tool"})
    + _sse_event({"v": "_c"})
    + _sse_event({"v": "alls"})
    + _sse_event({"v": ">"})
    + _sse_event({"p": "response/status", "o": "SET", "v": "FINISHED"})
    + b"event: finish\ndata: {}\n\n"
)


async def test_tool_calls_split_chunks():
    print("\n=== Tool Call 跨 chunk 拆分 ===")
    async def byte_stream():
        yield MOCK_TOOL_CALLS_SPLIT

    content = ""
    tool_calls = []
    finish_reasons = []
    async for chunk in full_sse_pipeline(byte_stream(), model="test"):
        choice = chunk.get("choices", [{}])[0]
        delta = choice.get("delta", {})
        content += delta.get("content") or ""
        tool_calls.extend(delta.get("tool_calls") or [])
        if choice.get("finish_reason"):
            finish_reasons.append(choice["finish_reason"])

    assert content == "Let me check. ", f"content 应为提示文字，实际: {content!r}"
    assert len(tool_calls) == 1, f"Expected 1 tool call, got {len(tool_calls)}"
    tc = tool_calls[0]
    assert tc["function"]["name"] == "get_weather", f"Wrong function: {tc}"
    assert json.loads(tc["function"]["arguments"]) == {"city": "Beijing"}, f"Wrong arguments: {tc}"
    assert finish_reasons[-1] == "tool_calls", f"finish_reason 应为 tool_calls，实际: {finish_reasons}"
    assert "<tool_calls>" not in content, "XML 标签不应透传到 content"
    print(f"  content: {content!r}")
    print(f"  tool call: {tc['function']['name']}({tc['function']['arguments']})")
    print(f"  finish_reason: {finish_reasons[-1]}")
    print("  \u2713 跨 chunk 的 <tool_calls> 正确识别")


# 模拟思考模式真实流: 先 THINK 片段，运行中才 APPEND RESPONSE 片段
MOCK_THINK_RUNTIME_APPEND = (
    b"event: ready\ndata: {}\n\n"
    + _sse_event({"v": {"response": {
        "thinking_enabled": True,
        "fragments": [{"type": "THINK", "content": "思考A"}],
    }}})
    + _sse_event({"p": "response/fragments/-1/content", "o": "APPEND", "v": "思考B"})
    + _sse_event({"p": "response/fragments/-1/elapsed_secs", "o": "SET", "v": 0.9})
    + _sse_event({"p": "response/fragments", "o": "APPEND",
                  "v": [{"type": "RESPONSE", "content": "回答A"}]})
    + _sse_event({"p": "response/fragments/-1/content", "v": "回答B"})
    + _sse_event({"v": "回答C"})
    + _sse_event({"p": "response/status", "o": "SET", "v": "FINISHED"})
    + b"event: finish\ndata: {}\n\n"
)


async def test_think_runtime_append():
    print("\n=== 思考模式: RESPONSE 片段运行中追加 ===")
    async def byte_stream():
        yield MOCK_THINK_RUNTIME_APPEND

    reasoning = []
    content = []
    async for chunk in full_sse_pipeline(byte_stream(), model="deepseek-v4-pro"):
        delta = chunk.get("choices", [{}])[0].get("delta", {})
        if delta.get("reasoning_content"):
            reasoning.append(delta["reasoning_content"])
        if delta.get("content"):
            content.append(delta["content"])

    assert "".join(reasoning) == "思考A思考B", f"reasoning 错误: {reasoning}"
    assert "".join(content) == "回答A回答B回答C", f"content 错误: {content}"
    print(f"  reasoning: {''.join(reasoning)!r}")
    print(f"  content:   {''.join(content)!r}")
    print("  \u2713 RESPONSE 片段追加后正文字段正确输出")


# 模拟 DeepSeek 原生 DSML 工具调用（Pi 等客户端会触发）
MOCK_DSML_TOOL_CALL = (
    b"event: ready\ndata: {}\n\n"
    + _sse_event({"v": {"response": {
        "thinking_enabled": True,
        "fragments": [{"type": "THINK", "content": "需要列目录"}],
    }}})
    + _sse_event({"p": "response/fragments", "o": "APPEND",
                  "v": [{"type": "RESPONSE", "content": "<"}]})
    + _sse_event({"v": "｜｜DSML｜｜"})
    + _sse_event({"v": " calls>\n<"})
    + _sse_event({"v": "｜｜DSML｜｜"})
    + _sse_event({"v": " invoke name=\"bash\">\n"})
    + _sse_event({"v": "<｜｜DSML｜｜ parameter name=\"command\" string=\"true\">ls"})
    + _sse_event({"v": " -la</｜｜DSML｜｜ parameter>\n"})
    + _sse_event({"v": "</｜｜DSML｜｜ invoke>\n"})
    + _sse_event({"v": "</｜｜DSML｜｜ calls>"})
    + _sse_event({"p": "response/status", "o": "SET", "v": "FINISHED"})
    + b"event: finish\ndata: {}\n\n"
)


async def test_dsml_tool_calls():
    print("\n=== DeepSeek 原生 DSML 工具调用 ===")
    async def byte_stream():
        yield MOCK_DSML_TOOL_CALL

    content = ""
    calls = []
    finish = None
    async for chunk in full_sse_pipeline(byte_stream(), model="deepseek-v4-pro"):
        ch = chunk.get("choices", [{}])[0]
        delta = ch.get("delta", {})
        content += delta.get("content") or ""
        calls.extend(delta.get("tool_calls") or [])
        finish = ch.get("finish_reason") or finish

    assert content == "", f"DSML 不应泄漏到 content: {content!r}"
    assert len(calls) == 1, f"Expected 1 tool call, got {len(calls)}"
    assert calls[0]["function"]["name"] == "bash"
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "ls -la"}
    assert finish == "tool_calls"
    print(f"  tool call: {calls[0]['function']['name']}({calls[0]['function']['arguments']})")
    print(f"  finish_reason: {finish}")
    print("  \u2713 DSML 工具调用正确转换")


# 模拟 DeepSeek 上游 generation_err: quasi_status/status = INCOMPLETE
MOCK_INCOMPLETE = (
    b"event: ready\ndata: {}\n\n"
    + _sse_event({"v": {"response": {
        "thinking_enabled": True,
        "fragments": [{"type": "RESPONSE", "content": "hello"}],
    }}})
    + _sse_event({"p": "response/has_pending_fragment", "v": True})
    + _sse_event({"v": False})
    + _sse_event({"p": "response", "o": "BATCH", "v": [
        {"p": "accumulated_token_usage", "v": 100},
        {"p": "quasi_status", "v": "INCOMPLETE"},
    ]})
    + _sse_event({"p": "response/status", "o": "SET", "v": "INCOMPLETE"})
    + b'event: hint\ndata: {"type":"error","content":"Server is temporarily unavailable.",'
        b'"finish_reason":"generation_err"}\n\n'
)


async def test_incomplete_status_not_content():
    print("\n=== INCOMPLETE 状态不应作为正文 ===")
    async def byte_stream():
        yield MOCK_INCOMPLETE

    content = ""
    async for chunk in full_sse_pipeline(byte_stream(), model="deepseek-v4-pro"):
        delta = chunk.get("choices", [{}])[0].get("delta", {})
        content += delta.get("content") or ""

    assert content == "hello", f"content 错误: {content!r}"
    assert "INCOMPLETE" not in content
    print(f"  content: {content!r}")
    print("  \u2713 quasi_status/status 控制字段未混入正文")


async def main():
    await test_sse_events()
    print("---")
    await test_dsframes()
    print("---")
    await test_full_pipeline()
    print("---")
    await test_tool_calls()
    await test_tool_calls_split_chunks()
    await test_think_answer_split()
    await test_think_runtime_append()
    await test_dsml_tool_calls()
    await test_incomplete_status_not_content()
    print("\n\u2705 All tests passed!")


# 模拟 V4-Pro 推理响应 (THINK/ANSWER 分流)
MOCK_THINK = (
    b'event: ready\ndata: {}\n\n'
    b'data: {"v":{"response":{"thinking_enabled":true,"fragments":['
    b'{"type":"THINK","content":"Let me reason about the sky."},'
    b'{"type":"ANSWER","content":"The sky is blue."}]}}}\n\n'
    b'data: {"p":"response/fragments/-1/content","o":"APPEND","v":" Rayleigh scattering."}\n\n'
    b'data: {"p":"response/fragments/0/content","o":"APPEND","v":" Blue light scatters."}\n\n'
    b'data: {"p":"response/status","o":"APPEND","v":"FINISHED"}\n\n'
    b'event: finish\ndata: {}\n\n'
)


async def test_think_answer_split():
    print("\n=== THINK/ANSWER 分流 ===")
    async def byte_stream():
        yield MOCK_THINK

    reasoning = []
    content = []
    async for chunk in full_sse_pipeline(byte_stream(), model="deepseek-v4-pro"):
        delta = chunk.get("choices", [{}])[0].get("delta", {})
        if delta.get("reasoning_content"):
            reasoning.append(delta["reasoning_content"])
        if delta.get("content"):
            content.append(delta["content"])

    assert reasoning == ["Let me reason about the sky.", " Blue light scatters."], \
        f"reasoning 分流错误: {reasoning}"
    assert content == ["The sky is blue.", " Rayleigh scattering."], \
        f"content 分流错误: {content}"
    print(f"  reasoning: {reasoning}")
    print(f"  content:   {content}")
    print("  \u2713 推理内容进 reasoning_content, 正文进 content")

if __name__ == "__main__":
    asyncio.run(main())
