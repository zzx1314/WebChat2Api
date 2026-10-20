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


# 模拟真实失败样本 (Pi 日志): 模型把工具调用写成正文，且参数里的双引号、`\|` 未转义
UNESCAPED_TOOL_CALLS_TEXT = (
    'Now let me verify remaining discrepancies:\n\n'
    '<tool_calls>[{"name": "bash", "arguments": {"command": '
    '"cd /repo && echo "=== A ===" && grep -n "root_password\\|insert" f; '
    'echo "=== B ===" && sed -n \'1,5p\' g"}}, '
    '{"name": "bash", "arguments": {"command": "echo hi"}}]</tool_calls>'
)


def _mock_from_text(text: str, pieces: int = 4):
    """把正文拆成多个 content delta，模拟跨 chunk 输出"""
    size = max(1, len(text) // pieces + 1)
    parts = [text[i:i + size] for i in range(0, len(text), size)]
    events = b"event: ready\ndata: {}\n\n" + _sse_event({"v": {"response": {
        "thinking_enabled": True,
        "fragments": [{"type": "RESPONSE", "content": pieces and parts[0]}],
    }}})
    for part in parts[1:]:
        events += _sse_event({"p": "response/fragments/-1/content", "o": "APPEND", "v": part})
    events += _sse_event({"p": "response/status", "o": "SET", "v": "FINISHED"})
    events += b"event: finish\ndata: {}\n\n"
    return events


async def test_unescaped_tool_calls_json():
    print("\n=== 参数 JSON 转义损坏的工具调用 (真实 Pi 日志样本) ===")
    async def byte_stream():
        yield _mock_from_text(UNESCAPED_TOOL_CALLS_TEXT)

    content = ""
    calls = []
    finish = None
    async for chunk in full_sse_pipeline(byte_stream(), model="deepseek-v4-pro"):
        ch = chunk.get("choices", [{}])[0]
        delta = ch.get("delta", {})
        content += delta.get("content") or ""
        calls.extend(delta.get("tool_calls") or [])
        finish = ch.get("finish_reason") or finish

    assert "<tool_calls>" not in content, f"工具调用泄漏到 content: {content!r}"
    assert content == "Now let me verify remaining discrepancies:\n\n", f"正文错误: {content!r}"
    assert len(calls) == 2, f"Expected 2 tool calls, got {len(calls)}: {calls}"
    assert finish == "tool_calls", f"finish_reason 应为 tool_calls，实际: {finish}"

    first = json.loads(calls[0]["function"]["arguments"])
    expected = (
        'cd /repo && echo "=== A ===" && grep -n "root_password\\|insert" f; '
        'echo "=== B ===" && sed -n \'1,5p\' g'
    )
    assert calls[0]["function"]["name"] == "bash"
    assert first["command"] == expected, f"命令被改坏:\n{first['command']!r}\n!= {expected!r}"
    assert json.loads(calls[1]["function"]["arguments"]) == {"command": "echo hi"}
    print(f"  tool call 1: {first['command']}")
    print(f"  finish_reason: {finish}")
    print("  \u2713 未转义引号/非法转义修复后正确转为 tool_calls")


# 模拟真实失败样本 (Pi 日志 2026-10-20): V4-Pro 把工具调用写进 THINK 片段
THINK_TOOL_CALLS_TEXT = (
    '我先查一下 chat.qwen.ai 是否还会重定向。\n\n'
    '<tool_calls>[{"name": "bash", "arguments": {"command": '
    '"curl -sS -m 30 -o /dev/null -w "%{http_code}" https://chat.qwen.ai/"}}, '
    '{"name": "bash", "arguments": {"command": "cat qw_pre.js < qw_head.txt"}}]</tool_calls>'
)


def _mock_think_from_text(text: str, pieces: int = 8):
    """把思考内容拆成多个 THINK delta，模拟跨 chunk 输出"""
    size = max(1, len(text) // pieces + 1)
    parts = [text[i:i + size] for i in range(0, len(text), size)]
    events = b"event: ready\ndata: {}\n\n" + _sse_event({"v": {"response": {
        "thinking_enabled": True,
        "fragments": [{"type": "THINK", "content": parts[0]}],
    }}})
    for part in parts[1:]:
        events += _sse_event({"p": "response/fragments/-1/content", "o": "APPEND", "v": part})
    events += _sse_event({"p": "response/status", "o": "SET", "v": "FINISHED"})
    events += b"event: finish\ndata: {}\n\n"
    return events


async def test_tool_calls_in_thinking():
    print("\n=== 思考通道里的工具调用 (真实 Pi 日志样本) ===")
    async def byte_stream():
        yield _mock_think_from_text(THINK_TOOL_CALLS_TEXT)

    reasoning = ""
    content = ""
    calls = []
    finish = None
    async for chunk in full_sse_pipeline(byte_stream(), model="deepseek-v4-pro"):
        ch = chunk.get("choices", [{}])[0]
        delta = ch.get("delta", {})
        reasoning += delta.get("reasoning_content") or ""
        content += delta.get("content") or ""
        calls.extend(delta.get("tool_calls") or [])
        finish = ch.get("finish_reason") or finish

    assert "<tool_calls>" not in reasoning, f"工具调用泄漏到 reasoning: {reasoning!r}"
    assert reasoning == "我先查一下 chat.qwen.ai 是否还会重定向。\n\n", f"思考内容错误: {reasoning!r}"
    assert content == "", f"思考片段不应进入 content: {content!r}"
    assert len(calls) == 2, f"Expected 2 tool calls, got {len(calls)}: {calls}"
    assert finish == "tool_calls", f"finish_reason 应为 tool_calls，实际: {finish}"
    assert json.loads(calls[0]["function"]["arguments"])["command"] == (
        'curl -sS -m 30 -o /dev/null -w "%{http_code}" https://chat.qwen.ai/'
    ), calls[0]
    assert json.loads(calls[1]["function"]["arguments"])["command"] == "cat qw_pre.js < qw_head.txt"
    print(f"  reasoning: {reasoning!r}")
    print(f"  tool calls: {[c['function']['name'] for c in calls]}")
    print(f"  finish_reason: {finish}")
    print("  \u2713 思考通道内的 <tool_calls> 正确转成 tool_calls")


def _aggregate_by_index(tool_calls: list[dict]) -> dict[int, dict]:
    """模拟 Pi / OpenAI SDK 等客户端: 按 index 拼接 arguments"""
    merged: dict[int, dict] = {}
    for tc in tool_calls:
        slot = merged.setdefault(tc["index"], {"name": tc["function"]["name"], "arguments": ""})
        slot["arguments"] += tc["function"]["arguments"]
    return merged


# 模型可能把一次响应拆成多个 <tool_calls> 块
MULTI_BLOCK_TOOL_CALLS_TEXT = (
    'first <tool_calls>[{"name": "bash", "arguments": {"command": "echo A"}}, '
    '{"name": "bash", "arguments": {"command": "echo B"}}]</tool_calls>\n'
    'second <tool_calls>[{"name": "bash", "arguments": {"command": "echo C"}}]</tool_calls>'
)


async def test_multiple_tool_call_blocks():
    print("\n=== 多个 <tool_calls> 块: index 必须全局唯一 ===")
    async def byte_stream():
        yield _mock_from_text(MULTI_BLOCK_TOOL_CALLS_TEXT)

    calls = []
    finish = None
    async for chunk in full_sse_pipeline(byte_stream(), model="deepseek-v4-pro"):
        ch = chunk.get("choices", [{}])[0]
        calls.extend(ch.get("delta", {}).get("tool_calls") or [])
        finish = ch.get("finish_reason") or finish

    assert [c["index"] for c in calls] == [0, 1, 2], f"index 应递增: {[c['index'] for c in calls]}"
    merged = _aggregate_by_index(calls)
    assert len(merged) == 3, f"客户端会合并成 {len(merged)} 个调用: {merged}"
    for slot in merged.values():
        json.loads(slot["arguments"])  # 拼接后的参数必须是合法 JSON
    assert [json.loads(merged[i]["arguments"])["command"] for i in range(3)] == [
        "echo A", "echo B", "echo C",
    ]
    assert finish == "tool_calls"
    print(f"  indices: {[c['index'] for c in calls]}")
    print("  \u2713 多个调用块不会在客户端被拼成一个非法 JSON")


MOCK_TOOL_CALLS_TWO_CHANNELS = (
    b"event: ready\ndata: {}\n\n"
    + _sse_event({"v": {"response": {
        "thinking_enabled": True,
        "fragments": [{
            "type": "THINK",
            "content": 'plan <tool_calls>[{"name": "bash", "arguments": {"command": "echo think"}}]</tool_calls>',
        }],
    }}})
    + _sse_event({"p": "response/fragments", "o": "APPEND", "v": [{
        "type": "RESPONSE",
        "content": 'done <tool_calls>[{"name": "bash", "arguments": {"command": "echo answer"}}]</tool_calls>',
    }]})
    + _sse_event({"p": "response/status", "o": "SET", "v": "FINISHED"})
    + b"event: finish\ndata: {}\n\n"
)


async def test_tool_calls_across_channels():
    print("\n=== THINK / ANSWER 各一个调用: index 跨通道递增 ===")
    async def byte_stream():
        yield MOCK_TOOL_CALLS_TWO_CHANNELS

    calls = []
    async for chunk in full_sse_pipeline(byte_stream(), model="deepseek-v4-pro"):
        calls.extend(chunk.get("choices", [{}])[0].get("delta", {}).get("tool_calls") or [])

    assert [c["index"] for c in calls] == [0, 1], f"index 应跨通道递增: {[c['index'] for c in calls]}"
    merged = _aggregate_by_index(calls)
    assert [json.loads(merged[i]["arguments"])["command"] for i in range(2)] == [
        "echo think", "echo answer",
    ]
    print(f"  indices: {[c['index'] for c in calls]}")
    print("  \u2713 思考通道与正文通道的调用各自独立")


async def test_non_stream_with_usage_chunk():
    print("\n=== 非流式请求: usage chunk 不应导致 500 ===")
    from deepseek_proxy.server import _handle_non_stream

    data = MOCK_TOOL_CALLS + _sse_event({"p": "response/usage/total_token_usage", "o": "SET", "v": 42})

    class FakeResp:
        _session_id = None

        async def aiter_bytes(self):
            yield data

    result = await _handle_non_stream(FakeResp(), "deepseek-v4-flash")
    choice = result["choices"][0]
    assert choice["finish_reason"] == "tool_calls", choice
    assert len(choice["message"]["tool_calls"]) == 1, choice
    assert "index" not in choice["message"]["tool_calls"][0]
    assert result["usage"]["total_tokens"] == 42, result["usage"]
    print(f"  usage: {result['usage']}")
    print("  \u2713 usage chunk 被正确提取，工具调用完整返回")


async def test_tool_call_argument_variants():
    print("\n=== 工具调用字段变体 ===")
    from deepseek_proxy.sse_parser import _parse_tool_calls_json

    # arguments 以 JSON 字符串给出
    parsed = _parse_tool_calls_json('[{"name":"bash","arguments":"{\\"command\\":\\"ls\\"}"}]')
    assert parsed == [{"name": "bash", "arguments": {"command": "ls"}}], parsed

    # markdown 代码块 + parameters 别名
    parsed = _parse_tool_calls_json('```json\n[{"name":"read","parameters":{"path":"a"}}]\n```')
    assert parsed == [{"name": "read", "arguments": {"path": "a"}}], parsed

    # 结构损坏 (括号未闭合) 时仍应识别出调用，而不是退化成正文
    parsed = _parse_tool_calls_json('[{"name": "bash", "arguments": {"command": "echo "a b"')
    assert len(parsed) == 1 and parsed[0]["name"] == "bash", parsed
    print("  \u2713 字符串参数 / 代码块 / 括号损坏均能解析")


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
    await test_unescaped_tool_calls_json()
    await test_tool_calls_in_thinking()
    await test_multiple_tool_call_blocks()
    await test_tool_calls_across_channels()
    await test_non_stream_with_usage_chunk()
    await test_tool_call_argument_variants()
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
