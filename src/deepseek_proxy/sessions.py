"""
会话管理 — 兼容两种模式: 复用 (REUSE) 和 新建 (NEW)

对应:
- ds-free-api: ds_core/accounts.rs (session 复用 + health_check)
- Chat2API: src/main/proxy/adapters/deepseek.ts (每次 createSession)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from .config import ProxyConfig, SessionMode
from .client import DsClient, CompletionPayload, EditMessagePayload, ClientError
from .pow_solver import PowSolver

logger = logging.getLogger(__name__)

# DeepSeek 上游限流时的重试策略
_RATE_LIMIT_RETRIES = 3
_RATE_LIMIT_DELAYS = (3, 8, 15)
_RETRYABLE_HINTS = {"rate_limit_reached", "edit_rate_limit", "generation_err", "server_error"}


class _PrefixedResponse:
    """包装 response，先吐出已探测的字节，再继续原有流"""

    def __init__(self, resp: Any, iterator: Any, prefix: bytes, session_id: str):
        self._resp = resp
        self._iterator = iterator
        self._prefix = prefix
        self._session_id = session_id

    async def aiter_bytes(self):
        if self._prefix:
            prefix, self._prefix = self._prefix, b""
            yield prefix
        async for chunk in self._iterator:
            yield chunk

    async def aclose(self) -> None:
        await self._resp.aclose()


async def _peek_upstream(resp: Any) -> tuple[bytes, str | None, str | None, Any]:
    """读取流开头，检测 DeepSeek 的 hint 错误（限流等）。

    返回 (已读前缀, 错误信息, finish_reason, 迭代器)。正常响应不会丢失字节。
    """
    iterator = resp.aiter_bytes()
    buf = b""
    max_peek = 16 * 1024

    def parse_hint(text: str) -> tuple[str, str] | None:
        match = re.search(
            r"event:\s*hint\s*\ndata:\s*(\{.*?\})\s*\n\n",
            text,
            re.DOTALL,
        )
        if not match:
            return None
        try:
            hint = json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
        if hint.get("type") != "error":
            return None
        message = hint.get("content") or "upstream error"
        return message, hint.get("finish_reason") or ""

    while len(buf) < max_peek:
        try:
            chunk = await anext(iterator)
        except StopAsyncIteration:
            break
        buf += chunk
        text = buf.decode("utf-8", "ignore")

        hint = parse_hint(text)
        if hint:
            return buf, hint[0], hint[1], iterator

        # 已出现正常响应内容，无需继续探测
        if re.search(r'data:\s*\{"v":', text) or "event: finish" in text:
            return buf, None, None, iterator

    hint = parse_hint(buf.decode("utf-8", "ignore"))
    if hint:
        return buf, hint[0], hint[1], iterator
    return buf, None, None, iterator


class SessionStrategy:
    """会话策略抽象"""

    def __init__(self, config: ProxyConfig, client: DsClient, solver: PowSolver,
                 token: str):
        self.config = config
        self.client = client
        self.solver = solver
        self.token = token

    async def prepare_pow(self, target_path: str = "/api/v0/chat/completion") -> str:
        """获取 PoW 挑战并求解, 返回 X-Ds-Pow-Response header 值"""
        challenge = await self.client.create_pow_challenge(self.token, target_path)
        result = self.solver.solve(challenge)
        return result.to_header()

    async def execute(self, prompt: str, thinking_enabled: bool = False,
                      search_enabled: bool = False, model_type: str = "default",
                      fresh_session: bool = False):
        raise NotImplementedError

    async def cleanup(self) -> None:
        pass


class ReuseSessionStrategy(SessionStrategy):
    """ds-free-api 模式: 初始化时创建 session + health_check, 永久复用

    会话绑定生命周期, 通过 edit_message 编辑后续消息。
    message_id 由 health_check 完成后动态跟踪。
    """

    def __init__(self, config: ProxyConfig, client: DsClient, solver: PowSolver,
                 token: str, model_types: list[str]):
        super().__init__(config, client, solver, token)
        self._sessions: dict[str, str] = {}       # model_type → session_id
        self._message_ids: dict[str, int] = {}    # model_type → next message_id

    async def init(self, model_types: list[str]) -> None:
        for mt in model_types:
            await self._init_model_type(mt)

    async def _init_model_type(self, model_type: str) -> None:
        # 1. 创建 session
        session_id = await self.client.create_session(self.token)

        # 2. health_check: 发送一条测试消息 (message 0)
        pow_header = await self.prepare_pow("/api/v0/chat/edit_message")
        payload = CompletionPayload(
            chat_session_id=session_id,
            prompt="只回复`Hello, world!`",
            thinking_enabled=False,
            search_enabled=False,
        )
        resp = await self.client.completion(self.token, pow_header, payload)
        async for _ in resp.aiter_bytes():
            pass

        self._sessions[model_type] = session_id
        # health_check 占用 message 0, 下一次可用 message_id = 1
        self._message_ids[model_type] = 1
        logger.info("Session initialized: model_type=%s, session=%s", model_type, session_id)

    async def execute(self, prompt: str, thinking_enabled: bool = False,
                      search_enabled: bool = False, model_type: str = "default",
                      fresh_session: bool = False):
        # 工具调用场景使用全新 session + completion，避免反复 edit_message
        # 触发 DeepSeek 的 "Editing/regeneration too frequently" 限制
        if fresh_session:
            fresh = NewSessionStrategy(self.config, self.client, self.solver, self.token)
            return await fresh.execute(prompt, thinking_enabled, search_enabled, model_type)

        session_id = self._sessions.get(model_type)
        if not session_id:
            raise RuntimeError(f"No session for model_type={model_type}. Call init() first.")

        msg_id = self._message_ids.get(model_type, 1)

        pow_header = await self.prepare_pow("/api/v0/chat/edit_message")
        payload = EditMessagePayload(
            chat_session_id=session_id,
            message_id=msg_id,
            prompt=prompt,
            search_enabled=search_enabled,
            thinking_enabled=thinking_enabled,
            model_type=model_type,
        )

        resp = await self.client.edit_message(self.token, pow_header, payload)
        return resp

    async def cleanup(self) -> None:
        for model_type, session_id in self._sessions.items():
            try:
                await self.client.delete_session(self.token, session_id)
            except Exception as e:
                logger.warning("Failed to delete session %s (model=%s): %s", session_id, model_type, e)
        self._sessions.clear()
        self._message_ids.clear()


class NewSessionStrategy(SessionStrategy):
    """Chat2API 模式: 每次请求创建新 session, 用 completion 端点"""

    async def execute(self, prompt: str, thinking_enabled: bool = False,
                      search_enabled: bool = False, model_type: str = "default",
                      fresh_session: bool = False):
        last_error = ""

        for attempt in range(1, _RATE_LIMIT_RETRIES + 1):
            session_id = await self.client.create_session(self.token)
            pow_header = await self.prepare_pow("/api/v0/chat/completion")
            payload = CompletionPayload(
                chat_session_id=session_id,
                prompt=prompt,
                thinking_enabled=thinking_enabled,
                search_enabled=search_enabled,
            )

            resp = await self.client.completion(self.token, pow_header, payload)
            prefix, error, finish_reason, iterator = await _peek_upstream(resp)

            if not error:
                return _PrefixedResponse(resp, iterator, prefix, session_id)

            last_error = error
            await resp.aclose()
            try:
                await self.client.delete_session(self.token, session_id)
            except Exception:
                pass

            retryable = (
                finish_reason in _RETRYABLE_HINTS
                or any(k in error.lower() for k in ("too frequent", "temporarily unavailable", "try again"))
            )
            if not retryable or attempt >= _RATE_LIMIT_RETRIES:
                raise ClientError(f"DeepSeek 上游错误: {error}")

            delay = _RATE_LIMIT_DELAYS[min(attempt - 1, len(_RATE_LIMIT_DELAYS) - 1)]
            logger.warning(
                "DeepSeek 上游限流 (%s)，%ds 后重试 (%d/%d)",
                error, delay, attempt + 1, _RATE_LIMIT_RETRIES,
            )
            await asyncio.sleep(delay)

        raise ClientError(f"DeepSeek 上游错误: {last_error or 'unknown'}")

    async def cleanup(self) -> None:
        pass


async def cleanup_session(client: DsClient, token: str, session_id: str) -> None:
    try:
        await client.delete_session(token, session_id)
    except Exception as e:
        logger.warning("Failed to cleanup session %s: %s", session_id, e)


def create_session_strategy(config: ProxyConfig, client: DsClient,
                            solver: PowSolver, token: str,
                            model_types: list[str]) -> SessionStrategy:
    if config.session_mode == SessionMode.REUSE:
        return ReuseSessionStrategy(config, client, solver, token, model_types)
    elif config.session_mode == SessionMode.NEW:
        return NewSessionStrategy(config, client, solver, token)
    else:
        raise ValueError(f"Unsupported session_mode: {config.session_mode}")
