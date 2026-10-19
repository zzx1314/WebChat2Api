"""
DeepSeek Proxy — 入口

Usage:
    python -m deepseek_proxy.main
    # 或
    python main.py
"""

import logging
import asyncio

from .config import CONFIG, DOTENV_PATH
from .client import ClientError
from .server import init_server, shutdown_server, create_app


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # 降低 httpx 日志
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def validate_config():
    """启动前校验必填配置，避免直接抛出 traceback"""
    if CONFIG.auth_mode.value == "token" and not CONFIG.user_token:
        raise SystemExit(
            "配置错误: TOKEN 模式下未设置 DS_TOKEN。\n"
            "请先 export DS_TOKEN=<你的 Bearer Token> 后再启动。"
        )
    if CONFIG.auth_mode.value == "password":
        if not (CONFIG.account_email or CONFIG.account_mobile):
            raise SystemExit("配置错误: PASSWORD 模式下需要设置 DS_EMAIL 或 DS_MOBILE。")
        if not CONFIG.account_password:
            raise SystemExit("配置错误: PASSWORD 模式下需要设置 DS_PASSWORD。")


async def main():
    setup_logging()
    logger = logging.getLogger("deepseek_proxy")

    validate_config()
    if DOTENV_PATH:
        logger.info("已加载环境变量文件: %s", DOTENV_PATH)
    logger.info("启动 DeepSeek Proxy...")
    logger.info("认证模式: %s", CONFIG.auth_mode.value)
    logger.info("会话模式: %s", CONFIG.session_mode.value)
    logger.info("服务器: %s:%d", CONFIG.server_host, CONFIG.server_port)

    # 初始化
    try:
        state = await init_server(CONFIG)
    except ClientError as e:
        logger.error("初始化失败: %s", e)
        raise SystemExit(1)

    # 创建 FastAPI app
    app = create_app(CONFIG)

    # 启动 uvicorn
    import uvicorn
    config = uvicorn.Config(
        app,
        host=CONFIG.server_host,
        port=CONFIG.server_port,
        log_level="info",
    )
    server = uvicorn.Server(config)

    try:
        await server.serve()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("收到关闭信号...")
    finally:
        await shutdown_server()
        logger.info("DeepSeek Proxy 已关闭")


if __name__ == "__main__":
    asyncio.run(main())


def cli():
    """命令行入口 (供 console_scripts 使用)"""
    asyncio.run(main())