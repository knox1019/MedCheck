"""应用入口。

启动链路的执行顺序（顺序本身是有意义的）：
    1. 先配置日志  —— 之后所有输出才有统一格式，配晚了前面的日志就没格式
    2. 再读取配置  —— 创建应用、打启动日志都要用到
    3. 最后创建 FastAPI 实例并登记路由

要分清两种代码：
    模块级代码（不在函数体内的）在 import 时执行一次，负责初始化。
    路由函数体只在收到请求时执行，写在那里的 def 是「待命」不是「立刻跑」。
"""

import logging

from fastapi import FastAPI

from app.core.config import get_settings
from app.core.logging import setup_logging

# 必须在任何 logger 使用之前调用，否则前面打的日志没有统一格式
setup_logging()

# __name__ 的值是当前模块名 "app.main"，它会出现在日志的 name 那一列
logger = logging.getLogger(__name__)

settings = get_settings()

# 用 %s 占位符而不是 f-string，不是因为好看：
# logging 会先判断级别，级别不够时连字符串拼接都省掉；
# 用 f-string 则无论是否会输出，字符串都会先被拼出来。
logger.info(
    "MedCheck 启动: env=%s version=%s port=%s",
    settings.app_env,
    settings.app_version,
    settings.port,
)

# version 从配置读取，这个值会显示在 /docs 自动生成的接口文档页面上
app = FastAPI(title="MedCheck", version=settings.app_version)


@app.get("/health")
def health() -> dict:
    """健康检查接口。

    容器编排和负载均衡靠它判断服务是否存活；返回的 env 和 version
    用于排查「线上跑的到底是哪个版本、哪个环境」。
    """
    return {
        "status": "ok",
        "env": settings.app_env,
        "version": settings.app_version,
    }
