"""日志配置模块。

只提供一个 setup_logging()，在程序入口调用一次即可。
单独拆成模块的原因：日志是所有模块的公共依赖，
如果写在各处，格式会不一致，也没法统一调级别。
"""

import logging

from app.core.config import get_settings


def setup_logging() -> None:
    """配置全局日志。整个程序只应该调用一次。

    为什么不用 print：
        print 没有级别（没法一键关掉调试信息）、没有时间戳、
        不知道是哪一行打的，也不能同时输出到文件。
    """
    settings = get_settings()

    logging.basicConfig(
        # 级别从配置读，线上要临时调成 DEBUG 时改环境变量即可，不用改代码重发版
        level=settings.log_level.upper(),
        # 格式里的占位符：
        #   %(asctime)s    时间
        #   %(levelname)   级别；-8 表示左对齐并补到 8 个字符宽，让这一列整齐
        #   %(name)s       打日志的模块名（即 getLogger(__name__) 传进去的名字）
        #   %(message)s    日志正文
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        # 不指定的话默认时间会带一串毫秒，很难看
        datefmt="%Y-%m-%d %H:%M:%S",
        # 先清空已存在的 handler 再配置。
        # 不加这个参数，热重载或重复调用时同一条日志会打两遍。
        force=True,
    )
