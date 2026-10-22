"""配置模块：把外部环境变量变成代码里可用的配置对象。

核心思想是「配置与代码分离」——同一份代码，靠外部注入变成不同环境的程序，
不需要为测试环境和生产环境各维护一个分支。

取值优先级（高到低）：
    真实环境变量  >  .env 文件  >  字段默认值

所以线上永远可以用环境变量覆盖本地 .env，这也是容器化部署的基础。

这个文件只负责「读取配置」，不负责「使用配置」。
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录。
# __file__ 是当前文件的路径 app/core/config.py，逐层往上退：
#     parents[0] = app/core
#     parents[1] = app
#     parents[2] = 项目根
# 这里刻意用绝对路径而不是 ".env" 这样的相对路径：
# 相对路径是相对「当前工作目录」的，换个目录启动 uvicorn 或跑 pytest 就读不到了。
BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """应用配置。

    继承 BaseSettings 是全部魔法的来源：它会让声明的字段自动去
    环境变量和 .env 文件里找对应的值，并按类型注解做转换和校验。
    """

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",  # .env 文件的位置
        env_file_encoding="utf-8",  # 不指定编码，Windows 上中文会乱码
        extra="ignore",  # .env 里存在本项目用不到的键时不报错
    )

    # 字段名用小写下划线即可，会自动匹配大写形式的环境变量（app_env ← APP_ENV）。
    # 语法：「字段名: 类型 = 默认值」，类型不对会报错，找不到值就用默认值。
    # port 声明成 int，所以普通环境变量传进来的字符串 "8000" 会自动转成整数。
    app_env: str = "dev"
    port: int = 8000
    log_level: str = "INFO"
    app_version: str = "0.1.0"

    # ---- Embedding ----
    # 换后端只改这三个值，业务代码一行不用动。
    # Ollama 和云端 API 都是 OpenAI 兼容协议，所以写法完全一致。
    embedding_base_url: str = "http://localhost:11434/v1"
    embedding_model: str = "bge-m3"
    # Ollama 不校验这个值，但客户端要求非空，填任意字符串都可以
    embedding_api_key: str = "ollama"

    # ---- 向量库 ----
    # 用绝对路径，避免换个目录执行命令就把库建到别处去
    chroma_dir: str = str(BASE_DIR / "data" / "chroma")

    # ---- 重排 ----
    # 本地 ONNX 模型（int8 量化，约 0.57GB）。fp32 版本是 2.27GB，检索质量差别很小。
    rerank_model: str = "BAAI/bge-reranker-v2-m3-int8"
    # 模型从 HuggingFace 下载，国内直连不通，默认走镜像
    hf_endpoint: str = "https://hf-mirror.com"


@lru_cache
def get_settings() -> Settings:
    """返回全局唯一的配置对象。

    @lru_cache 会缓存第一次的返回值，之后每次调用直接返回同一个对象，
    既避免重复读文件，也保证各模块拿到的配置是一致的。

    代价：测试里修改环境变量后必须调用 get_settings.cache_clear()，
    否则读到的还是上一次缓存的结果。
    """
    return Settings()
