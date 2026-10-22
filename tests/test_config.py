"""配置模块的单元测试。

测试的价值不是「证明当前代码是对的」，而是「以后改代码时能立刻发现改坏了没有」。

这里测两件事：
    1. 没有任何外部配置时，字段默认值是否正确
    2. 环境变量能否覆盖默认值（也就是优先级链是否真的生效）
"""

from app.core.config import Settings, get_settings


def test_defaults(monkeypatch):
    """验证字段默认值：不提供任何外部配置时，应该是 dev / 8000。"""
    # 先清掉可能存在的系统环境变量，否则开发机上恰好设过就会导致误判。
    # monkeypatch 做的修改会在测试结束后自动还原，不会污染后面的用例；
    # raising=False 表示这个变量本来就不存在时也不要报错。
    for key in ("APP_ENV", "PORT", "LOG_LEVEL", "APP_VERSION"):
        monkeypatch.delenv(key, raising=False)

    # 清掉 @lru_cache 的缓存，否则拿到的是上一个测试留下的旧对象
    get_settings.cache_clear()

    # _env_file=None 表示「假装项目里没有 .env 文件」。
    # 不这么做的话读到的会是 .env 里的真实值，就测不到默认值了。
    settings = Settings(_env_file=None)

    assert settings.app_env == "dev"
    assert settings.port == 8000

    # 收尾再清一次，避免本用例的缓存影响后面的测试
    get_settings.cache_clear()


def test_env_override(monkeypatch):
    """验证环境变量优先级高于默认值。"""
    monkeypatch.setenv("APP_VERSION", "9.9.9")
    get_settings.cache_clear()

    settings = get_settings()

    # 环境变量里的 9.9.9 应该盖掉默认的 0.1.0
    assert settings.app_version == "9.9.9"

    get_settings.cache_clear()
