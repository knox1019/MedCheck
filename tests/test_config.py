from app.core.config import Settings, get_settings


def test_defaults(monkeypatch):
    for key in ("APP_ENV", "PORT", "LOG_LEVEL", "APP_VERSION"):
        monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()

    settings = Settings(_env_file=None)

    assert settings.app_env == "dev"
    assert settings.port == 8000
    get_settings.cache_clear()


def test_env_override(monkeypatch):
    monkeypatch.setenv("APP_VERSION", "9.9.9")
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.app_version == "9.9.9"
    get_settings.cache_clear()
