from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings,SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    app_env:str = "dev"
    port:int = 8000
    log_level:str = "INFO"
    app_version:str = "0.1.0"


@lru_cache
def get_settings() -> Settings:
    return Settings()