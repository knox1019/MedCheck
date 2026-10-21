import logging
from fastapi import FastAPI

from app.core.config import get_settings
from app.core.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
settings = get_settings()

logger.info(
    "MedCheck 启动: env=%s version=%s port=%s",
    settings.app_env,
    settings.app_version,
    settings.port,
)

app = FastAPI(title="MedCheck", version=settings.app_version)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "env": settings.app_env,
        "version": settings.app_version,
    }
