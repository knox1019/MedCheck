import logging
from app.core.config import get_settings
def setup_logging() -> None:
    settings = get_settings()

    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True
    )