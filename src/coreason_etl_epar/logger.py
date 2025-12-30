import sys

from loguru import logger

# Remove default handler
logger.remove()

# Sink 1: Stderr (Human Readable)
logger.add(
    sys.stderr,
    level="INFO",
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
)

# Sink 2: File (JSON, Machine Readable)
logger.add(
    "logs/app.log",
    rotation="500 MB",
    retention="10 days",
    serialize=True,
    enqueue=True,
    level="INFO",
)

__all__ = ["logger"]
