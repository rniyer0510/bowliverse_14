from __future__ import annotations

import os
import time
from typing import Dict

from app.common.logger import get_logger

logger = get_logger(__name__)

DEFAULT_RENDER_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "storage", "renders")
)
FALLBACK_RENDER_DIR = "/tmp/actionlab_renders"


def get_render_dir() -> str:
    configured = os.getenv("ACTIONLAB_RENDER_DIR", DEFAULT_RENDER_DIR)
    try:
        os.makedirs(configured, exist_ok=True)
        return configured
    except OSError:
        os.makedirs(FALLBACK_RENDER_DIR, exist_ok=True)
        return FALLBACK_RENDER_DIR


def render_retention_days() -> int:
    raw = os.getenv("ACTIONLAB_RENDER_RETENTION_DAYS", "15").strip()
    try:
        return max(1, int(raw))
    except Exception:
        return 15


def cleanup_old_renders(render_dir: str, *, retention_days: int | None = None) -> Dict[str, int]:
    retention = retention_days or render_retention_days()
    cutoff = time.time() - (retention * 86400)
    scanned = 0
    removed = 0

    try:
        entries = list(os.scandir(render_dir))
    except FileNotFoundError:
        return {"scanned": 0, "removed": 0}
    except OSError as exc:
        logger.warning("[render_storage] cleanup scan failed dir=%s error=%s", render_dir, exc)
        return {"scanned": 0, "removed": 0}

    for entry in entries:
        if not entry.is_file() or not entry.name.endswith(".mp4"):
            continue
        scanned += 1
        try:
            stat = entry.stat()
        except OSError:
            continue
        if stat.st_mtime >= cutoff:
            continue
        try:
            os.remove(entry.path)
            removed += 1
        except OSError as exc:
            logger.warning("[render_storage] cleanup delete failed path=%s error=%s", entry.path, exc)

    if removed:
        logger.info(
            "[render_storage] removed %s old renders from %s (retention_days=%s)",
            removed,
            render_dir,
            retention,
        )
    return {"scanned": scanned, "removed": removed}
