from __future__ import annotations

import os
import time
from typing import Dict, Optional

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


def render_bucket_name() -> str:
    return (os.getenv("ACTIONLAB_RENDER_BUCKET") or "").strip()


def render_object_prefix() -> str:
    return (os.getenv("ACTIONLAB_RENDER_OBJECT_PREFIX") or "walkthrough-renders").strip().strip("/")


def normalize_render_filename(filename: str) -> str:
    return os.path.basename((filename or "").strip())


def render_object_name(filename: str) -> str:
    safe_name = normalize_render_filename(filename)
    prefix = render_object_prefix()
    return f"{prefix}/{safe_name}" if prefix else safe_name


def _google_storage():
    try:
        from google.cloud import storage  # type: ignore

        return storage
    except Exception:
        return None


def upload_render_artifact(local_path: str, *, artifact_name: Optional[str] = None) -> Dict[str, Optional[str]]:
    bucket_name = render_bucket_name()
    safe_name = normalize_render_filename(artifact_name or os.path.basename(local_path))
    object_name = render_object_name(safe_name)

    if not bucket_name:
        return {
            "uploaded": False,
            "storage_backend": "local",
            "bucket": None,
            "object_name": None,
            "reason": "render_bucket_not_configured",
        }

    if not local_path or not os.path.exists(local_path):
        return {
            "uploaded": False,
            "storage_backend": "gcs",
            "bucket": bucket_name,
            "object_name": object_name,
            "reason": "local_render_missing",
        }

    storage = _google_storage()
    if storage is None:
        logger.warning("[render_storage] google-cloud-storage unavailable; skipping upload path=%s", local_path)
        return {
            "uploaded": False,
            "storage_backend": "gcs",
            "bucket": bucket_name,
            "object_name": object_name,
            "reason": "google_cloud_storage_unavailable",
        }

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(object_name)
        blob.upload_from_filename(local_path, content_type="video/mp4")
        logger.info(
            "[render_storage] uploaded render path=%s bucket=%s object=%s",
            local_path,
            bucket_name,
            object_name,
        )
        return {
            "uploaded": True,
            "storage_backend": "gcs",
            "bucket": bucket_name,
            "object_name": object_name,
            "reason": None,
        }
    except Exception as exc:
        logger.warning(
            "[render_storage] upload failed path=%s bucket=%s object=%s error=%s",
            local_path,
            bucket_name,
            object_name,
            exc,
        )
        return {
            "uploaded": False,
            "storage_backend": "gcs",
            "bucket": bucket_name,
            "object_name": object_name,
            "reason": "upload_failed",
        }


def download_render_artifact(filename: str) -> Optional[bytes]:
    bucket_name = render_bucket_name()
    if not bucket_name:
        return None

    storage = _google_storage()
    if storage is None:
        logger.warning("[render_storage] google-cloud-storage unavailable; cannot download %s", filename)
        return None

    safe_name = normalize_render_filename(filename)
    object_name = render_object_name(safe_name)

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(object_name)
        if not blob.exists():
            return None
        return blob.download_as_bytes()
    except Exception as exc:
        logger.warning(
            "[render_storage] download failed bucket=%s object=%s error=%s",
            bucket_name,
            object_name,
            exc,
        )
        return None


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
