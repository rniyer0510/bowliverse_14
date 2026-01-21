import os
import cv2
from typing import Optional, Tuple

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# Directory served by FastAPI as /visuals
VISUAL_DIR = "/tmp/actionlab_frames"
os.makedirs(VISUAL_DIR, exist_ok=True)

# Base public URL (LOCAL / PROD SAFE)
PUBLIC_BASE_URL = os.environ.get(
    "ACTIONLAB_PUBLIC_BASE_URL",
    "http://127.0.0.1:8000",
)

DEFAULT_COLOR = (0, 0, 255)  # Red
THICKNESS = 2


# ---------------------------------------------------------------------
# URL helper (🔒 SINGLE SOURCE OF TRUTH)
# ---------------------------------------------------------------------

def _public_url(path: str) -> str:
    """
    Converts a relative path into a fully-qualified public URL.
    """
    return PUBLIC_BASE_URL.rstrip("/") + "/" + path.lstrip("/")


# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------

def _safe_int(x):
    try:
        return int(x)
    except Exception:
        return None


def _clamp(val, lo, hi):
    return max(lo, min(hi, val))


# ---------------------------------------------------------------------
# Core visual renderer (USED BY risk_worker ONLY)
# ---------------------------------------------------------------------

def draw_and_save_visual(
    *,
    video_path: str,
    frame_idx: int,
    risk_id: str,
    pose_frames=None,
    visual_confidence: str = "LOW",
    run_id: Optional[str] = None,
):
    """
    Draws a minimal explanatory overlay and saves the frame.

    This function is:
    - risk-agnostic
    - event-anchored
    - deterministic
    """

    frame_idx = _safe_int(frame_idx)
    if frame_idx is None:
        return None

    if not video_path or not os.path.exists(video_path):
        return None

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        return None

    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # Minimal, neutral arrow (event-centric)
    cv2.arrowedLine(
        frame,
        (cx, cy + 40),
        (cx, cy - 40),
        DEFAULT_COLOR,
        THICKNESS,
        tipLength=0.25,
    )

    # Folder per analysis run (prevents collisions)
    run_id = run_id or "analysis_default"
    out_dir = os.path.join(VISUAL_DIR, run_id)
    os.makedirs(out_dir, exist_ok=True)

    fname = f"{risk_id}_{frame_idx}.png"
    abs_path = os.path.join(out_dir, fname)
    cv2.imwrite(abs_path, frame)

    rel_path = f"/visuals/{run_id}/{fname}"

    return {
        "frame": frame_idx,
        "anchor": "event",
        "visual_confidence": visual_confidence,
        "image_url": _public_url(rel_path),
    }


# ---------------------------------------------------------------------
# Window-based visual selection helper
# ---------------------------------------------------------------------

def select_best_visual_frame(
    *,
    anchor_frame: int,
    window: Tuple[int, int],
    fallback: Optional[int] = None,
):
    """
    Picks the most reasonable frame inside a window.

    Strategy:
      1. Use anchor_frame if inside window
      2. Else midpoint of window
      3. Else fallback
    """
    if anchor_frame is not None:
        if window[0] <= anchor_frame <= window[1]:
            return anchor_frame

    mid = (window[0] + window[1]) // 2
    if window[0] <= mid <= window[1]:
        return mid

    return fallback


# ---------------------------------------------------------------------
# LEGACY HELPERS (DO NOT REMOVE — REQUIRED BY OLD RISK FILES)
# ---------------------------------------------------------------------

def extract_frame(video_path: str, frame_idx: int):
    """
    Legacy helper used by older risk modules.
    Returns raw BGR frame or None.
    """
    frame_idx = _safe_int(frame_idx)
    if frame_idx is None or not video_path:
        return None

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        return None
    return frame


def save_visual(
    image,
    risk_id: str,
    frame_idx: int,
    run_id: Optional[str] = None,
):
    """
    Legacy helper expected by older risk modules.
    Saves raw frame without overlays.
    """

    if image is None:
        return None

    frame_idx = _safe_int(frame_idx)
    if frame_idx is None:
        return None

    run_id = run_id or "analysis_default"
    out_dir = os.path.join(VISUAL_DIR, run_id)
    os.makedirs(out_dir, exist_ok=True)

    fname = f"{risk_id}_{frame_idx}.png"
    abs_path = os.path.join(out_dir, fname)
    cv2.imwrite(abs_path, image)

    rel_path = f"/visuals/{run_id}/{fname}"

    return {
        "frame": frame_idx,
        "anchor": "event",
        "visual_confidence": "LOW",
        "image_url": _public_url(rel_path),
    }

