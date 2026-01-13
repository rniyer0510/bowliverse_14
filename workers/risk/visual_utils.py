import os
import cv2
from typing import Optional, Tuple

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

# DO NOT hardcode host / port / protocol
# This directory is mounted by FastAPI at /visuals
VISUAL_DIR = "/tmp/actionlab_frames"
os.makedirs(VISUAL_DIR, exist_ok=True)

DEFAULT_COLOR = (0, 0, 255)   # Red
THICKNESS = 2


# ---------------------------------------------------------------------
# Frame utilities
# ---------------------------------------------------------------------

def _safe_int(x):
    try:
        return int(x)
    except Exception:
        return None


def _clamp(val, lo, hi):
    return max(lo, min(hi, val))


# ---------------------------------------------------------------------
# Core visual renderer (used by risk_worker)
# ---------------------------------------------------------------------

def draw_and_save_visual(
    *,
    video_path: str,
    frame_idx: int,
    risk_id: str,
    pose_frames=None,
    visual_confidence: str = "LOW",
):
    """
    Draws a minimal visual overlay and saves the frame.
    This is intentionally conservative: arrows/markers only.
    """
    frame_idx = _safe_int(frame_idx)
    if frame_idx is None:
        return None

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        return None

    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # Minimal explanatory arrow (generic but informative)
    cv2.arrowedLine(
        frame,
        (cx, cy + 40),
        (cx, cy - 40),
        DEFAULT_COLOR,
        THICKNESS,
        tipLength=0.25,
    )

    fname = f"{risk_id}_{frame_idx}.png"
    path = os.path.join(VISUAL_DIR, fname)
    cv2.imwrite(path, frame)

    # 🔒 CRITICAL FIX: RELATIVE URL ONLY
    return {
        "frame": frame_idx,
        "anchor": "center",
        "visual_confidence": visual_confidence,
        "image_url": f"/visuals/{fname}",
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
      1. Prefer anchor_frame if inside window
      2. Else pick mid-point of window
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
# LEGACY COMPATIBILITY HELPERS (DO NOT REMOVE)
# Required by existing risk modules
# ---------------------------------------------------------------------

def extract_frame(video_path: str, frame_idx: int):
    """
    Legacy helper expected by older risk modules.
    Returns raw BGR frame or None.
    """
    frame_idx = _safe_int(frame_idx)
    if frame_idx is None:
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
):
    """
    Legacy helper expected by older risk modules.
    Saves raw image without overlays.
    """
    if image is None:
        return None

    frame_idx = _safe_int(frame_idx)
    if frame_idx is None:
        return None

    fname = f"{risk_id}_{frame_idx}.png"
    path = os.path.join(VISUAL_DIR, fname)
    cv2.imwrite(path, image)

    # 🔒 SAME FIX HERE
    return {
        "frame": frame_idx,
        "anchor": "center",
        "visual_confidence": "LOW",
        "image_url": f"/visuals/{fname}",
    }

