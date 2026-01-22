import os
import cv2
from typing import Optional, Tuple

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

VISUAL_DIR = "/tmp/actionlab_frames"
os.makedirs(VISUAL_DIR, exist_ok=True)

PUBLIC_BASE_URL = os.environ.get(
    "ACTIONLAB_PUBLIC_BASE_URL",
    "http://127.0.0.1:8000",
)

DEFAULT_COLOR = (0, 0, 255)
THICKNESS = 2

# ---------------------------------------------------------------------
# Normalized visual geometry (NO hardcoded pixels)
# ---------------------------------------------------------------------

GROUND_PAD_FRAC   = 0.04
PELVIS_FRAC       = 0.58
KNEE_FRAC         = 0.72
TORSO_FRAC        = 0.52
TORSO_SPAN_FRAC   = 0.22

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _public_url(path: str) -> str:
    return PUBLIC_BASE_URL.rstrip("/") + "/" + path.lstrip("/")


def _safe_int(x):
    try:
        return int(x)
    except Exception:
        return None


def _ground_y(h: int) -> int:
    return int(h * (1.0 - GROUND_PAD_FRAC))


def _risk_style(conf: str):
    if conf == "HIGH":
        return (0, 0, 255), THICKNESS + 2
    elif conf == "MEDIUM":
        return (0, 165, 255), THICKNESS + 1
    return (255, 200, 0), THICKNESS


# ---------------------------------------------------------------------
# Risk visuals (FRAME ONLY — NO POSE)
# ---------------------------------------------------------------------

def _draw_ffbs(frame, conf):
    h, w = frame.shape[:2]
    color, t = _risk_style(conf)

    x = int(w * 0.55)
    ground = _ground_y(h)
    pelvis = int(h * PELVIS_FRAC)

    cv2.arrowedLine(
        frame,
        (x, ground),
        (x, pelvis),
        color,
        t + 1,
        tipLength=0.30,
        line_type=cv2.LINE_AA,
    )
    cv2.circle(frame, (x, ground), 6, color, -1)


def _draw_knee_brace(frame, conf):
    h, w = frame.shape[:2]
    color, t = _risk_style(conf)

    x = int(w * 0.55)
    knee_y = int(h * KNEE_FRAC)

    cv2.arrowedLine(
        frame,
        (x, knee_y - int(h * 0.04)),
        (x, knee_y + int(h * 0.10)),
        color,
        t + 2,
        tipLength=0.40,
        line_type=cv2.LINE_AA,
    )
    cv2.circle(frame, (x, knee_y), 10, color, -1)


def _draw_trunk_rotation(frame, conf):
    h, w = frame.shape[:2]
    color, t = _risk_style(conf)

    y = int(h * TORSO_FRAC)
    left  = (int(w * 0.40), y)
    right = (int(w * 0.60), y)

    cv2.arrowedLine(frame, left,  right, color, t + 1, tipLength=0.35)
    cv2.arrowedLine(frame, right, left,  color, t + 1, tipLength=0.35)


def _draw_hip_shoulder(frame, conf):
    h, w = frame.shape[:2]
    color, t = _risk_style(conf)

    start = (int(w * 0.48), int(h * 0.60))
    end   = (int(w * 0.56), int(h * 0.45))

    cv2.arrowedLine(
        frame,
        start,
        end,
        color,
        t + 1,
        tipLength=0.30,
        line_type=cv2.LINE_AA,
    )


def _draw_lateral_trunk(frame, conf):
    h, w = frame.shape[:2]
    color, t = _risk_style(conf)

    torso_y = int(h * TORSO_FRAC)
    span = int(w * TORSO_SPAN_FRAC)
    cx = int(w * 0.52)

    left  = (cx - span, torso_y)
    right = (cx + span, torso_y)

    cv2.arrowedLine(frame, left,  right, color, t + 2, tipLength=0.40)
    cv2.arrowedLine(frame, right, left,  color, t + 2, tipLength=0.40)


# ---------------------------------------------------------------------
# Core visual renderer
# ---------------------------------------------------------------------

def draw_and_save_visual(
    *,
    video_path: str,
    frame_idx: int,
    risk_id: str,
    pose_frames=None,   # intentionally ignored
    visual_confidence: str = "LOW",
    run_id: Optional[str] = None,
):
    # ---------------------------
    # Enforce per-analysis isolation
    # ---------------------------
    if not run_id:
        raise ValueError("run_id is required for visual generation")

    frame_idx = _safe_int(frame_idx)
    if frame_idx is None or not video_path or not os.path.exists(video_path):
        return None

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None

    if risk_id == "front_foot_braking_shock":
        _draw_ffbs(frame, visual_confidence)
    elif risk_id == "knee_brace_failure":
        _draw_knee_brace(frame, visual_confidence)
    elif risk_id == "trunk_rotation_snap":
        _draw_trunk_rotation(frame, visual_confidence)
    elif risk_id == "hip_shoulder_mismatch":
        _draw_hip_shoulder(frame, visual_confidence)
    elif risk_id == "lateral_trunk_lean":
        _draw_lateral_trunk(frame, visual_confidence)
    else:
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        cv2.arrowedLine(
            frame,
            (cx, cy + 40),
            (cx, cy - 40),
            DEFAULT_COLOR,
            THICKNESS,
            tipLength=0.25,
        )

    out_dir = os.path.join(VISUAL_DIR, run_id)
    os.makedirs(out_dir, exist_ok=True)

    fname = f"{risk_id}_{frame_idx}.png"
    cv2.imwrite(os.path.join(out_dir, fname), frame)

    return {
        "frame": frame_idx,
        "anchor": "event",
        "visual_confidence": visual_confidence,
        "image_url": _public_url(f"/visuals/{run_id}/{fname}"),
    }

