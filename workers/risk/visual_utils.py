import os
import cv2
import numpy as np
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

# ---------------------------------------------------------------------
# Vertical anchors (fractions of frame height)
# ---------------------------------------------------------------------

FFBS_BASE_Y_FRAC   = 0.69
KNEE_BASE_Y_FRAC   = 0.70
TORSO_Y_FRAC       = 0.52
UPPER_TORSO_FRAC   = 0.45

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


def _risk_style(conf: str, scale: int):
    if conf == "HIGH":
        return (0, 0, 255), max(2, int(scale * 0.015))
    elif conf == "MEDIUM":
        return (0, 165, 255), max(2, int(scale * 0.012))
    return (255, 200, 0), max(2, int(scale * 0.010))


# ---------------------------------------------------------------------
# Subject geometry (FRAME ONLY)
# ---------------------------------------------------------------------

_HOG = None

def _get_hog():
    global _HOG
    if _HOG is None:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        _HOG = hog
    return _HOG


def _estimate_subject_geometry(frame) -> Tuple[int, int]:
    h, w = frame.shape[:2]
    min_x, max_x = int(w * 0.15), int(w * 0.85)

    try:
        hog = _get_hog()
        scale_down = 640 / max(w, 1)
        small = cv2.resize(frame, (int(w * scale_down), int(h * scale_down))) if scale_down < 1 else frame

        rects, _ = hog.detectMultiScale(small, winStride=(8,8), padding=(8,8), scale=1.05)
        if len(rects):
            x, y, rw, rh = max(rects, key=lambda r: r[2]*r[3])
            cx = int((x + rw//2) / max(scale_down, 1e-6))
            return max(min_x, min(max_x, cx)), max(1, int(rh / max(scale_down,1e-6)))
    except Exception:
        pass

    return int(w * 0.5), int(h * 0.45)


# ---------------------------------------------------------------------
# Horizontal refinements
# ---------------------------------------------------------------------

def _refine_front_foot(frame, cx):
    h, _ = frame.shape[:2]
    roi = frame[int(h*0.62):int(h*0.80), :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
    xs = np.where(mask > 0)[1]
    if len(xs) < 200:
        return cx

    p90, p10 = np.percentile(xs, [90, 10])
    lead = p90 if abs(p90-cx) > abs(p10-cx) else p10
    return int(0.75*cx + 0.25*lead)


def _refine_mid_body(frame, cx, y_frac):
    h, _ = frame.shape[:2]
    y1 = int(h * (y_frac - 0.05))
    y2 = int(h * (y_frac + 0.05))
    roi = frame[y1:y2, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
    xs = np.where(mask > 0)[1]
    if len(xs) < 200:
        return cx
    return int(0.8*cx + 0.2*int(xs.mean()))


# ---------------------------------------------------------------------
# Risk visuals
# ---------------------------------------------------------------------

def _draw_ffbs(frame, conf):
    h, _ = frame.shape[:2]
    cx, subject_h = _estimate_subject_geometry(frame)
    cx = _refine_front_foot(frame, cx)

    color, t = _risk_style(conf, subject_h)
    base_y = int(h * FFBS_BASE_Y_FRAC)
    arrow_len = int(subject_h * 0.45)

    cv2.line(frame, (cx, base_y), (cx, base_y-arrow_len), color, t, cv2.LINE_AA)
    cv2.circle(frame, (cx, base_y), max(3,int(subject_h*0.05)), color, -1)


def _draw_knee_brace(frame, conf):
    h, _ = frame.shape[:2]
    cx, subject_h = _estimate_subject_geometry(frame)
    cx = _refine_front_foot(frame, cx)

    color, t = _risk_style(conf, subject_h)
    knee_y = int(h * KNEE_BASE_Y_FRAC)

    cv2.line(frame, (cx, knee_y-int(subject_h*0.1)), (cx, knee_y+int(subject_h*0.15)), color, t, cv2.LINE_AA)
    cv2.circle(frame, (cx, knee_y), max(3,int(subject_h*0.045)), color, -1)


def _draw_hip_shoulder(frame, conf):
    h, _ = frame.shape[:2]
    cx, subject_h = _estimate_subject_geometry(frame)
    cx = _refine_mid_body(frame, cx, TORSO_Y_FRAC)

    color, t = _risk_style(conf, subject_h)
    y1, y2 = int(h*0.60), int(h*0.45)
    offset = int(subject_h*0.22)

    cv2.line(frame, (cx-offset,y1), (cx+offset,y2), color, t, cv2.LINE_AA)


def _draw_trunk_rotation(frame, conf):
    h, _ = frame.shape[:2]
    cx, subject_h = _estimate_subject_geometry(frame)
    cx = _refine_mid_body(frame, cx, TORSO_Y_FRAC)

    color, t = _risk_style(conf, subject_h)
    span = int(subject_h*0.4)
    y = int(h * TORSO_Y_FRAC)

    cv2.line(frame, (cx-span,y), (cx+span,y), color, t, cv2.LINE_AA)
    cv2.line(frame, (cx+span,y), (cx-span,y), color, t, cv2.LINE_AA)


def _draw_lateral_trunk(frame, conf):
    h, _ = frame.shape[:2]
    cx, subject_h = _estimate_subject_geometry(frame)
    cx = _refine_mid_body(frame, cx, UPPER_TORSO_FRAC)

    color, t = _risk_style(conf, subject_h)
    span = int(subject_h*0.45)
    y = int(h * UPPER_TORSO_FRAC)

    cv2.line(frame, (cx-span,y), (cx+span,y), color, t, cv2.LINE_AA)
    cv2.line(frame, (cx+span,y), (cx-span,y), color, t, cv2.LINE_AA)


# ---------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------

def draw_and_save_visual(
    *, video_path: str, frame_idx: int, risk_id: str,
    pose_frames=None, visual_confidence: str = "LOW", run_id: Optional[str] = None
):
    if not run_id:
        raise ValueError("run_id required")

    frame_idx = _safe_int(frame_idx)
    if frame_idx is None or not os.path.exists(video_path):
        return None

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None

    {
        "front_foot_braking_shock": _draw_ffbs,
        "knee_brace_failure": _draw_knee_brace,
        "hip_shoulder_mismatch": _draw_hip_shoulder,
        "trunk_rotation_snap": _draw_trunk_rotation,
        "lateral_trunk_lean": _draw_lateral_trunk,
    }.get(risk_id, lambda *_: None)(frame, visual_confidence)

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

