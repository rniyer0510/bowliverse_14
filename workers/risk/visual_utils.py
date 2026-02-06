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
# Semantic vertical anchors (fractions of frame height)
# ---------------------------------------------------------------------

FFBS_BASE_Y_FRAC = 0.69
TORSO_Y_FRAC = 0.52

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
# Subject geometry estimation (FRAME ONLY — NO POSE)
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
    min_x = int(w * 0.15)
    max_x = int(w * 0.85)

    try:
        hog = _get_hog()
        scale_down = 640 / max(w, 1)

        small = (
            cv2.resize(frame, (int(w * scale_down), int(h * scale_down)))
            if scale_down < 1.0
            else frame
        )

        rects, _ = hog.detectMultiScale(
            small,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05
        )

        if rects is not None and len(rects) > 0:
            x, y, rw, rh = max(rects, key=lambda r: r[2] * r[3])
            cx = int((x + rw // 2) / max(scale_down, 1e-6))
            subject_h = int(rh / max(scale_down, 1e-6))
            return max(min_x, min(max_x, cx)), max(1, subject_h)
    except Exception:
        pass

    return int(w * 0.50), int(h * 0.45)


# ---------------------------------------------------------------------
# Horizontal refinement for FRONT FOOT
# ---------------------------------------------------------------------

def _refine_x_using_ground_mass(frame, cx):
    h, w = frame.shape[:2]
    roi = frame[int(h * 0.62):int(h * 0.80), :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)

    xs = np.where(mask > 0)[1]
    if len(xs) < 200:
        return cx

    p90 = int(np.percentile(xs, 90))
    p10 = int(np.percentile(xs, 10))
    lead_x = p90 if abs(p90 - cx) > abs(p10 - cx) else p10
    return int(0.75 * cx + 0.25 * lead_x)


# ---------------------------------------------------------------------
# Risk visuals (FRAME ONLY — NO POSE)
# ---------------------------------------------------------------------

def _draw_ffbs(frame, conf):
    h, _ = frame.shape[:2]
    cx, subject_h = _estimate_subject_geometry(frame)
    cx = _refine_x_using_ground_mass(frame, cx)
    color, t = _risk_style(conf, subject_h)

    base_y = int(h * FFBS_BASE_Y_FRAC)
    end_y = max(0, base_y - int(subject_h * 0.45))

    cv2.line(frame, (cx, base_y), (cx, end_y), color, t, cv2.LINE_AA)
    head = max(4, int(subject_h * 0.05))
    cv2.line(frame, (cx, end_y), (cx - head, end_y + head), color, t)
    cv2.line(frame, (cx, end_y), (cx + head, end_y + head), color, t)
    cv2.circle(frame, (cx, base_y), max(3, int(subject_h * 0.05)), color, -1)


def _draw_knee_brace(frame, conf):
    h, _ = frame.shape[:2]
    cx, subject_h = _estimate_subject_geometry(frame)
    cx = _refine_x_using_ground_mass(frame, cx)
    color, base_t = _risk_style(conf, subject_h)

    knee_y = int(h * 0.615)
    arrow_len = int(subject_h * 0.45)

    cv2.arrowedLine(
        frame,
        (cx, knee_y - int(arrow_len * 0.4)),
        (cx, knee_y + int(arrow_len * 0.6)),
        color,
        max(6, int(base_t * 2.3)),
        tipLength=0.16,
        line_type=cv2.LINE_AA,
    )


def _draw_trunk_rotation(frame, conf):
    h, w = frame.shape[:2]
    cx, subject_h = _estimate_subject_geometry(frame)
    color, base_t = _risk_style(conf, subject_h)

    torso_y = int((h * 0.5) - subject_h // 2 + subject_h * 0.40)
    span = int(subject_h * (0.32 if conf == "HIGH" else 0.26 if conf == "MEDIUM" else 0.22))

    cv2.arrowedLine(
        frame,
        (cx - span // 2, torso_y),
        (cx + span // 2, torso_y),
        color,
        max(5, int(base_t * 2.0)),
        tipLength=0.12,
        line_type=cv2.LINE_AA,
    )


def _draw_hip_shoulder(frame, conf):
    h, w = frame.shape[:2]
    cx, subject_h = _estimate_subject_geometry(frame)
    color, base_t = _risk_style(conf, subject_h)

    torso_y = int(h * TORSO_Y_FRAC)
    L = int(subject_h * (0.38 if conf == "HIGH" else 0.32 if conf == "MEDIUM" else 0.26))

    cv2.arrowedLine(
        frame,
        (cx - int(L * 0.375), torso_y + int(L * 0.35)),
        (cx + int(L * 0.375), torso_y - int(L * 0.35)),
        color,
        max(4, int(base_t * 1.8)),
        tipLength=0.12,
        line_type=cv2.LINE_AA,
    )


def _draw_lateral_trunk(frame, conf):
    h, _ = frame.shape[:2]
    cx, subject_h = _estimate_subject_geometry(frame)
    color, base_t = _risk_style(conf, subject_h)

    y = int(h * TORSO_Y_FRAC)
    arrow_len = int(subject_h * 0.30)

    cv2.arrowedLine(
        frame,
        (cx, y),
        (cx + int(arrow_len * 0.7), y + int(arrow_len * 0.35)),
        color,
        max(5, int(base_t * 1.8)),
        tipLength=0.14,
        line_type=cv2.LINE_AA,
    )


def _draw_foot_line_deviation(frame, conf):
    h, _ = frame.shape[:2]
    cx, subject_h = _estimate_subject_geometry(frame)
    color, base_t = _risk_style(conf, subject_h)

    y = int(h * 0.60)
    arrow_len = int(subject_h * 0.34)

    cv2.arrowedLine(
        frame,
        (cx, y),
        (cx + int(arrow_len * 0.75), y + int(arrow_len * 0.22)),
        color,
        max(5, int(base_t * 1.8)),
        tipLength=0.14,
        line_type=cv2.LINE_AA,
    )


# ---------------------------------------------------------------------
# Core visual renderer
# ---------------------------------------------------------------------

def draw_and_save_visual(
    *,
    video_path: str,
    frame_idx: int,
    risk_id: str,
    pose_frames=None,
    visual_confidence: str = "LOW",
    run_id: Optional[str] = None,
    load_body: Optional[str] = None,
    load_level: Optional[str] = None,
):
    if not run_id:
        raise ValueError("run_id is required")

    frame_idx = _safe_int(frame_idx)
    if frame_idx is None or not os.path.exists(video_path):
        return None

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None

    {
        "front_foot_braking_shock": _draw_ffbs,
        "knee_brace_failure": _draw_knee_brace,
        "trunk_rotation_snap": _draw_trunk_rotation,
        "hip_shoulder_mismatch": _draw_hip_shoulder,
        "lateral_trunk_lean": _draw_lateral_trunk,
        "foot_line_deviation": _draw_foot_line_deviation,
    }.get(risk_id, lambda *_: None)(frame, visual_confidence)

    if load_body and load_level:
        h, w = frame.shape[:2]
        text = f"Load on {load_body} – {load_level}"
        scale = max(0.5, w / 1200)
        thickness = max(1, int(scale * 2))
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        x = max(10, (w - tw) // 2)
        y = h - max(12, int(th * 1.8))

        cv2.rectangle(frame, (x - 8, y - th - 8), (x + tw + 8, y + 8), (0, 0, 0), -1)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (230, 230, 230), thickness, cv2.LINE_AA)

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

