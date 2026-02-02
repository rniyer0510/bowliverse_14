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

DEFAULT_COLOR = (0, 0, 255)

# ---------------------------------------------------------------------
# Semantic vertical anchors (fractions of frame height)
# ---------------------------------------------------------------------

FFBS_BASE_Y_FRAC = 0.69
KNEE_BASE_Y_FRAC = 0.70
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
    """
    Returns:
        (cx, subject_height_px)
    """
    h, w = frame.shape[:2]
    min_x = int(w * 0.15)
    max_x = int(w * 0.85)

    try:
        hog = _get_hog()
        scale_down = 640 / max(w, 1)

        if scale_down < 1.0:
            small = cv2.resize(
                frame,
                (int(w * scale_down), int(h * scale_down))
            )
        else:
            small = frame

        rects, _ = hog.detectMultiScale(
            small,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05
        )

        if rects is not None and len(rects) > 0:
            x, y, rw, rh = max(rects, key=lambda r: r[2] * r[3])
            cx_small = x + rw // 2

            cx = int(cx_small / max(scale_down, 1e-6))
            subject_h = int(rh / max(scale_down, 1e-6))

            return (
                max(min_x, min(max_x, cx)),
                max(1, subject_h),
            )
    except Exception:
        pass

    # Fallback
    return int(w * 0.50), int(h * 0.45)


# ---------------------------------------------------------------------
# Horizontal refinement for FRONT FOOT (not centroid)
# ---------------------------------------------------------------------

def _refine_x_using_ground_mass(frame, cx):
    """
    Bias x toward the LEADING (front) foot using foreground mass
    in the ground-contact band.
    """
    h, w = frame.shape[:2]

    y1 = int(h * 0.62)
    y2 = int(h * 0.80)

    roi = frame[y1:y2, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)

    xs = np.where(mask > 0)[1]
    if len(xs) < 200:
        return cx

    # Leading edge, not centroid
    p90 = int(np.percentile(xs, 90))
    p10 = int(np.percentile(xs, 10))

    # Pick forward-most side relative to body center
    if abs(p90 - cx) > abs(p10 - cx):
        lead_x = p90
    else:
        lead_x = p10

    # Gentle bias toward front foot
    return int(0.75 * cx + 0.25 * lead_x)


# ---------------------------------------------------------------------
# Risk visuals (FRAME ONLY — NO POSE)
# ---------------------------------------------------------------------

def _draw_ffbs(frame, conf):
    """
    Front Foot Braking Shock
    """
    h, w = frame.shape[:2]

    cx, subject_h = _estimate_subject_geometry(frame)
    cx = _refine_x_using_ground_mass(frame, cx)

    color, t = _risk_style(conf, subject_h)

    base_y = int(h * FFBS_BASE_Y_FRAC)
    arrow_len = int(subject_h * 0.45)
    end_y = max(0, base_y - arrow_len)

    cv2.line(frame, (cx, base_y), (cx, end_y), color, t, cv2.LINE_AA)

    head = max(4, int(subject_h * 0.05))
    cv2.line(frame, (cx, end_y), (cx - head, end_y + head), color, t, cv2.LINE_AA)
    cv2.line(frame, (cx, end_y), (cx + head, end_y + head), color, t, cv2.LINE_AA)

    cv2.circle(
        frame,
        (cx, base_y),
        max(3, int(subject_h * 0.05)),
        color,
        -1,
    )


def _draw_knee_brace(frame, conf, anchor_x: Optional[int] = None):
    """
    Knee Brace Failure
    Visual: thick downward arrow crossing the knee joint,
    indicating front knee yielding / failing to brace under load.
    """
    h, w = frame.shape[:2]

    # Use same geometry logic as rest of visual_utils
    cx, subject_h = _estimate_subject_geometry(frame)
    cx = _refine_x_using_ground_mass(frame, cx)

    scale = subject_h
    color, base_t = _risk_style(conf, scale)

    # Knee joint anchor (stable for your footage)
    KNEE_FRAC = 0.615
    knee_y = int(h * KNEE_FRAC)

    # Arrow length scales with bowler size
    arrow_len = int(scale * 0.45)

    # Arrow crosses the joint
    start_y = max(0, knee_y - int(arrow_len * 0.40))
    end_y   = min(h - 1, knee_y + int(arrow_len * 0.60))

    thickness = max(6, int(base_t * 2.3))

    cv2.arrowedLine(
        frame,
        (cx, start_y),
        (cx, end_y),
        color,
        thickness,
        tipLength=0.16,          # ↓ smaller, force-like
        line_type=cv2.LINE_AA,
    )

    # Small anatomical knee hinge
    joint_radius = max(4, int(scale * 0.035))
    cv2.circle(
        frame,
        (cx, knee_y),
        joint_radius,
        color,
        -1,
    )

    if conf == "HIGH":
        rad = int(joint_radius * 1.4)
        for angle in (-35, 35):
            dx = int(rad * np.cos(np.deg2rad(angle)))
            dy = int(rad * np.sin(np.deg2rad(angle)))
            cv2.line(
                frame,
                (cx, knee_y),
                (cx + dx, knee_y + dy),
                color,
                max(2, thickness // 3),
                cv2.LINE_AA,
            )


def _draw_trunk_rotation(frame, conf):
    """
    Trunk Rotation Snap
    Visual: short, thick horizontal arrow at mid-upper trunk,
    indicating abrupt rotational snap (angular acceleration),
    not lateral lean or directional movement.
    """
    h, w = frame.shape[:2]

    cx, subject_h = _estimate_subject_geometry(frame)
    color, base_t = _risk_style(conf, subject_h)

    # ── SUBJECT-RELATIVE torso anchor (critical fix) ───────────────────
    # Anchor slightly ABOVE hip-shoulder midpoint → true trunk / ribcage
    subject_top = int(h * 0.5) - subject_h // 2
    torso_y = int(subject_top + subject_h * 0.40)   # 0.38–0.42 sweet spot

    torso_y = max(0, min(h - 1, torso_y))

    # ── SNAP length (short, impulsive) ──────────────────────────────────
    if conf == "HIGH":
        span = int(subject_h * 0.32)
    elif conf == "MEDIUM":
        span = int(subject_h * 0.26)
    else:
        span = int(subject_h * 0.22)

    x1 = max(0, cx - span // 2)
    x2 = min(w - 1, cx + span // 2)

    thickness = max(5, int(base_t * 2.0))

    # Main snap arrow
    cv2.arrowedLine(
        frame,
        (x1, torso_y),
        (x2, torso_y),
        color,
        thickness,
        tipLength=0.12,
        line_type=cv2.LINE_AA,
    )

    # Counter-stroke → rotational cue (subtle)
    cv2.line(
        frame,
        (x2 - int(span * 0.15), torso_y),
        (x1 + int(span * 0.15), torso_y),
        color,
        max(2, thickness // 3),
        cv2.LINE_AA,
    )

    # Central pivot marker
    pivot_r = max(3, int(subject_h * 0.018))
    cv2.circle(
        frame,
        (cx, torso_y),
        pivot_r,
        color,
        -1,
    )


def _draw_hip_shoulder(frame, conf):
    """
    Hip–Shoulder Mismatch
    Visual: diagonal torsional cue spanning hip → shoulder region,
    indicating poor sequencing / mismatch (not pure rotation, not impact).
    """
    h, w = frame.shape[:2]
    cx, subject_h = _estimate_subject_geometry(frame)

    color, base_t = _risk_style(conf, subject_h)

    # ── Anchor in torso band (balanced, not hip-heavy) ──────────────────
    torso_y = int(h * TORSO_Y_FRAC)

    # Slight confidence-based vertical nuance (purely visual)
    if conf == "HIGH":
        torso_y -= int(subject_h * 0.03)
    elif conf == "LOW":
        torso_y += int(subject_h * 0.01)

    # ── Length: technique cue, not force cue ────────────────────────────
    if conf == "HIGH":
        L = int(subject_h * 0.38)
    elif conf == "MEDIUM":
        L = int(subject_h * 0.32)
    else:
        L = int(subject_h * 0.26)

    # Diagonal span (clear hip ↔ shoulder separation)
    dx = int(L * 0.75)
    dy = int(L * 0.35)   # ↑ slightly more vertical than before

    # Hip (lower) → Shoulder (upper)
    x1 = max(0, min(w - 1, cx - dx // 2))
    x2 = max(0, min(w - 1, cx + dx // 2))
    y1 = min(h - 1, torso_y + dy)
    y2 = max(0, torso_y - dy)

    thickness = max(4, int(base_t * 1.8))

    # Main torsional arrow
    cv2.arrowedLine(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        thickness,
        tipLength=0.12,   # smaller head → informational, not forceful
        line_type=cv2.LINE_AA,
    )

    # Subtle anatomical anchors (avoid blobs)
    r = max(3, int(subject_h * 0.018))
    cv2.circle(frame, (x1, y1), r, color, -1)
    cv2.circle(frame, (x2, y2), r, color, -1)


def _draw_lateral_trunk(frame, conf):
    """
    Lateral Trunk Lean
    Visual: short diagonal arrow indicating sideways collapse of torso mass.
    """
    h, w = frame.shape[:2]

    cx, subject_h = _estimate_subject_geometry(frame)
    color, base_t = _risk_style(conf, subject_h)

    # Torso anchor (same semantic level as other torso visuals)
    y = int(h * TORSO_Y_FRAC)

    # ── Arrow sizing (learned from knee brace) ──────────────────────────
    # Must be local, not body-dominant
    arrow_len = int(subject_h * 0.30)   # shorter than FFBS, similar to knee brace

    # Slight diagonal downward lean (COM collapse)
    dx = int(arrow_len * 0.70)
    dy = int(arrow_len * 0.35)

    # Thickness communicates load, not impact
    thickness = max(5, int(base_t * 1.8))

    start_pt = (cx, y)
    end_pt   = (cx + dx, y + dy)

    cv2.arrowedLine(
        frame,
        start_pt,
        end_pt,
        color,
        thickness,
        tipLength=0.14,          # small head, consistent with knee brace
        line_type=cv2.LINE_AA,
    )

    # Optional HIGH confidence accent (subtle)
    if conf == "HIGH":
        accent_len = int(arrow_len * 0.35)
        cv2.line(
            frame,
            (cx - accent_len, y - accent_len // 2),
            (cx + accent_len, y + accent_len // 2),
            color,
            max(2, thickness // 3),
            cv2.LINE_AA,
        )



def _draw_foot_line_deviation(frame, conf):
    """
    Foot-Line Deviation
    Visual: diagonal lateral cue near pelvis/lower trunk indicating sideways correction.
    (Best viewed at FFC+1)
    """
    h, w = frame.shape[:2]

    cx, subject_h = _estimate_subject_geometry(frame)
    color, base_t = _risk_style(conf, subject_h)

    y = int(h * 0.60)
    y = max(0, min(h - 1, y))

    arrow_len = int(subject_h * 0.34)
    dx = int(arrow_len * 0.75)
    dy = int(arrow_len * 0.22)

    thickness = max(5, int(base_t * 1.8))

    start_pt = (cx, y)
    end_pt = (max(0, min(w - 1, cx + dx)), max(0, min(h - 1, y + dy)))

    cv2.arrowedLine(
        frame,
        start_pt,
        end_pt,
        color,
        thickness,
        tipLength=0.14,
        line_type=cv2.LINE_AA,
    )

    r = max(4, int(subject_h * 0.04))
    cv2.circle(frame, start_pt, r, color, -1)


# ---------------------------------------------------------------------
# Core visual renderer
# ---------------------------------------------------------------------

def draw_and_save_visual(
    *,
    video_path: str,
    frame_idx: int,
    risk_id: str,
    pose_frames=None,  # intentionally ignored
    visual_confidence: str = "LOW",
    run_id: Optional[str] = None,
):
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
    elif risk_id == "foot_line_deviation":
        _draw_foot_line_deviation(frame, visual_confidence)

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

