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
    h, w = frame.shape[:2]
    cx, subject_h = _estimate_subject_geometry(frame)

    color, t = _risk_style(conf, subject_h)

    y = int(h * TORSO_Y_FRAC)
    span = int(subject_h * 0.40)

    cv2.line(frame, (cx - span, y), (cx + span, y), color, t, cv2.LINE_AA)
    cv2.line(frame, (cx + span, y), (cx - span, y), color, t, cv2.LINE_AA)


def _draw_hip_shoulder(frame, conf):
    """
    Hip–Shoulder Separation / Sequencing cue (FRAME ONLY — NO POSE)

    Visual language (borrows the "FFBS/Knee Brace discipline"):
    - One primary mark only: a slanted "separation axis" arrow through torso.
    - Magnitude encoded via arrow length (LOW < MEDIUM < HIGH).
    - Timing encoded via a subtle vertical shift (HIGH shifts slightly upward = early shoulder open cue).
    - Sequencing quality (jerk) encoded via a subtle "double-stroke" + small chevrons for HIGH only.
    - Color stays driven by _risk_style(conf, scale). (No special-casing.)
    """
    h, w = frame.shape[:2]

    cx, subject_h = _estimate_subject_geometry(frame)
    color, base_t = _risk_style(conf, subject_h)

    # -------------------------
    # Anchor band (torso)
    # -------------------------
    # Keep this near the torso band, not the ground.
    # Slight upward shift on HIGH (proxy for "early shoulder open / timing fault").
    torso_y = int(h * TORSO_Y_FRAC)
    if conf == "HIGH":
        torso_y = max(0, torso_y - int(subject_h * 0.04))  # subtle, not dramatic
    elif conf == "LOW":
        torso_y = min(h - 1, torso_y + int(subject_h * 0.01))  # tiny stabilizing shift

    # -------------------------
    # Magnitude (arrow length)
    # -------------------------
    # Keep it local (technique cue), not FFBS-big.
    # Knee brace taught us: avoid cartoonish spans.
    if conf == "HIGH":
        L = int(subject_h * 0.40)
    elif conf == "MEDIUM":
        L = int(subject_h * 0.34)
    else:
        L = int(subject_h * 0.28)

    # Slant: hips vs shoulders differential reads best as an up-and-across axis.
    # (Not vertical. Not symmetric horizontal.)
    dx = int(L * 0.78)
    dy = int(L * 0.28)  # vertical component

    # Define endpoints (hip lower, shoulder higher)
    hip_y = min(h - 1, torso_y + dy)
    sh_y  = max(0,     torso_y - dy)
    x1 = max(0, min(w - 1, cx - dx // 2))
    x2 = max(0, min(w - 1, cx + dx // 2))

    # -------------------------
    # Thickness (load emphasis)
    # -------------------------
    # Forceful but not impact-level (FFBS).
    thickness = max(4, int(base_t * 1.8))

    # Main separation arrow (hip -> shoulder)
    cv2.arrowedLine(
        frame,
        (x1, hip_y),
        (x2, sh_y),
        color,
        thickness,
        tipLength=0.14,  # compact head (knee brace style)
        line_type=cv2.LINE_AA,
    )

    # Small anatomical anchors (subtle, not blobs)
    r = max(3, int(subject_h * 0.018))
    cv2.circle(frame, (x1, hip_y), r, color, -1)
    cv2.circle(frame, (x2, sh_y), r, color, -1)

    # -------------------------
    # Sequencing quality (jerk) — HIGH only
    # -------------------------
    if conf == "HIGH":
        # 1) A subtle parallel "double-stroke" to imply abrupt snap (without turning everything red)
        off = max(2, thickness // 3)
        cv2.line(
            frame,
            (max(0, x1), min(h - 1, hip_y + off)),
            (min(w - 1, x2), max(0, sh_y + off)),
            color,
            max(2, thickness // 2),
            cv2.LINE_AA,
        )

        # 2) Tiny chevrons along the shaft (reads as "snap/jerk" on mobile)
        # Place them around the middle third.
        mx1 = int(0.60 * x1 + 0.40 * x2)
        my1 = int(0.60 * hip_y + 0.40 * sh_y)
        mx2 = int(0.40 * x1 + 0.60 * x2)
        my2 = int(0.40 * hip_y + 0.60 * sh_y)

        che = max(4, int(subject_h * 0.025))
        ct = max(2, thickness // 3)
        # Two small V's oriented roughly perpendicular to the axis
        cv2.line(frame, (mx1, my1), (mx1 - che, my1 + che // 2), color, ct, cv2.LINE_AA)
        cv2.line(frame, (mx1, my1), (mx1 + che, my1 - che // 2), color, ct, cv2.LINE_AA)
        cv2.line(frame, (mx2, my2), (mx2 - che, my2 + che // 2), color, ct, cv2.LINE_AA)
        cv2.line(frame, (mx2, my2), (mx2 + che, my2 - che // 2), color, ct, cv2.LINE_AA)


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

