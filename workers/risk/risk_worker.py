from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.common.logger import get_logger

from app.workers.risk.front_foot_braking import compute_front_foot_braking_shock
from app.workers.risk.knee_brace_failure import compute_knee_brace_failure
from app.workers.risk.trunk_rotation_snap import compute_trunk_rotation_snap
from app.workers.risk.hip_shoulder_mismatch import compute_hip_shoulder_mismatch
from app.workers.risk.lateral_trunk_lean import compute_lateral_trunk_lean
from app.workers.risk.foot_line_deviation import compute_foot_line_deviation

from app.workers.risk.visual_utils import draw_and_save_visual
from app.workers.risk.benchmarks import attach_deviation_and_impact

logger = get_logger(__name__)

# ---------------------------------------------------------------------
# Risk configuration (LOCKED floors)
# ---------------------------------------------------------------------
RISK_CONFIG = {
    "front_foot_braking_shock": {"floor": 0.15},
    "knee_brace_failure": {"floor": 0.15},
    "trunk_rotation_snap": {"floor": 0.15},
    "hip_shoulder_mismatch": {"floor": 0.15},
    "lateral_trunk_lean": {"floor": 0.15},
    "foot_line_deviation": {"floor": 0.15},
}

# ---------------------------------------------------------------------
# Semantic override: "what body area should the footer mention?"
# This is NOT physics, not a correction cue - just a safe, honest label.
# ---------------------------------------------------------------------
PRIMARY_LOAD_OVERRIDE: Dict[str, str] = {
    # Foot-line deviation loads adductors/groin first (knee is downstream/secondary)
    "foot_line_deviation": "groin",
    # Sequencing/torso risks: keep these broad and non-prescriptive
    "hip_shoulder_mismatch": "hip",
    "trunk_rotation_snap": "lower back",
    "lateral_trunk_lean": "lower back",
}

# Pose landmarks
LS, LE, LW = 11, 13, 15
RS, RE, RW = 12, 14, 16
LH, RH = 23, 24
LK, RK = 25, 26
LA, RA = 27, 28
LHEEL, RHEEL = 29, 30
LFOOT, RFOOT = 31, 32
NOSE = 0
LEYE = 2
REYE = 5
LEAR = 7
REAR = 8
MIN_VIS = 0.25
FULL_BODY_GUIDANCE_MESSAGE = (
    "Record one full delivery from the front-side angle. "
    "Keep your full body visible throughout."
)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _f(x: Any, d: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return d


def _emit(obj: Optional[Dict[str, Any]], risk_id: str) -> Dict[str, Any]:
    """
    Normalize output and enforce floor.
    """
    floor = float(RISK_CONFIG[risk_id]["floor"])

    if not isinstance(obj, dict):
        return {
            "risk_id": risk_id,
            "signal_strength": floor,
            "confidence": 0.0,
        }

    out = dict(obj)
    out["risk_id"] = risk_id
    out["signal_strength"] = max(_f(out.get("signal_strength"), 0.0), floor)
    out["confidence"] = _f(out.get("confidence"), 0.0)
    return out


def _event_frame(events: Dict[str, Any], key: str) -> Optional[int]:
    v = events.get(key) or {}
    f = v.get("frame")
    if isinstance(f, int):
        return f
    try:
        return int(f)
    except Exception:
        return None


def _load_level_from_band(band: Optional[int]) -> Optional[str]:
    """
    Maps deviation band to footer-friendly load level.
    """
    if band is None:
        return None
    if band <= 2:
        return "low"
    if band == 3:
        return "moderate"
    return "high"


def _is_visible(lm: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(lm, dict):
        return False
    return _f(lm.get("visibility"), 0.0) >= MIN_VIS


def _get_landmark(landmarks: List[Dict[str, Any]], idx: int) -> Optional[Dict[str, Any]]:
    if idx < 0 or idx >= len(landmarks):
        return None
    val = landmarks[idx]
    return val if isinstance(val, dict) else None


def _point_vis(pt: Any) -> float:
    if isinstance(pt, dict):
        return _f(pt.get("visibility", pt.get("v", 0.0)), 0.0)
    if isinstance(pt, (tuple, list)) and len(pt) >= 4:
        return _f(pt[3], 0.0)
    return 1.0 if isinstance(pt, (tuple, list)) and len(pt) >= 2 else 0.0


def _point_xy(pt: Any) -> Optional[tuple[float, float]]:
    try:
        if isinstance(pt, dict):
            return (_f(pt.get("x")), _f(pt.get("y")))
        if isinstance(pt, (tuple, list)) and len(pt) >= 2:
            return (float(pt[0]), float(pt[1]))
    except Exception:
        return None
    return None


def _visible_point(
    landmarks: List[Any],
    idx: int,
    *,
    min_vis: float = MIN_VIS,
) -> Optional[tuple[float, float]]:
    if idx < 0 or idx >= len(landmarks):
        return None
    pt = landmarks[idx]
    if _point_vis(pt) < min_vis:
        return None
    return _point_xy(pt)


def _first_visible_point(
    landmarks: List[Any],
    indices: List[int],
    *,
    min_vis: float = MIN_VIS,
) -> Optional[tuple[float, float]]:
    for idx in indices:
        pt = _visible_point(landmarks, idx, min_vis=min_vis)
        if pt is not None:
            return pt
    return None


def _is_rear_view_capture(pose_frames: List[Dict[str, Any]]) -> bool:
    """
    Heuristic for rear-view only:
    - Torso (shoulders + hips) is visible in enough frames.
    - Wrists are mostly not visible while torso remains visible.
    """
    torso_tracked = 0
    wrist_visible = 0
    elbow_visible = 0

    for frame in pose_frames or []:
        landmarks = (frame or {}).get("landmarks")
        if not isinstance(landmarks, list) or not landmarks:
            continue

        ls = _get_landmark(landmarks, LS)
        rs = _get_landmark(landmarks, RS)
        lh = _get_landmark(landmarks, LH)
        rh = _get_landmark(landmarks, RH)

        shoulders_ok = _is_visible(ls) and _is_visible(rs)
        hips_ok = _is_visible(lh) and _is_visible(rh)
        if not (shoulders_ok and hips_ok):
            continue

        torso_tracked += 1

        lw = _get_landmark(landmarks, LW)
        rw = _get_landmark(landmarks, RW)
        if _is_visible(lw) or _is_visible(rw):
            wrist_visible += 1

        le = _get_landmark(landmarks, LE)
        re = _get_landmark(landmarks, RE)
        if _is_visible(le) or _is_visible(re):
            elbow_visible += 1

    if torso_tracked < 20:
        return False

    wrist_ratio = wrist_visible / float(torso_tracked)
    elbow_ratio = elbow_visible / float(torso_tracked)

    # Rear-view signature: torso/elbows visible, wrists mostly occluded.
    return wrist_ratio < 0.20 and elbow_ratio >= 0.55


def _has_full_body_front_visual_support(frame: Dict[str, Any]) -> bool:
    landmarks = (frame or {}).get("landmarks")
    if not isinstance(landmarks, list) or len(landmarks) <= RFOOT:
        return False

    head = _first_visible_point(landmarks, [NOSE, LEYE, REYE, LEAR, REAR], min_vis=0.35)
    shoulders = [
        _visible_point(landmarks, LS, min_vis=0.35),
        _visible_point(landmarks, RS, min_vis=0.35),
    ]
    hips = [
        _visible_point(landmarks, LH, min_vis=0.35),
        _visible_point(landmarks, RH, min_vis=0.35),
    ]
    knees = [
        _visible_point(landmarks, LK, min_vis=0.30),
        _visible_point(landmarks, RK, min_vis=0.30),
    ]
    left_foot = _first_visible_point(landmarks, [LA, LHEEL, LFOOT], min_vis=0.30)
    right_foot = _first_visible_point(landmarks, [RA, RHEEL, RFOOT], min_vis=0.30)

    if (
        head is None
        or any(pt is None for pt in shoulders)
        or any(pt is None for pt in hips)
        or left_foot is None
        or right_foot is None
    ):
        return False

    tracked = [head, left_foot, right_foot]
    tracked.extend(pt for pt in shoulders if pt is not None)
    tracked.extend(pt for pt in hips if pt is not None)
    tracked.extend(pt for pt in knees if pt is not None)

    xs = [pt[0] for pt in tracked]
    ys = [pt[1] for pt in tracked]
    foot_bottom = max(left_foot[1], right_foot[1])

    if min(xs) <= 0.03 or max(xs) >= 0.97:
        return False
    if head[1] <= 0.02 or foot_bottom >= 0.98:
        return False
    if max(ys) - min(ys) < 0.45:
        return False

    return True


def _front_visual_block_reason(
    pose_frames: List[Dict[str, Any]],
    anchor: int,
) -> Optional[str]:
    if not pose_frames:
        return FULL_BODY_GUIDANCE_MESSAGE

    start = max(0, int(anchor) - 2)
    end = min(len(pose_frames), int(anchor) + 3)
    window = pose_frames[start:end]
    if not window:
        return FULL_BODY_GUIDANCE_MESSAGE

    supported = sum(
        1 for frame in window
        if _has_full_body_front_visual_support(frame)
    )
    required = min(3, len(window))
    if supported >= required:
        return None
    return FULL_BODY_GUIDANCE_MESSAGE


# ---------------------------------------------------------------------
# Visual anchoring (EVENT-DRIVEN, LOCKED)
# ---------------------------------------------------------------------
def _pick_visual_anchor_frame(
    risk_id: str,
    events: Dict[str, Any],
) -> Optional[int]:
    """
    VISUAL ANCHORS:
    - Front Foot Braking Shock -> FFC
    - Knee Brace Failure       -> FFC
    - Trunk Rotation Snap      -> UAH (else Release)
    - Hip-Shoulder Mismatch    -> UAH (else Release)
    - Lateral Trunk Lean       -> Release (else FFC)
    - Foot-Line Deviation      -> FFC + 1
    """
    ffc = _event_frame(events, "ffc")
    bfc = _event_frame(events, "bfc")
    uah = _event_frame(events, "uah")
    rel = _event_frame(events, "release")

    if risk_id in ("front_foot_braking_shock", "knee_brace_failure"):
        return ffc

    if risk_id in ("trunk_rotation_snap", "hip_shoulder_mismatch"):
        return uah if uah is not None else rel

    if risk_id == "lateral_trunk_lean":
        return rel if rel is not None else ffc

    if risk_id == "foot_line_deviation":
        if ffc is not None:
            return ffc + 1
        return rel

    return rel or ffc or bfc or uah


def _visual_window_for_anchor(
    anchor: int,
    fps: float,
    pre_s: float = 0.20,
    post_s: float = 0.20,
) -> Dict[str, int]:
    pre = max(1, int(round(pre_s * max(1.0, fps))))
    post = max(1, int(round(post_s * max(1.0, fps))))
    return {
        "start": max(0, anchor - pre),
        "end": max(0, anchor + post),
    }


def _attach_visual(
    risk: Dict[str, Any],
    *,
    pose_frames: List[Dict[str, Any]],
    video: Dict[str, Any],
    events: Dict[str, Any],
    run_id: Optional[str],
    rear_view_only: bool,
) -> Dict[str, Any]:
    """
    Attach visual evidence using EVENT-derived anchor.
    """
    if not isinstance(risk, dict):
        return risk

    if rear_view_only:
        risk["capture_feedback"] = {
            "view": "rear",
            "message": FULL_BODY_GUIDANCE_MESSAGE,
        }
        risk["visual_unavailable_reason"] = risk["capture_feedback"]["message"]
        return risk

    video_path = video.get("path") or video.get("file_path")
    if not video_path:
        logger.warning("[risk_worker] Missing video path; skipping visual.")
        return risk

    fps = float(video.get("fps") or 25.0)
    rid = str(risk.get("risk_id") or "")

    anchor = _pick_visual_anchor_frame(rid, events)
    if anchor is None:
        logger.warning(f"[risk_worker] No anchor for {rid}; skipping visual.")
        return risk

    anchor = max(0, min(int(anchor), len(pose_frames) - 1))
    window = _visual_window_for_anchor(anchor, fps)

    front_view_block_reason = _front_visual_block_reason(pose_frames, anchor)
    if front_view_block_reason:
        risk["capture_feedback"] = {
            "view": "front",
            "issue": "cropped_or_incomplete",
            "message": front_view_block_reason,
        }
        risk["visual_unavailable_reason"] = front_view_block_reason
        return risk

    strength = float(risk.get("signal_strength", 0.0))
    if strength >= 0.6:
        visual_band = "HIGH"
    elif strength >= 0.3:
        visual_band = "MEDIUM"
    else:
        visual_band = "LOW"

    # -----------------------------------------------------------------
    # Footer load information (SAFE, OPTIONAL)
    # -----------------------------------------------------------------
    load_body: Optional[str] = None
    load_level: Optional[str] = None

    deviation = risk.get("deviation", {})
    band = deviation.get("band")
    load_level = _load_level_from_band(band)

    # Prefer semantic override for known "force-leak / sequencing" risks
    if rid in PRIMARY_LOAD_OVERRIDE:
        load_body = PRIMARY_LOAD_OVERRIDE[rid]
    else:
        impact = risk.get("impact", {})
        primary = impact.get("primary") or []
        if primary:
            load_body = str(primary[0]).lower()

    visual = draw_and_save_visual(
        video_path=video_path,
        frame_idx=anchor,
        risk_id=rid,
        pose_frames=pose_frames,
        visual_confidence=visual_band,
        run_id=run_id,
        load_body=load_body,
        load_level=load_level,
    )

    if visual:
        risk["visual"] = visual
        risk["visual_window"] = window

    return risk


# ---------------------------------------------------------------------
# Percentile mapping (TEMPORARY, PHASE-2)
# ---------------------------------------------------------------------
def _percentile_from_signal_strength(v: float) -> float:
    """
    TEMPORARY mapping until real benchmark distributions are plugged in.
    This preserves monotonicity and allows frontend to render bands.
    """
    if v >= 0.6:
        return 95.0
    if v >= 0.4:
        return 85.0
    if v >= 0.25:
        return 70.0
    return 50.0


# ---------------------------------------------------------------------
# Main worker
# ---------------------------------------------------------------------
def run_risk_worker(
    pose_frames: List[Dict[str, Any]],
    video: Dict[str, Any],
    events: Dict[str, Any],
    action: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    action = action or {}
    fps = float(video.get("fps") or 25.0)

    ffc = _event_frame(events, "ffc")
    bfc = _event_frame(events, "bfc")
    uah = _event_frame(events, "uah")
    rel = _event_frame(events, "release")

    rear_view_only = _is_rear_view_capture(pose_frames)

    raw = [
        _emit(
            compute_front_foot_braking_shock(
                pose_frames, ffc, fps, {}, action=action
            ),
            "front_foot_braking_shock",
        ),
        _emit(
            compute_knee_brace_failure(
                pose_frames, ffc, fps, {}
            ),
            "knee_brace_failure",
        ),
        _emit(
            compute_trunk_rotation_snap(
                pose_frames, ffc, uah, fps, {}
            ),
            "trunk_rotation_snap",
        ),
        _emit(
            compute_hip_shoulder_mismatch(
                pose_frames, ffc, rel, fps, {}
            ),
            "hip_shoulder_mismatch",
        ),
        _emit(
            compute_lateral_trunk_lean(
                pose_frames, bfc, ffc, rel, fps, {}
            ),
            "lateral_trunk_lean",
        ),
        _emit(
            compute_foot_line_deviation(
                pose_frames, bfc, ffc, fps, {}, action=action
            ),
            "foot_line_deviation",
        ),
    ]

    out: List[Dict[str, Any]] = []
    for r in raw:
        percentile = _percentile_from_signal_strength(
            float(r.get("signal_strength", 0.0))
        )

        r = attach_deviation_and_impact(
            r,
            risk_id=r["risk_id"],
            percentile=percentile,
        )

        out.append(
            _attach_visual(
                r,
                pose_frames=pose_frames,
                video=video,
                events=events,
                run_id=run_id,
                rear_view_only=rear_view_only,
            )
        )

    return out
