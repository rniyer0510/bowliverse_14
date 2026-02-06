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
# This is NOT physics, not a correction cue — just a safe, honest label.
# ---------------------------------------------------------------------
PRIMARY_LOAD_OVERRIDE: Dict[str, str] = {
    # Foot-line deviation loads adductors/groin first (knee is downstream/secondary)
    "foot_line_deviation": "groin",
    # Sequencing/torso risks: keep these broad and non-prescriptive
    "hip_shoulder_mismatch": "hip",
    "trunk_rotation_snap": "lower back",
    "lateral_trunk_lean": "lower back",
}

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
    - Hip–Shoulder Mismatch    -> UAH (else Release)
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
) -> Dict[str, Any]:
    """
    Attach visual evidence using EVENT-derived anchor.
    """
    if not isinstance(risk, dict):
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
            )
        )

    return out

