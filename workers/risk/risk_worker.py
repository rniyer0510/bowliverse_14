from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from app.common.logger import get_logger

from app.workers.risk.front_foot_braking import compute_front_foot_braking_shock
from app.workers.risk.knee_brace_failure import compute_knee_brace_failure
from app.workers.risk.trunk_rotation_snap import compute_trunk_rotation_snap
from app.workers.risk.hip_shoulder_mismatch import compute_hip_shoulder_mismatch
from app.workers.risk.lateral_trunk_lean import compute_lateral_trunk_lean
from app.workers.risk.foot_line_deviation import compute_foot_line_deviation

from app.workers.risk.benchmarks import attach_deviation_and_impact

logger = get_logger(__name__)

ENABLE_SYNTHETIC_BENCHMARK_PERCENTILES = (
    os.getenv("ACTIONLAB_ENABLE_SYNTHETIC_BENCHMARK_PERCENTILES", "").strip().lower()
    == "true"
)

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


def _event_value(events: Dict[str, Any], key: str, field: str, default: Any = None) -> Any:
    obj = events.get(key) or {}
    if not isinstance(obj, dict):
        return default
    return obj.get(field, default)


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


def _benchmark_percentile(signal_strength: float) -> Optional[float]:
    if not ENABLE_SYNTHETIC_BENCHMARK_PERCENTILES:
        return None
    return _percentile_from_signal_strength(signal_strength)


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
        percentile = _benchmark_percentile(
            float(r.get("signal_strength", 0.0))
        )

        r = attach_deviation_and_impact(
            r,
            risk_id=r["risk_id"],
            percentile=percentile,
        )
        out.append(r)

    return out
