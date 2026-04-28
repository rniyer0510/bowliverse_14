from __future__ import annotations
from .shared import *

def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None
def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None
def _risk_lookup(risks: Optional[List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for risk in risks or []:
        if not isinstance(risk, dict):
            continue
        risk_id = str(risk.get("risk_id") or "").strip()
        if risk_id:
            lookup[risk_id] = risk
    return lookup
def _risk_weight(risk: Optional[Dict[str, Any]]) -> float:
    if not isinstance(risk, dict):
        return 0.0
    signal = float(risk.get("signal_strength") or 0.0)
    confidence = float(risk.get("confidence") or 0.0)
    return max(0.0, signal) * max(0.0, confidence)
def _event_confidence(events: Optional[Dict[str, Any]], key: str) -> float:
    event = (events or {}).get(key) or {}
    confidence = _safe_float(event.get("confidence"))
    if confidence is None:
        return 1.0 if _safe_int(event.get("frame")) is not None else 0.0
    return max(0.0, confidence)
def _event_chain_quality(events: Optional[Dict[str, Any]]) -> float:
    quality = _safe_float((((events or {}).get("event_chain") or {}).get("quality")))
    if quality is None:
        return 1.0
    return max(0.0, quality)
def _supports_ffc_story(events: Optional[Dict[str, Any]]) -> bool:
    method = str((((events or {}).get("ffc") or {}).get("method")) or "").strip()
    return _event_confidence(events, "ffc") >= MIN_FFC_STORY_CONFIDENCE and (
        _event_chain_quality(events) >= MIN_EVENT_CHAIN_QUALITY
        or method == "render_phase_fallback"
    )
def _speed_display_text(speed: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(speed, dict):
        return None
    if not speed.get("available"):
        return None
    display = str(speed.get("display") or "").strip()
    if not display:
        return None
    return display
def _risk_supported_for_phase(
    risk_id: Optional[str],
    *,
    phase_key: str,
    events: Optional[Dict[str, Any]],
) -> bool:
    if not risk_id:
        return False
    if phase_key == "ffc":
        return risk_id in FFC_DEPENDENT_RISKS and _supports_ffc_story(events)
    if phase_key == "release":
        return risk_id in {
            "lateral_trunk_lean",
            "hip_shoulder_mismatch",
            "trunk_rotation_snap",
            "front_foot_braking_shock",
        }
    return False
