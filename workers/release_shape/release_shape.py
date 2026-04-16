from typing import Any, Dict, Iterable, Optional

from .kinematics import compute_release_shape_metrics

_SUPPORTED_CATEGORIES = (
    {"key": "standard", "label": "Standard"},
    {"key": "round_arm", "label": "Round Arm"},
)


def _normalized_hand(hand: Optional[str]) -> Optional[str]:
    text = str(hand or "").strip().lower()
    return text or None


def _normalized_action(action: Optional[Dict[str, Any]]) -> Optional[str]:
    raw = str((action or {}).get("action") or "").strip().lower()
    return raw or None


def _release_frame(events: Optional[Dict[str, Any]]) -> Optional[int]:
    frame = ((events or {}).get("release") or {}).get("frame")
    return int(frame) if isinstance(frame, (int, float)) else None


def _release_confidence(events: Optional[Dict[str, Any]]) -> float:
    confidence = ((events or {}).get("release") or {}).get("confidence")
    if isinstance(confidence, (int, float)):
        return round(float(confidence), 2)
    return 0.0


def _trusted(metrics: Dict[str, Any], key: str) -> bool:
    fields = metrics.get("trusted_fields")
    if isinstance(fields, dict):
        return bool(fields.get(key))
    return False


def build_release_shape_skeleton(
    *,
    pose_frames: Optional[Iterable[Dict[str, Any]]] = None,
    events: Optional[Dict[str, Any]] = None,
    hand: Optional[str] = None,
    action: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    frame_count = None
    frames = list(pose_frames) if pose_frames is not None else []
    if pose_frames is not None:
        try:
            frame_count = len(pose_frames)  # type: ignore[arg-type]
        except TypeError:
            frame_count = None
    metrics = compute_release_shape_metrics(
        pose_frames=frames,
        events=events,
        hand=hand,
    )
    category_key = metrics.get("category_key")
    category_label = "Round Arm" if category_key == "round_arm" else "Standard" if category_key == "standard" else None

    return {
        "version": "release_shape_v1",
        "available": category_key is not None,
        "category": {
            "key": category_key,
            "label": category_label,
        },
        "supported_categories": list(_SUPPORTED_CATEGORIES),
        "release_geometry": {
            "height_m": None,
            "height_ratio": metrics.get("height_ratio") if _trusted(metrics, "height_ratio") else None,
            "angle_deg": metrics.get("angle_deg") if _trusted(metrics, "angle_deg") else None,
            "arc_deg": metrics.get("arc_deg") if _trusted(metrics, "arc_deg") else None,
        },
        "trusted_fields": dict(metrics.get("trusted_fields") or {}),
        "drift_from_usual": {
            "available": False,
            "status": "not_available",
            "summary": None,
            "delta_deg": None,
        },
        "confidence": float(metrics.get("confidence") or 0.0),
        "source": {
            "release_frame": metrics.get("release_frame", _release_frame(events)),
            "release_confidence": _release_confidence(events),
            "event_chain_quality": metrics.get("event_chain_quality"),
            "hand": _normalized_hand(hand),
            "action_type": _normalized_action(action),
            "pose_frame_count": frame_count,
        },
        "reason": str(metrics.get("reason") or "release_shape_not_computed_yet"),
    }
