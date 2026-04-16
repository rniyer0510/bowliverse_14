from __future__ import annotations

from typing import Any, Dict, List, Optional

from .render_constants import (
    FEATURE_TO_RENDER_LABEL,
    FFC_DEPENDENT_RISKS,
    LEFT_ANKLE, LEFT_FOOT_INDEX, LEFT_HIP, LEFT_KNEE,
    RIGHT_ANKLE, RIGHT_FOOT_INDEX, RIGHT_HEEL, RIGHT_HIP, RIGHT_KNEE,
    MIN_EVENT_CHAIN_QUALITY, MIN_FFC_STORY_CONFIDENCE,
)

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
    return (
        _event_confidence(events, "ffc") >= MIN_FFC_STORY_CONFIDENCE
        and _event_chain_quality(events) >= MIN_EVENT_CHAIN_QUALITY
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


def _story_feature_labels(report_story: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(report_story, dict):
        return []
    labels: List[str] = []
    watch_focus = report_story.get("watch_focus") or {}
    watch_label = str(watch_focus.get("label") or "").strip()
    if watch_label:
        labels.append(watch_label)
    for item in report_story.get("key_metrics") or []:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip()
        label = str(item.get("label") or "").strip()
        resolved = FEATURE_TO_RENDER_LABEL.get(key) or label
        if resolved and resolved not in labels:
            labels.append(resolved)
    return labels


def _positive_recap_lines(report_story: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(report_story, dict):
        return []
    theme = str(report_story.get("theme") or "").strip()
    if theme not in {"working_pattern", "good_base"}:
        return []

    lines: List[str] = []
    watch_focus = report_story.get("watch_focus") or {}
    watch_label = str(watch_focus.get("label") or "").strip()
    if watch_label:
        lines.append(f"Keep watching {watch_label}")

    positive_by_key = {
        "upper_body_alignment": "Upper body stays aligned",
        "lower_body_alignment": "Lower body stays aligned",
        "whole_body_alignment": "Action shape stays connected",
        "momentum_forward": "Carries forward well",
        "front_leg_support": "Front leg support looks steady",
        "trunk_lean": "Body stays fairly tall",
        "upper_body_opening": "Top half stays in order",
        "action_flow": "Action carries through well",
        "front_foot_line": "Feet stay balanced and in line",
    }

    for item in report_story.get("key_metrics") or []:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip()
        label = positive_by_key.get(key)
        if not label or label in lines:
            continue
        lines.append(label)
        if len(lines) >= 4:
            break

    if not lines:
        headline = str(report_story.get("headline") or "").strip()
        if "strong working pattern" in headline.lower():
            lines = ["Strong working pattern", "Keep repeating this shape"]
        else:
            lines = ["Action has a good base", "Keep repeating this shape"]
    return lines[:4]


def _story_risk_for_phase(
    report_story: Optional[Dict[str, Any]],
    *,
    phase_key: str,
    events: Optional[Dict[str, Any]],
) -> Optional[str]:
    if not isinstance(report_story, dict):
        return None
    hero_risk_id = str(report_story.get("hero_risk_id") or "").strip()
    if _risk_supported_for_phase(hero_risk_id, phase_key=phase_key, events=events):
        return hero_risk_id

    watch_focus = report_story.get("watch_focus") or {}
    watch_key = str(watch_focus.get("key") or "").strip()
    mapped = {
        "front_leg_support": "knee_brace_failure",
        "front_foot_line": "foot_line_deviation",
        "trunk_lean": "lateral_trunk_lean",
        "upper_body_opening": "hip_shoulder_mismatch",
        "action_flow": "front_foot_braking_shock",
        "trunk_rotation_load": "trunk_rotation_snap",
    }.get(watch_key)
    if _risk_supported_for_phase(mapped, phase_key=phase_key, events=events):
        return mapped
    return None


def _format_action_label(action: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(action, dict):
        return None
    raw = str(action.get("action") or "").strip()
    if not raw or raw.upper() == "UNKNOWN":
        return None
    return raw.replace("_", " ").title()


def _front_leg_joints(hand: Optional[str]) -> Tuple[int, int, int]:
    is_left_handed = str(hand or "R").upper().startswith("L")
    if is_left_handed:
        return RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE
    return LEFT_HIP, LEFT_KNEE, LEFT_ANKLE


def _foot_indices(hand: Optional[str]) -> Tuple[int, int, int]:
    is_left_handed = str(hand or "R").upper().startswith("L")
    if is_left_handed:
        return RIGHT_FOOT_INDEX, RIGHT_HEEL, LEFT_FOOT_INDEX
    return LEFT_FOOT_INDEX, LEFT_HEEL, RIGHT_FOOT_INDEX


