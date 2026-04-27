from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

_LEG_RISKS = {"knee_brace_failure", "foot_line_deviation", "front_foot_braking_shock"}
_TRUNK_RISKS = {"lateral_trunk_lean", "trunk_rotation_snap", "hip_shoulder_mismatch"}
_RELEASE_RISKS = _TRUNK_RISKS | {"front_foot_braking_shock"}
_REGION_LABELS = {
    "groin": "Groin",
    "knee": "Knee",
    "shin": "Shin",
    "upper_trunk": "Upper trunk",
    "side_trunk": "Side trunk",
    "lumbar": "Lumbar",
}


def _risk_weight(risk: Optional[Dict[str, Any]]) -> float:
    if not isinstance(risk, dict):
        return 0.0
    return max(0.0, float(risk.get("signal_strength") or 0.0)) * max(0.0, float(risk.get("confidence") or 0.0))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _event_quality(events: Optional[Dict[str, Any]], key: str) -> float:
    event = (events or {}).get(key) or {}
    confidence = event.get("confidence")
    if isinstance(confidence, (int, float)):
        return max(0.0, float(confidence))
    return 1.0 if _safe_int(event.get("frame")) is not None else 0.0


def _supports_ffc_story(events: Optional[Dict[str, Any]]) -> bool:
    chain = ((events or {}).get("event_chain") or {}).get("quality")
    chain_quality = max(0.0, float(chain)) if isinstance(chain, (int, float)) else 1.0
    method = str((((events or {}).get("ffc") or {}).get("method")) or "").strip()
    return _event_quality(events, "ffc") >= 0.35 and (
        chain_quality >= 0.20 or method == "render_phase_fallback"
    )


def _risk_supported_for_phase(
    risk_id: Optional[str],
    *,
    phase_key: str,
    events: Optional[Dict[str, Any]],
) -> bool:
    if not risk_id:
        return False
    if phase_key == "ffc":
        return risk_id in _LEG_RISKS and _supports_ffc_story(events)
    if phase_key == "release":
        return risk_id in _RELEASE_RISKS
    return False


def _root_cause_anchor_risk_id(
    root_cause: Optional[Dict[str, Any]],
    *,
    phase_key: str,
) -> Optional[str]:
    guidance = ((root_cause or {}).get("renderer_guidance") or {})
    anchor_risk_ids = guidance.get("anchor_risk_ids") or {}
    if not isinstance(anchor_risk_ids, dict):
        return None
    risk_id = str(anchor_risk_ids.get(phase_key) or "").strip()
    if not risk_id:
        return None
    if phase_key == "ffc" and risk_id in _LEG_RISKS:
        return risk_id
    if phase_key == "release" and risk_id in _RELEASE_RISKS:
        return risk_id
    return None


def _root_cause_phase_target(
    root_cause: Optional[Dict[str, Any]],
    *,
    phase_key: str,
) -> Optional[Dict[str, Any]]:
    guidance = ((root_cause or {}).get("renderer_guidance") or {})
    phase_targets = guidance.get("phase_targets") or {}
    if not isinstance(phase_targets, dict):
        return None
    target = phase_targets.get(phase_key)
    return target if isinstance(target, dict) else None


def _root_cause_controls_renderer(root_cause: Optional[Dict[str, Any]]) -> bool:
    status = str((root_cause or {}).get("status") or "").strip().lower()
    return status in {"clear", "holdback", "no_clear_problem", "not_interpretable"}


def _story_risk_for_phase(
    report_story: Optional[Dict[str, Any]],
    *,
    phase_key: str,
    events: Optional[Dict[str, Any]],
    root_cause: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    phase_target = _root_cause_phase_target(root_cause, phase_key=phase_key)
    if isinstance(phase_target, dict):
        phase_target_risk = str(phase_target.get("risk_id") or "").strip()
        if _risk_supported_for_phase(phase_target_risk, phase_key=phase_key, events=events):
            return phase_target_risk
    root_cause_risk = _root_cause_anchor_risk_id(root_cause, phase_key=phase_key)
    if root_cause_risk:
        if phase_key == "ffc" and _supports_ffc_story(events):
            return root_cause_risk
        if phase_key == "release":
            return root_cause_risk
    if _root_cause_controls_renderer(root_cause):
        return None
    if not isinstance(report_story, dict):
        return None
    hero = str(report_story.get("hero_risk_id") or "").strip()
    if hero:
        if phase_key == "ffc" and hero in _LEG_RISKS and _supports_ffc_story(events):
            return hero
        if phase_key == "release" and hero in _RELEASE_RISKS:
            return hero
    watch_key = str(((report_story.get("watch_focus") or {}).get("key")) or "").strip()
    mapping = {
        "front_leg_support": "knee_brace_failure",
        "front_foot_line": "foot_line_deviation",
        "trunk_lean": "lateral_trunk_lean",
        "upper_body_opening": "hip_shoulder_mismatch",
        "action_flow": "front_foot_braking_shock",
        "trunk_rotation_load": "trunk_rotation_snap",
    }
    mapped = mapping.get(watch_key)
    if phase_key == "ffc" and mapped in _LEG_RISKS and _supports_ffc_story(events):
        return mapped
    if phase_key == "release" and mapped in _RELEASE_RISKS:
        return mapped
    return None


def _preferred_ffc_cue_risk_id(
    risk_by_id: Dict[str, Dict[str, Any]],
    *,
    report_story: Optional[Dict[str, Any]],
    events: Optional[Dict[str, Any]],
    root_cause: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    preferred = _story_risk_for_phase(
        report_story,
        phase_key="ffc",
        events=events,
        root_cause=root_cause,
    )
    if preferred:
        return preferred
    if _root_cause_controls_renderer(root_cause):
        return None
    if not _supports_ffc_story(events):
        return None
    ranked = sorted(((rid, _risk_weight(risk_by_id.get(rid))) for rid in _LEG_RISKS), key=lambda item: item[1], reverse=True)
    return next((rid for rid, weight in ranked if weight > 0.0), None)


def _release_hotspot_risk_id(
    risk_by_id: Dict[str, Dict[str, Any]],
    *,
    events: Optional[Dict[str, Any]],
    report_story: Optional[Dict[str, Any]],
    root_cause: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    preferred = _story_risk_for_phase(
        report_story,
        phase_key="release",
        events=events,
        root_cause=root_cause,
    )
    if preferred:
        return preferred
    if _root_cause_controls_renderer(root_cause):
        return None
    ranked = sorted(((rid, _risk_weight(risk_by_id.get(rid))) for rid in _RELEASE_RISKS), key=lambda item: item[1], reverse=True)
    return next((rid for rid, weight in ranked if weight > 0.0), None)


def _body_family(risk_id: Optional[str]) -> Optional[str]:
    if risk_id in _LEG_RISKS:
        return "leg"
    if risk_id in _TRUNK_RISKS:
        return "trunk"
    return None


def _load_watch_label(risk_id: Optional[str]) -> Optional[str]:
    if risk_id in _LEG_RISKS:
        return "Front knee / leg chain"
    if risk_id in _TRUNK_RISKS:
        return "Lower back / side trunk"
    return None


def _summary_symptom_text(
    risk_by_id: Dict[str, Dict[str, Any]],
    *,
    events: Optional[Dict[str, Any]] = None,
    report_story: Optional[Dict[str, Any]] = None,
    root_cause: Optional[Dict[str, Any]] = None,
) -> str:
    root_cause_guidance = ((root_cause or {}).get("renderer_guidance") or {})
    root_cause_simple_text = str(root_cause_guidance.get("simple_symptom_text") or "").strip()
    if root_cause_simple_text:
        return root_cause_simple_text
    root_cause_text = str(root_cause_guidance.get("symptom_text") or "").strip()
    if root_cause_text:
        return root_cause_text
    root_cause_status = str((root_cause or {}).get("status") or "").strip().lower()
    if root_cause_status == "not_interpretable":
        return "Unable to verify this clip yet."
    if root_cause_status == "no_clear_problem":
        return "Action stays connected through landing and release."
    if isinstance(report_story, dict) and str(report_story.get("theme") or "") in {"working_pattern", "good_base"}:
        label = str(((report_story.get("watch_focus") or {}).get("label")) or "").strip()
        if label:
            return f"Keep watching {label}"
        return "Action has a usable base, but one part still needs watching."
    release_risk_id = _release_hotspot_risk_id(
        risk_by_id,
        events=events,
        report_story=report_story,
        root_cause=root_cause,
    )
    if release_risk_id == "lateral_trunk_lean":
        return "Body falls away at release"
    if release_risk_id == "trunk_rotation_snap":
        return "Body rotates sharply at release"
    if release_risk_id == "hip_shoulder_mismatch":
        pattern = str((((risk_by_id.get("hip_shoulder_mismatch") or {}).get("debug")) or {}).get("sequence_pattern") or "").lower()
        return "Shoulders start too soon" if pattern == "shoulders_lead" else ("Hips lead too far" if pattern == "hips_lead" else "Hips and shoulders drift apart")
    ffc_risk_id = _preferred_ffc_cue_risk_id(
        risk_by_id,
        report_story=report_story,
        events=events,
        root_cause=root_cause,
    )
    if ffc_risk_id == "knee_brace_failure":
        return "Front leg softens at landing"
    if ffc_risk_id == "foot_line_deviation":
        return "Front foot lands across line"
    return "No clear coaching cue from this clip."


def _summary_symptom_title(
    *,
    report_story: Optional[Dict[str, Any]] = None,
    root_cause: Optional[Dict[str, Any]] = None,
) -> str:
    root_cause_status = str((root_cause or {}).get("status") or "").strip().lower()
    if root_cause_status == "not_interpretable":
        return "Unable To Verify"
    if root_cause_status == "no_clear_problem":
        return "What Is Working"
    if isinstance(report_story, dict) and str(report_story.get("theme") or "") in {"working_pattern", "good_base"}:
        label = str(((report_story.get("watch_focus") or {}).get("label")) or "").strip()
        return "What Is Working" if label else "What To Notice"
    return "What To Notice"


def _summary_load_watch_text(
    risk_by_id: Dict[str, Dict[str, Any]],
    *,
    events: Optional[Dict[str, Any]] = None,
    report_story: Optional[Dict[str, Any]] = None,
    root_cause: Optional[Dict[str, Any]] = None,
) -> str:
    root_cause_guidance = ((root_cause or {}).get("renderer_guidance") or {})
    root_cause_simple_text = str(root_cause_guidance.get("simple_load_watch_text") or "").strip()
    if root_cause_simple_text:
        return root_cause_simple_text
    root_cause_text = str(root_cause_guidance.get("load_watch_text") or "").strip()
    if root_cause_text:
        return root_cause_text
    root_cause_status = str((root_cause or {}).get("status") or "").strip().lower()
    if root_cause_status == "not_interpretable":
        return "Retake from side-on with the full body and release in frame."
    if root_cause_status == "no_clear_problem":
        return "No one area is taking too much load."
    primary_risk_id = _story_risk_for_phase(
        report_story,
        phase_key="ffc",
        events=events,
        root_cause=root_cause,
    ) or _story_risk_for_phase(
        report_story,
        phase_key="release",
        events=events,
        root_cause=root_cause,
    )
    ranked = sorted(risk_by_id.items(), key=lambda item: _risk_weight(item[1]), reverse=True)
    primary_risk_id = primary_risk_id or (ranked[0][0] if ranked else None)
    primary_label = _load_watch_label(primary_risk_id)
    if not primary_label:
        return "Need a clearer release view to read load."
    primary_family = _body_family(primary_risk_id)
    secondary = next((_load_watch_label(rid) for rid, risk in ranked if rid != primary_risk_id and _body_family(rid) != primary_family and _risk_weight(risk) >= 0.45), None)
    return primary_label if not secondary else f"{primary_label}\n{secondary}"


def _summary_load_watch_title(
    *,
    root_cause: Optional[Dict[str, Any]] = None,
) -> str:
    root_cause_status = str((root_cause or {}).get("status") or "").strip().lower()
    if root_cause_status == "not_interpretable":
        return "Retake This Video"
    if root_cause_status == "no_clear_problem":
        return "Load Stays Shared"
    return "Works Harder Here"


def _mix(a: Tuple[int, int], b: Tuple[int, int], t: float) -> Tuple[int, int]:
    return (int(round(a[0] + (b[0] - a[0]) * t)), int(round(a[1] + (b[1] - a[1]) * t)))


def _dist(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return math.hypot(float(a[0] - b[0]), float(a[1] - b[1]))


def _line_distance(point: Tuple[int, int], start: Tuple[int, int], end: Tuple[int, int]) -> float:
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    denom = math.hypot(dx, dy)
    if denom <= 1e-6:
        return 0.0
    return abs(dy * point[0] - dx * point[1] + end[0] * start[1] - end[1] * start[0]) / denom


def _direction(center: Tuple[int, int], body_mid: Tuple[int, int]) -> Tuple[float, float]:
    dx = float(center[0] - body_mid[0])
    dy = float(center[1] - body_mid[1])
    mag = math.hypot(dx, dy)
    if mag <= 1e-6:
        return (1.0, -0.25)
    return (dx / mag, dy / mag)


def _point_at(tracks: Dict[int, Dict[str, Any]], joint_idx: int, frame_idx: int) -> Optional[Tuple[int, int]]:
    raw = (tracks.get(joint_idx) or {}).get("raw") or []
    return raw[frame_idx] if 0 <= frame_idx < len(raw) else None


def _front_leg_joints(hand: Optional[str]) -> Tuple[int, int, int]:
    return (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE) if str(hand or "R").upper().startswith("L") else (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)


def _region(region_key: str, center: Tuple[int, int], weight: float, body_mid: Tuple[int, int]) -> Dict[str, Any]:
    return {
        "region_key": region_key,
        "label": _REGION_LABELS[region_key],
        "center": center,
        "direction": _direction(center, body_mid),
        "weight": max(0.0, min(1.0, weight)),
    }


def _load_hotspot_regions(*, tracks: Dict[int, Dict[str, Any]], frame_idx: int, hand: Optional[str], risk_id: Optional[str], risk_by_id: Optional[Dict[str, Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    risk_by_id = risk_by_id or {}
    left_shoulder = _point_at(tracks, LEFT_SHOULDER, frame_idx)
    right_shoulder = _point_at(tracks, RIGHT_SHOULDER, frame_idx)
    left_hip = _point_at(tracks, LEFT_HIP, frame_idx)
    right_hip = _point_at(tracks, RIGHT_HIP, frame_idx)
    if not (left_shoulder and right_shoulder and left_hip and right_hip):
        return []
    body_mid = _mix(_mix(left_shoulder, right_shoulder, 0.5), _mix(left_hip, right_hip, 0.5), 0.5)

    if risk_id in _LEG_RISKS:
        hip_idx, knee_idx, ankle_idx = _front_leg_joints(hand)
        hip = _point_at(tracks, hip_idx, frame_idx)
        knee = _point_at(tracks, knee_idx, frame_idx)
        ankle = _point_at(tracks, ankle_idx, frame_idx)
        if not (hip and knee and ankle):
            return []
        leg_len = max(1.0, _dist(hip, ankle))
        collapse = min(1.0, _line_distance(knee, hip, ankle) / leg_len * 2.8)
        knee_brace = _risk_weight(risk_by_id.get("knee_brace_failure"))
        foot_line = _risk_weight(risk_by_id.get("foot_line_deviation"))
        braking = _risk_weight(risk_by_id.get("front_foot_braking_shock"))
        if risk_id == "knee_brace_failure":
            groin_w = _clamp01(max(collapse * 0.72, foot_line * 0.48, knee_brace * 0.42))
            knee_w = _clamp01(max(knee_brace, collapse * 0.78, braking * 0.42))
            shin_w = _clamp01(max(braking * 0.78, knee_brace * 0.46, foot_line * 0.26))
        elif risk_id == "foot_line_deviation":
            groin_w = _clamp01(max(foot_line, collapse * 0.82))
            knee_w = _clamp01(max(foot_line * 0.62, collapse * 0.52, knee_brace * 0.28))
            shin_w = _clamp01(max(foot_line * 0.34, braking * 0.22))
        else:
            groin_w = _clamp01(max(collapse * 0.38, foot_line * 0.28, braking * 0.16))
            knee_w = _clamp01(max(braking * 0.62, knee_brace * 0.44))
            shin_w = _clamp01(max(braking, knee_brace * 0.34, foot_line * 0.18))
        return [
            _region("groin", _mix(hip, knee, 0.28), groin_w, body_mid),
            _region("knee", knee, knee_w, body_mid),
            _region("shin", _mix(knee, ankle, 0.54), shin_w, body_mid),
        ]

    if risk_id in _TRUNK_RISKS:
        release_left = str(hand or "R").upper().startswith("L")
        side_shoulder = left_shoulder if release_left else right_shoulder
        side_hip = left_hip if release_left else right_hip
        upper_trunk = _mix(side_shoulder, side_hip, 0.26)
        side_trunk = _mix(side_shoulder, side_hip, 0.5)
        lumbar = _mix(left_hip, right_hip, 0.5)
        lean = _risk_weight(risk_by_id.get("lateral_trunk_lean"))
        mismatch = _risk_weight(risk_by_id.get("hip_shoulder_mismatch"))
        rotation = _risk_weight(risk_by_id.get("trunk_rotation_snap"))
        pattern = str((((risk_by_id.get("hip_shoulder_mismatch") or {}).get("debug")) or {}).get("sequence_pattern") or "").lower()
        if risk_id == "lateral_trunk_lean":
            upper_w = _clamp01(max(lean * 0.36, mismatch * 0.30))
            side_w = _clamp01(max(lean, mismatch * 0.42, rotation * 0.18))
            lumbar_w = _clamp01(max(lean * 0.42, rotation * 0.34, mismatch * 0.28))
        elif risk_id == "trunk_rotation_snap":
            upper_w = _clamp01(max(rotation * 0.28, mismatch * 0.26))
            side_w = _clamp01(max(rotation * 0.34, mismatch * 0.22, lean * 0.20))
            lumbar_w = _clamp01(max(rotation, mismatch * 0.58, lean * 0.32))
        else:
            if pattern == "shoulders_lead":
                upper_w = _clamp01(max(mismatch, lean * 0.30))
                side_w = _clamp01(max(mismatch, lean * 0.34))
                lumbar_w = _clamp01(max(mismatch * 0.42, rotation * 0.38))
            elif pattern == "hips_lead":
                upper_w = _clamp01(max(mismatch * 0.34, lean * 0.24))
                side_w = _clamp01(max(mismatch * 0.36, lean * 0.28))
                lumbar_w = _clamp01(max(mismatch, rotation * 0.56))
            else:
                upper_w = _clamp01(max(mismatch * 0.82, lean * 0.28))
                side_w = _clamp01(max(mismatch * 0.68, lean * 0.34))
                lumbar_w = _clamp01(max(mismatch * 0.74, rotation * 0.44))
        return [
            _region("upper_trunk", upper_trunk, upper_w, body_mid),
            _region("side_trunk", side_trunk, side_w, body_mid),
            _region("lumbar", lumbar, lumbar_w, body_mid),
        ]

    return []
