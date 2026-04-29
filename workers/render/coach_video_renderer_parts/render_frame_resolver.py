from __future__ import annotations

from typing import Any, Dict, Optional, Set

from app.workers.events.timing_constants import render_timing

from .analytics import _event_confidence, _safe_int
from .shared import WEAK_PHASE_METHODS
from .tracks import _track_frame_quality


def _render_event_payload(
    *,
    event: Optional[Dict[str, Any]],
    detected_frame: Optional[int],
    render_frame: int,
    detected_confidence: float,
    render_method: str,
    render_resolved: bool,
) -> Dict[str, Any]:
    payload = dict(event or {})
    payload["frame"] = int(render_frame)
    payload["render_frame"] = int(render_frame)
    payload["render_method"] = str(render_method)
    payload["render_resolved"] = bool(render_resolved)
    if detected_frame is not None:
        payload["detected_frame"] = int(detected_frame)
    if "method" in payload:
        payload["detected_method"] = str(payload.get("method") or "")
    if "confidence" in payload:
        payload["detected_confidence"] = max(0.0, float(payload.get("confidence") or 0.0))
    return payload


def resolve_render_timeline_events(
    *,
    start: int,
    stop: int,
    fps: float,
    events: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if all(
        isinstance(((events or {}).get(key) or {}), dict)
        and "render_frame" in (((events or {}).get(key) or {}))
        for key in ("bfc", "ffc", "release")
        if ((events or {}).get(key) or {}) or key == "release"
    ):
        return dict(events or {})

    timeline = dict(events or {})
    span = max(4, stop - start)
    last = max(start + 3, stop - 1)
    tolerance = int(render_timing(fps)["render_tolerance"])

    raw_bfc = _safe_int(((events or {}).get("bfc") or {}).get("frame"))
    raw_ffc = _safe_int(((events or {}).get("ffc") or {}).get("frame"))
    raw_uah = _safe_int(((events or {}).get("uah") or {}).get("frame"))
    raw_release = _safe_int(((events or {}).get("release") or {}).get("frame"))
    phase_release = raw_uah if raw_uah is not None and raw_uah >= (raw_ffc if raw_ffc is not None else start) else raw_release

    target_bfc = start + int(round(span * 0.32))
    target_ffc = start + int(round(span * 0.58))
    target_release = phase_release if phase_release is not None else start + int(round(span * 0.82))

    weak_ffc = (
        raw_ffc is None
        or raw_ffc < start + 2
        or raw_ffc >= stop - 1
        or _event_confidence(events, "ffc") < 0.30
        or str((((events or {}).get("ffc") or {}).get("method")) or "").strip().lower() in WEAK_PHASE_METHODS
    )
    if not weak_ffc and raw_release is not None and raw_ffc >= raw_release - 1:
        weak_ffc = True

    detected_release = raw_release
    detected_ffc = raw_ffc
    detected_bfc = raw_bfc

    if weak_ffc:
        if detected_ffc is None:
            render_ffc = target_ffc
        else:
            render_ffc = max(detected_ffc - tolerance, min(detected_ffc + tolerance, target_ffc))
    else:
        render_ffc = detected_ffc

    weak_bfc = (
        raw_bfc is None
        or raw_bfc < start + 1
        or raw_bfc >= stop - 2
        or (render_ffc is not None and raw_bfc >= render_ffc)
    )

    if weak_bfc:
        if detected_bfc is None:
            render_bfc = target_bfc
        else:
            render_bfc = max(detected_bfc - tolerance, min(detected_bfc + tolerance, target_bfc))
    else:
        render_bfc = detected_bfc

    if detected_release is None:
        render_release = target_release
        release_render_method = "phase_band_fallback"
        release_resolved = True
    else:
        render_release = detected_release
        release_render_method = "detected_frame"
        release_resolved = False

    if detected_release is not None:
        release_limit = max(start + 2, min(last, int(detected_release)))
        render_release = release_limit
        render_ffc = max(start + 1, min(release_limit - 1, int(render_ffc)))
        render_bfc = max(start + 1, min(render_ffc - 1, int(render_bfc)))
    else:
        render_bfc = max(start + 1, min(last - 2, int(render_bfc)))
        render_ffc = max(render_bfc + 1, min(last - 1, int(render_ffc)))
        render_release = max(render_ffc + 1, min(last, int(render_release)))

    if weak_bfc:
        timeline["bfc"] = _render_event_payload(
            event=(timeline.get("bfc") or {}),
            detected_frame=detected_bfc,
            render_frame=render_bfc,
            detected_confidence=_event_confidence(events, "bfc"),
            render_method="phase_band_fallback",
            render_resolved=True,
        )
    elif detected_bfc is not None:
        timeline["bfc"] = _render_event_payload(
            event=(timeline.get("bfc") or {}),
            detected_frame=detected_bfc,
            render_frame=render_bfc,
            detected_confidence=_event_confidence(events, "bfc"),
            render_method="detected_frame",
            render_resolved=False,
        )

    if weak_ffc:
        timeline["ffc"] = _render_event_payload(
            event=(timeline.get("ffc") or {}),
            detected_frame=detected_ffc,
            render_frame=render_ffc,
            detected_confidence=_event_confidence(events, "ffc"),
            render_method="phase_band_fallback",
            render_resolved=True,
        )
    elif detected_ffc is not None:
        timeline["ffc"] = _render_event_payload(
            event=(timeline.get("ffc") or {}),
            detected_frame=detected_ffc,
            render_frame=render_ffc,
            detected_confidence=_event_confidence(events, "ffc"),
            render_method="detected_frame",
            render_resolved=False,
        )

    timeline["release"] = _render_event_payload(
        event=(timeline.get("release") or {}),
        detected_frame=detected_release,
        render_frame=render_release,
        detected_confidence=_event_confidence(events, "release"),
        render_method=release_render_method,
        render_resolved=release_resolved,
    )

    return timeline


def _is_likely_slow_motion(playback_mode: Optional[Dict[str, Any]]) -> bool:
    return str((playback_mode or {}).get("mode") or "").strip().lower() == "likely_slow_motion"


def _candidate_render_frames_for_event(
    *,
    payload: Dict[str, Any],
    fallback_frame: Optional[int],
    fps: float,
) -> Set[int]:
    frames: Set[int] = set()
    if fallback_frame is not None:
        frames.add(int(fallback_frame))

    detected_frame = _safe_int(payload.get("detected_frame"))
    if detected_frame is not None:
        frames.add(int(detected_frame))

    window = max(2, int(render_timing(fps)["render_tolerance"]))
    center = detected_frame if detected_frame is not None else fallback_frame
    if center is not None:
        for frame_idx in range(int(center) - window, int(center) + window + 1):
            frames.add(int(frame_idx))

    for candidate in payload.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        candidate_frame = _safe_int(candidate.get("frame"))
        if candidate_frame is not None:
            frames.add(int(candidate_frame))
    return frames


def _resolve_low_quality_render_frame(
    *,
    event_key: str,
    payload: Dict[str, Any],
    tracks: Dict[int, Dict[str, Any]],
    fps: float,
    playback_mode: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    render_frame = _safe_int(payload.get("render_frame"))
    if render_frame is None:
        render_frame = _safe_int(payload.get("frame"))
    if render_frame is None:
        return payload

    current_quality = float(_track_frame_quality(tracks, render_frame))
    slow_motion_mode = _is_likely_slow_motion(playback_mode)
    release_like_event = event_key == "release"
    if not slow_motion_mode and (not release_like_event or current_quality >= 0.55):
        return payload
    if slow_motion_mode and current_quality >= 0.62:
        return payload

    detected_frame = _safe_int(payload.get("detected_frame"))
    best_frame = int(render_frame)
    best_score = current_quality

    for candidate_frame in sorted(
        _candidate_render_frames_for_event(
            payload=payload,
            fallback_frame=render_frame,
            fps=fps,
        )
    ):
        quality = float(_track_frame_quality(tracks, candidate_frame))
        if quality <= 0.0:
            continue

        distance_penalty = 0.02 * abs(candidate_frame - (detected_frame if detected_frame is not None else render_frame))
        late_penalty = 0.0
        if release_like_event and detected_frame is not None and candidate_frame > detected_frame:
            late_penalty = 0.03 * float(candidate_frame - detected_frame)
        score = quality - distance_penalty - late_penalty
        if score > best_score + 0.08:
            best_score = score
            best_frame = int(candidate_frame)

    if best_frame == int(render_frame):
        return payload

    resolved = dict(payload)
    resolved["frame"] = int(best_frame)
    resolved["render_frame"] = int(best_frame)
    resolved["render_method"] = "quality_resolved_neighbor"
    resolved["render_resolved"] = True
    return resolved


def attach_render_quality_metadata(
    *,
    events: Optional[Dict[str, Any]],
    tracks: Dict[int, Dict[str, Any]],
    fps: float = 30.0,
    playback_mode: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    timeline = dict(events or {})
    quality_values = []
    for key in ("bfc", "ffc", "uah", "release"):
        event = timeline.get(key)
        if not isinstance(event, dict):
            continue
        payload = _resolve_low_quality_render_frame(
            event_key=key,
            payload=dict(event),
            tracks=tracks,
            fps=fps,
            playback_mode=playback_mode,
        )
        render_frame = _safe_int(payload.get("render_frame"))
        detected_frame = _safe_int(payload.get("detected_frame"))
        if render_frame is None:
            render_frame = _safe_int(payload.get("frame"))
        render_quality = _track_frame_quality(tracks, render_frame) if render_frame is not None else 0.0
        payload["render_quality"] = round(float(render_quality), 3)
        payload["render_confidence"] = payload["render_quality"]
        quality_values.append(float(render_quality))
        if detected_frame is not None:
            payload["detected_frame_quality"] = round(
                float(_track_frame_quality(tracks, detected_frame)),
                3,
            )
        timeline[key] = payload

    timeline["render_quality"] = {
        "overall": round(float(sum(quality_values) / max(1, len(quality_values))), 3) if quality_values else 0.0,
        "event_count": len(quality_values),
    }
    return timeline
