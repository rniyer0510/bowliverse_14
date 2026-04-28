from __future__ import annotations
from .shared import *
from .analytics import _safe_int, _safe_float, _event_confidence
from .tracks import *

def _phase_cut_points(
    *,
    start: int,
    stop: int,
    events: Optional[Dict[str, Any]],
) -> Tuple[int, int, int]:
    last = max(start + 3, stop - 1)
    span = max(4, stop - start)
    render_events = _render_timeline_events(start=start, stop=stop, events=events)
    bfc = _safe_int(((render_events or {}).get("bfc") or {}).get("frame"))
    ffc = _safe_int(((render_events or {}).get("ffc") or {}).get("frame"))
    uah = _safe_int(((render_events or {}).get("uah") or {}).get("frame"))
    release = _safe_int(((render_events or {}).get("release") or {}).get("frame"))
    phase_release = uah if uah is not None and uah >= (ffc if ffc is not None else start) else release

    cp1 = bfc if bfc is not None else start + int(round(span * 0.32))
    cp2 = ffc if ffc is not None else start + int(round(span * 0.58))
    cp3 = phase_release if phase_release is not None else start + int(round(span * 0.82))

    cp1 = max(start + 1, min(last - 2, cp1))
    cp2 = max(cp1 + 1, min(last - 1, cp2))
    cp3 = max(cp2 + 1, min(last, cp3))
    return cp1, cp2, cp3
def _event_method(events: Optional[Dict[str, Any]], key: str) -> str:
    event = (events or {}).get(key) or {}
    return str(event.get("method") or "").strip().lower()
def _tracked_joint_quality(
    pose_frames: List[Dict[str, Any]],
    frame_idx: int,
) -> float:
    if frame_idx < 0 or frame_idx >= len(pose_frames):
        return 0.0
    landmarks = (pose_frames[frame_idx] or {}).get("landmarks") or []
    if not isinstance(landmarks, list) or not landmarks:
        return 0.0
    visible = 0
    for joint_idx in TRACKED_JOINTS:
        visibility = _safe_landmark_value(landmarks, joint_idx, "visibility") or 0.0
        if visibility >= MIN_VISIBILITY:
            visible += 1
    return float(visible) / float(max(1, len(TRACKED_JOINTS)))
def _safe_landmark_value(landmarks: List[Dict[str, Any]], idx: int, key: str) -> Optional[float]:
    try:
        value = ((landmarks[idx] or {}).get(key))
    except Exception:
        return None
    return _safe_float(value)
def _should_draw_skeleton_frame(
    *,
    pose_frames: List[Dict[str, Any]],
    frame_idx: int,
    events: Optional[Dict[str, Any]],
    fps: float,
) -> bool:
    quality = _tracked_joint_quality(pose_frames, frame_idx)
    if quality < MIN_TRACK_QUALITY:
        return False
    release_frame = _safe_int(((events or {}).get("release") or {}).get("frame"))
    if release_frame is None or frame_idx <= release_frame:
        return True
    post_release_tail = max(1, int(round(float(fps) * POST_RELEASE_SKELETON_TAIL_SECONDS)))
    if frame_idx > release_frame + post_release_tail:
        return False
    return quality >= MIN_POST_RELEASE_TRACK_QUALITY
def _render_timeline_events(
    *,
    start: int,
    stop: int,
    events: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    timeline = dict(events or {})
    span = max(4, stop - start)
    last = max(start + 3, stop - 1)

    raw_bfc = _safe_int(((events or {}).get("bfc") or {}).get("frame"))
    raw_ffc = _safe_int(((events or {}).get("ffc") or {}).get("frame"))
    raw_uah = _safe_int(((events or {}).get("uah") or {}).get("frame"))
    raw_release = _safe_int(((events or {}).get("release") or {}).get("frame"))
    phase_release = raw_uah if raw_uah is not None and raw_uah >= (raw_ffc if raw_ffc is not None else start) else raw_release
    fallback_bfc = start + int(round(span * 0.32))
    fallback_ffc = start + int(round(span * 0.58))
    fallback_release = phase_release if phase_release is not None else start + int(round(span * 0.82))

    weak_ffc = (
        raw_ffc is None
        or raw_ffc < start + 2
        or raw_ffc >= stop - 1
        or _event_confidence(events, "ffc") < 0.30
        or _event_method(events, "ffc") in WEAK_PHASE_METHODS
    )
    if not weak_ffc and raw_release is not None and raw_ffc >= raw_release - 1:
        weak_ffc = True

    candidate_ffc = fallback_ffc if weak_ffc else raw_ffc
    weak_bfc = (
        raw_bfc is None
        or raw_bfc < start + 1
        or raw_bfc >= stop - 2
        or (candidate_ffc is not None and raw_bfc >= candidate_ffc)
    )

    bfc_frame = raw_bfc if not weak_bfc else fallback_bfc
    ffc_frame = raw_ffc if not weak_ffc else fallback_ffc
    release_frame = raw_release if raw_release is not None else fallback_release

    bfc_frame = max(start + 1, min(last - 2, int(bfc_frame)))
    ffc_frame = max(bfc_frame + 1, min(last - 1, int(ffc_frame)))
    release_frame = max(ffc_frame + 1, min(last, int(release_frame)))

    if weak_bfc:
        event = dict((timeline.get("bfc") or {}))
        event.update(
            {
                "frame": int(bfc_frame),
                "confidence": max(_event_confidence(events, "bfc"), 0.30),
                "method": "render_phase_fallback",
            }
        )
        timeline["bfc"] = event

    if weak_ffc:
        event = dict((timeline.get("ffc") or {}))
        event.update(
            {
                "frame": int(ffc_frame),
                "confidence": max(_event_confidence(events, "ffc"), 0.40),
                "method": "render_phase_fallback",
            }
        )
        timeline["ffc"] = event

    if raw_release is None:
        event = dict((timeline.get("release") or {}))
        event.update(
            {
                "frame": int(release_frame),
                "confidence": max(_event_confidence(events, "release"), 0.50),
                "method": "render_phase_fallback",
            }
        )
        timeline["release"] = event

    return timeline
