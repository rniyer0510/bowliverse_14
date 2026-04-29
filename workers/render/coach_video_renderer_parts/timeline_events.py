from __future__ import annotations
from .shared import *
from .analytics import _safe_int, _safe_float, _event_confidence
from .tracks import *
from .render_frame_resolver import resolve_render_timeline_events

def _phase_cut_points(
    *,
    start: int,
    stop: int,
    events: Optional[Dict[str, Any]],
    fps: float = 30.0,
) -> Tuple[int, int, int]:
    last = max(start + 3, stop - 1)
    span = max(4, stop - start)
    render_events = _render_timeline_events(start=start, stop=stop, events=events, fps=fps)
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
    quality_sum = 0.0
    for joint_idx in TRACKED_JOINTS:
        visibility = _safe_landmark_value(landmarks, joint_idx, "visibility") or 0.0
        x = _safe_landmark_value(landmarks, joint_idx, "x")
        y = _safe_landmark_value(landmarks, joint_idx, "y")
        if x is None or y is None:
            continue
        quality_sum += _visibility_weight(visibility)
    return float(quality_sum) / float(max(1, len(TRACKED_JOINTS)))
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
    fps: float = 30.0,
) -> Dict[str, Any]:
    return resolve_render_timeline_events(
        start=start,
        stop=stop,
        fps=fps,
        events=events,
    )
