from __future__ import annotations
from .shared import *
from .analytics import _risk_lookup, _safe_int
from .render_output import _make_output_path, _intermediate_render_path, _finalize_render_video
from .tracks import _build_smoothed_tracks, _track_point
from .drawing_base import _draw_skeleton, _draw_skeleton_legend
from .timeline_events import _should_draw_skeleton_frame, _render_timeline_events
from .render_frame_resolver import attach_render_quality_metadata
from .phase_rail import _draw_phase_overlay
from .pause_logic import _pause_anchor_frames
from .render_pause_payloads import _prepare_pause_context
from .render_pause_sequence import _render_pause_sequence
from .summary_legacy import _draw_end_summary


def _subject_height_ratio(
    *,
    tracks: Dict[int, Dict[str, Any]],
    start: int,
    stop: int,
    frame_height: int,
) -> float:
    if frame_height <= 0 or not tracks or stop <= start:
        return 0.0

    sampled_ratios: List[float] = []
    step = max(1, int(round((stop - start) / 18.0)))
    sample_joints = (0, 11, 12, 23, 24, 27, 28)

    for frame_idx in range(start, stop, step):
        ys: List[int] = []
        for joint_idx in sample_joints:
            point = _track_point(tracks, joint_idx, frame_idx)
            if point is None:
                continue
            ys.append(int(point[1]))
        if len(ys) < 4:
            continue
        sampled_ratios.append(max(0.0, min(1.0, (max(ys) - min(ys)) / float(frame_height))))

    if not sampled_ratios:
        return 0.0
    return float(np.median(sampled_ratios))


def _adaptive_hold_seconds(
    *,
    base_seconds: float,
    fps: float,
    subject_height_ratio: float,
    purpose: str,
) -> float:
    seconds = max(0.0, float(base_seconds or 0.0))
    if seconds <= 0.0:
        return 0.0

    scale = 1.0
    if fps >= 50.0:
        scale *= 0.84
    elif fps >= 35.0:
        scale *= 0.92

    if subject_height_ratio > 0.0 and subject_height_ratio < 0.42:
        scale *= 0.78
    elif subject_height_ratio < 0.56:
        scale *= 0.88

    if purpose == "pause":
        minimum = 2.6
        return round(min(seconds, max(minimum, seconds * scale)), 2)

    minimum = 1.2
    scale *= 0.82
    return round(min(seconds, max(minimum, seconds * scale)), 2)


def render_skeleton_video(*, video_path: str, pose_frames: List[Dict[str, Any]], events: Optional[Dict[str, Any]] = None, hand: Optional[str] = None, action: Optional[Dict[str, Any]] = None, elbow: Optional[Dict[str, Any]] = None, risks: Optional[List[Dict[str, Any]]] = None, estimated_release_speed: Optional[Dict[str, Any]] = None, kinetic_chain: Optional[Dict[str, Any]] = None, report_story: Optional[Dict[str, Any]] = None, root_cause: Optional[Dict[str, Any]] = None, output_path: Optional[str] = None, start_frame: int = 0, end_frame: Optional[int] = None, pause_seconds: float = 5.0, slow_motion_factor: float = 5.0, end_summary_seconds: float = 2.5) -> Dict[str, Any]:
    if not video_path or not os.path.exists(video_path):
        return {"available": False, "reason": "missing_video_path"}
    if not pose_frames:
        return {"available": False, "reason": "missing_pose_frames"}
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"available": False, "reason": "video_open_failed"}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or len(pose_frames))
    if width <= 0 or height <= 0:
        cap.release()
        return {"available": False, "reason": "missing_video_geometry"}
    start = max(0, int(start_frame))
    stop = min(total_frames, len(pose_frames), int(end_frame) if end_frame is not None else min(total_frames, len(pose_frames)))
    if stop <= start:
        cap.release()
        return {"available": False, "reason": "empty_render_window"}
    out_path = _make_output_path(output_path)
    intermediate_path = _intermediate_render_path(out_path)
    writer = cv2.VideoWriter(intermediate_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        return {"available": False, "reason": "writer_open_failed"}
    try:
        tracks = _build_smoothed_tracks(pose_frames, width=width, height=height, fps=fps)
        subject_height_ratio = _subject_height_ratio(
            tracks=tracks,
            start=start,
            stop=stop,
            frame_height=height,
        )
        effective_pause_seconds = _adaptive_hold_seconds(
            base_seconds=float(pause_seconds or 0.0),
            fps=fps,
            subject_height_ratio=subject_height_ratio,
            purpose="pause",
        )
        effective_summary_seconds = _adaptive_hold_seconds(
            base_seconds=float(end_summary_seconds or 0.0),
            fps=fps,
            subject_height_ratio=subject_height_ratio,
            purpose="summary",
        )
        pause_frames = max(0, int(round(effective_pause_seconds * fps)))
        render_events = _render_timeline_events(start=start, stop=stop, events=events, fps=fps)
        render_events = attach_render_quality_metadata(events=render_events, tracks=tracks)
        pause_anchors = _pause_anchor_frames(start=start, stop=stop, events=render_events, fps=fps)
        ffc_frame = _safe_int(((render_events or {}).get("ffc") or {}).get("frame"))
        release_frame = _safe_int(((render_events or {}).get("release") or {}).get("frame"))
        slow_motion_extra_frames = max(0, int(round(float(slow_motion_factor or 1.0))) - 1)
        slow_motion_start = ffc_frame if ffc_frame is not None and release_frame is not None and start <= ffc_frame <= release_frame < stop else None
        slow_motion_end = release_frame if slow_motion_start is not None else None
        legend_end_frame = min(stop, start + int(round(float(fps) * LEGEND_DURATION_SECONDS)))
        risk_by_id = _risk_lookup(risks)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frame_idx = start
        frames_rendered = 0
        final_summary_frame: Optional[np.ndarray] = None
        while frame_idx < stop:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            raw_frame = frame.copy()
            if _should_draw_skeleton_frame(pose_frames=pose_frames, tracks=tracks, frame_idx=frame_idx, events=render_events, fps=fps):
                _draw_skeleton(frame, tracks, frame_idx)
            _draw_phase_overlay(frame, frame_idx=frame_idx, start=start, stop=stop, events=render_events, fps=fps)
            if frame_idx < legend_end_frame:
                _draw_skeleton_legend(frame, fps=fps, frame_idx=frame_idx, legend_end_frame=legend_end_frame)
            final_summary_frame = raw_frame
            writer.write(frame)
            frames_rendered += 1
            if slow_motion_start is not None and slow_motion_end is not None and slow_motion_start <= frame_idx <= slow_motion_end:
                for _ in range(slow_motion_extra_frames):
                    writer.write(frame)
                    frames_rendered += 1
            pause_key = pause_anchors.get(frame_idx)
            if pause_frames > 0 and pause_key:
                pause_context = _prepare_pause_context(frame=frame, pose_frames=pose_frames, tracks=tracks, frame_idx=frame_idx, pause_key=pause_key, hand=hand, risk_by_id=risk_by_id, render_events=render_events, report_story=report_story, root_cause=root_cause, kinetic_chain=kinetic_chain)
                frames_rendered += _render_pause_sequence(writer=writer, frame=frame, tracks=tracks, frame_idx=frame_idx, hand=hand, pause_key=pause_key, pause_frames=pause_frames, fps=fps, risk_by_id=risk_by_id, paused_frame=pause_context["paused_frame"], hotspot_payload=pause_context["hotspot_payload"], leakage_payload=pause_context["leakage_payload"], proof_step=pause_context["proof_step"], start=start, stop=stop)
            frame_idx += 1
        summary_hold_frames = max(0, int(round(effective_summary_seconds * fps)))
        if final_summary_frame is not None and summary_hold_frames > 0:
            summary_frame = final_summary_frame.copy()
            _draw_end_summary(summary_frame, risk_by_id=risk_by_id, events=render_events, action=action, speed=estimated_release_speed, elbow=elbow, report_story=report_story, root_cause=root_cause)
            for _ in range(summary_hold_frames):
                writer.write(summary_frame)
                frames_rendered += 1
        writer.release()
        cap.release()
    except Exception as exc:
        writer.release()
        cap.release()
        try:
            if os.path.exists(intermediate_path):
                os.remove(intermediate_path)
        except OSError:
            pass
        logger.exception("[coach_video_renderer] Skeleton render failed: %s", exc)
        return {"available": False, "reason": "render_failed", "detail": str(exc)}
    final_path, encoding = _finalize_render_video(intermediate_path, out_path)
    logger.info("[coach_video_renderer] Rendered skeleton video path=%s frames=%s fps=%.2f", final_path, frames_rendered, fps)
    return {"available": True, "path": final_path, "fps": round(fps, 3), "frames_rendered": frames_rendered, "width": width, "height": height, "start_frame": start, "end_frame": max(start, stop - 1), "style": "skeleton_phase_v1", "pause_seconds": round(float(effective_pause_seconds or 0.0), 2), "slow_motion_factor": round(float(slow_motion_factor or 1.0), 2), "end_summary_seconds": round(float(effective_summary_seconds or 0.0), 2), "encoding": encoding, "render_events": render_events, "render_quality": (render_events or {}).get("render_quality") or {"overall": 0.0, "event_count": 0}, "subject_height_ratio": round(subject_height_ratio, 3)}
