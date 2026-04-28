from __future__ import annotations
from .shared import *
from .bubble_base import _reading_hold_frames
from .pause_logic import _pause_sequence_plan, _proof_bubble_text_for_phase, _select_hotspot_frame_idx, _hotspot_stage_plan
from .transfer_phase import _draw_transfer_leak_phase
from .body_pay import _draw_body_pay_phase
from .hotspot_phase import _draw_load_watch_phase


def _render_pause_sequence(*, writer: Any, frame: np.ndarray, tracks: Dict[int, Dict[str, Any]], frame_idx: int, hand: Optional[str], pause_key: str, pause_frames: int, fps: float, risk_by_id: Dict[str, Dict[str, Any]], paused_frame: np.ndarray, hotspot_payload: Optional[Dict[str, Any]], leakage_payload: Optional[Dict[str, Any]], proof_step: Optional[Dict[str, Any]], start: int, stop: int) -> int:
    frames_rendered = 0
    sequence_plan = _pause_sequence_plan(pause_frames=pause_frames, has_hotspot=hotspot_payload is not None, has_leakage=leakage_payload is not None)
    proof_hold = int(sequence_plan.get("proof") or 0)
    leakage_hold = int(sequence_plan.get("leak") or 0)
    body_pay_hold = int(sequence_plan.get("pay") or 0)
    hotspot_hold = int(sequence_plan.get("hotspot") or 0)
    proof_bubble_text = _proof_bubble_text_for_phase(phase_key=pause_key, risk_id=str((hotspot_payload or {}).get("risk_id") or ""), proof_step=proof_step, risk_by_id=risk_by_id)
    if proof_bubble_text:
        proof_hold = max(proof_hold, _reading_hold_frames(text=proof_bubble_text, fps=fps, minimum_seconds=1.85, max_seconds=3.35))
    if leakage_payload:
        leakage_hold = max(leakage_hold, _reading_hold_frames(text=str((leakage_payload or {}).get("bubble") or ""), fps=fps, minimum_seconds=1.55, max_seconds=2.6))
    if leakage_payload and hotspot_payload:
        body_pay_hold = max(body_pay_hold, _reading_hold_frames(text="Body pays here.", fps=fps, minimum_seconds=1.45, max_seconds=2.35))
    for _ in range(proof_hold):
        writer.write(paused_frame)
        frames_rendered += 1
    if leakage_payload and leakage_hold > 0:
        for leak_idx in range(leakage_hold):
            leakage_frame = frame.copy()
            _draw_transfer_leak_phase(leakage_frame, tracks=tracks, frame_idx=frame_idx, hand=hand, payload=leakage_payload, progress=float(leak_idx + 1) / float(max(1, leakage_hold)))
            writer.write(leakage_frame)
            frames_rendered += 1
    hotspot_frame_idx = frame_idx
    if hotspot_payload:
        hotspot_frame_idx = _select_hotspot_frame_idx(tracks=tracks, hand=hand, risk_id=str((hotspot_payload or {}).get("risk_id") or ""), risk_by_id=risk_by_id, phase_key=pause_key, anchor_frame=frame_idx, start=start, stop=stop)
    if leakage_payload and hotspot_payload and body_pay_hold > 0:
        for pay_idx in range(body_pay_hold):
            pay_frame = frame.copy()
            _draw_body_pay_phase(pay_frame, tracks=tracks, frame_idx=hotspot_frame_idx, hand=hand, risk_id=str((hotspot_payload or {}).get("risk_id") or ""), risk_by_id=risk_by_id, region_priority=list((hotspot_payload or {}).get("region_priority") or []), progress=float(pay_idx + 1) / float(max(1, body_pay_hold)))
            writer.write(pay_frame)
            frames_rendered += 1
    if hotspot_payload and hotspot_hold > 0:
        stage_frames: List[Tuple[np.ndarray, int]] = []
        for stage, repeat_count in _hotspot_stage_plan(hotspot_hold):
            if repeat_count <= 0:
                continue
            hotspot_frame = frame.copy()
            pulse_phase = 0.50 if stage == "rings" else (0.85 if stage == "label" else 0.0)
            _draw_load_watch_phase(hotspot_frame, tracks=tracks, frame_idx=hotspot_frame_idx, hand=hand, risk_id=str((hotspot_payload or {}).get("risk_id") or ""), risk_by_id=risk_by_id, load_watch_text=str((hotspot_payload or {}).get("load_watch_text") or ""), region_priority=list((hotspot_payload or {}).get("region_priority") or []), pulse_phase=pulse_phase, stage=stage)
            stage_frames.append((hotspot_frame, repeat_count))
        for hotspot_frame, repeat_count in stage_frames:
            for _ in range(repeat_count):
                writer.write(hotspot_frame)
                frames_rendered += 1
    return frames_rendered
