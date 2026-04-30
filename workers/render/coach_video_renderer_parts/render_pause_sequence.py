from __future__ import annotations
from .shared import *
from .bubble_base import _reading_hold_frames
from .pause_logic import _pause_sequence_plan, _proof_bubble_text_for_phase, _select_hotspot_frame_idx, _hotspot_stage_plan
from .transfer_phase import _draw_transfer_leak_phase
from .body_pay import _draw_body_pay_phase
from .hotspot_phase import _draw_load_watch_phase


def _fit_stage_holds_to_budget(
    *,
    total_budget: int,
    desired_holds: Dict[str, int],
) -> Dict[str, int]:
    budget = max(0, int(total_budget))
    stages = [stage for stage in ("proof", "leak", "pay", "hotspot") if int(desired_holds.get(stage) or 0) > 0]
    if budget <= 0 or not stages:
        return {stage: 0 for stage in ("proof", "leak", "pay", "hotspot")}

    active_count = min(len(stages), budget)
    fitted = {stage: 0 for stage in ("proof", "leak", "pay", "hotspot")}
    for stage in stages[:active_count]:
        fitted[stage] = 1

    remaining = budget - active_count
    extras = {
        stage: max(0, int(desired_holds.get(stage) or 0) - fitted[stage])
        for stage in stages[:active_count]
    }
    total_extras = sum(extras.values())
    if remaining <= 0 or total_extras <= 0:
        return fitted

    remainders: List[Tuple[float, str]] = []
    used = 0
    for stage in stages[:active_count]:
        target = (remaining * extras[stage]) / float(total_extras)
        alloc = min(extras[stage], int(target))
        fitted[stage] += alloc
        used += alloc
        remainders.append((target - alloc, stage))

    leftover = remaining - used
    for _, stage in sorted(remainders, reverse=True):
        if leftover <= 0:
            break
        if fitted[stage] >= int(desired_holds.get(stage) or 0):
            continue
        fitted[stage] += 1
        leftover -= 1

    return fitted


def _transition_frame_count(*, fps: float, stage_frames: int) -> int:
    if stage_frames <= 1:
        return 0
    return max(0, min(stage_frames - 1, max(2, int(round(max(1.0, fps) * 0.12)))))

def _write_stage_frames(
    *,
    writer: Any,
    stage_frames: List[np.ndarray],
    previous_frame: Optional[np.ndarray],
    fps: float,
) -> Tuple[int, Optional[np.ndarray]]:
    if not stage_frames:
        return 0, previous_frame
    transition_frames = _transition_frame_count(fps=fps, stage_frames=len(stage_frames))
    frames_rendered = 0
    for idx, stage_frame in enumerate(stage_frames):
        output_frame = stage_frame
        if previous_frame is not None and idx < transition_frames:
            alpha = float(idx + 1) / float(transition_frames + 1)
            output_frame = cv2.addWeighted(previous_frame, 1.0 - alpha, stage_frame, alpha, 0.0)
        writer.write(output_frame)
        frames_rendered += 1
    return frames_rendered, stage_frames[-1]

def _render_pause_sequence(*, writer: Any, frame: np.ndarray, tracks: Dict[int, Dict[str, Any]], frame_idx: int, hand: Optional[str], pause_key: str, pause_frames: int, fps: float, risk_by_id: Dict[str, Dict[str, Any]], paused_frame: np.ndarray, hotspot_payload: Optional[Dict[str, Any]], leakage_payload: Optional[Dict[str, Any]], proof_step: Optional[Dict[str, Any]], start: int, stop: int) -> int:
    frames_rendered = 0
    sequence_plan = _pause_sequence_plan(pause_frames=pause_frames, has_hotspot=hotspot_payload is not None, has_leakage=leakage_payload is not None)
    desired_holds = {
        "proof": int(sequence_plan.get("proof") or 0),
        "leak": int(sequence_plan.get("leak") or 0),
        "pay": int(sequence_plan.get("pay") or 0),
        "hotspot": int(sequence_plan.get("hotspot") or 0),
    }
    proof_bubble_text = _proof_bubble_text_for_phase(phase_key=pause_key, risk_id=str((hotspot_payload or {}).get("risk_id") or ""), proof_step=proof_step, risk_by_id=risk_by_id)
    if proof_bubble_text:
        desired_holds["proof"] = max(
            desired_holds["proof"],
            _reading_hold_frames(text=proof_bubble_text, fps=fps, minimum_seconds=3.00, max_seconds=4.80),
        )
    if leakage_payload:
        desired_holds["leak"] = max(
            desired_holds["leak"],
            _reading_hold_frames(text=str((leakage_payload or {}).get("bubble") or ""), fps=fps, minimum_seconds=2.50, max_seconds=3.90),
        )
    if leakage_payload and hotspot_payload:
        desired_holds["pay"] = max(
            desired_holds["pay"],
            _reading_hold_frames(text="Body pays here.", fps=fps, minimum_seconds=2.30, max_seconds=3.40),
        )
    if hotspot_payload:
        desired_holds["hotspot"] = max(
            desired_holds["hotspot"],
            _reading_hold_frames(
                text="Load / fault point",
                fps=fps,
                minimum_seconds=2.00,
                max_seconds=2.80,
            ),
        )
    fitted_holds = _fit_stage_holds_to_budget(
        total_budget=int(pause_frames),
        desired_holds=desired_holds,
    )
    proof_hold = int(fitted_holds.get("proof") or 0)
    leakage_hold = int(fitted_holds.get("leak") or 0)
    body_pay_hold = int(fitted_holds.get("pay") or 0)
    hotspot_hold = int(fitted_holds.get("hotspot") or 0)
    previous_stage_frame: Optional[np.ndarray] = None
    proof_frames = [paused_frame.copy() for _ in range(proof_hold)]
    added_frames, previous_stage_frame = _write_stage_frames(
        writer=writer,
        stage_frames=proof_frames,
        previous_frame=previous_stage_frame,
        fps=fps,
    )
    frames_rendered += added_frames
    if leakage_payload and leakage_hold > 0:
        leakage_frames: List[np.ndarray] = []
        for leak_idx in range(leakage_hold):
            leakage_frame = frame.copy()
            _draw_transfer_leak_phase(leakage_frame, tracks=tracks, frame_idx=frame_idx, hand=hand, payload=leakage_payload, progress=float(leak_idx + 1) / float(max(1, leakage_hold)))
            leakage_frames.append(leakage_frame)
        added_frames, previous_stage_frame = _write_stage_frames(
            writer=writer,
            stage_frames=leakage_frames,
            previous_frame=previous_stage_frame,
            fps=fps,
        )
        frames_rendered += added_frames
    hotspot_frame_idx = frame_idx
    if hotspot_payload:
        hotspot_frame_idx = _select_hotspot_frame_idx(tracks=tracks, hand=hand, risk_id=str((hotspot_payload or {}).get("risk_id") or ""), risk_by_id=risk_by_id, phase_key=pause_key, anchor_frame=frame_idx, start=start, stop=stop)
    if leakage_payload and hotspot_payload and body_pay_hold > 0:
        pay_frames: List[np.ndarray] = []
        for pay_idx in range(body_pay_hold):
            pay_frame = frame.copy()
            _draw_body_pay_phase(pay_frame, tracks=tracks, frame_idx=hotspot_frame_idx, hand=hand, risk_id=str((hotspot_payload or {}).get("risk_id") or ""), risk_by_id=risk_by_id, region_priority=list((hotspot_payload or {}).get("region_priority") or []), progress=float(pay_idx + 1) / float(max(1, body_pay_hold)))
            pay_frames.append(pay_frame)
        added_frames, previous_stage_frame = _write_stage_frames(
            writer=writer,
            stage_frames=pay_frames,
            previous_frame=previous_stage_frame,
            fps=fps,
        )
        frames_rendered += added_frames
    if hotspot_payload and hotspot_hold > 0:
        hotspot_frames: List[np.ndarray] = []
        for stage, repeat_count in _hotspot_stage_plan(hotspot_hold):
            if repeat_count <= 0:
                continue
            hotspot_frame = frame.copy()
            pulse_phase = 0.50 if stage == "rings" else (0.85 if stage == "label" else 0.0)
            _draw_load_watch_phase(hotspot_frame, tracks=tracks, frame_idx=hotspot_frame_idx, hand=hand, risk_id=str((hotspot_payload or {}).get("risk_id") or ""), risk_by_id=risk_by_id, load_watch_text=str((hotspot_payload or {}).get("load_watch_text") or ""), region_priority=list((hotspot_payload or {}).get("region_priority") or []), pulse_phase=pulse_phase, stage=stage)
            hotspot_frames.extend([hotspot_frame.copy() for _ in range(repeat_count)])
        added_frames, previous_stage_frame = _write_stage_frames(
            writer=writer,
            stage_frames=hotspot_frames,
            previous_frame=previous_stage_frame,
            fps=fps,
        )
        frames_rendered += added_frames
    return frames_rendered
