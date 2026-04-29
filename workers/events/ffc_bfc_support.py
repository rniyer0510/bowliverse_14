from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import math

import numpy as np

from app.workers.events.event_confidence import annotate_detection_contract, build_candidate
from app.workers.events.timing_constants import foot_contact_timing

MIN_VIS = 0.25


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def as_int(v) -> Optional[int]:
    try:
        return int(v)
    except Exception:
        return None


def _interp_nans(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    idx = np.arange(len(x))
    good = np.isfinite(x)
    if good.sum() < 2:
        return np.full_like(x, np.nan)
    x[~good] = np.interp(idx[~good], idx[good], x[good])
    return x


def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    k = max(1, int(k))
    if k <= 1:
        return x
    if k % 2 == 0:
        k += 1
    w = np.ones(k, dtype=float) / float(k)
    return np.convolve(x, w, mode="same")


def robust_percentile(x: np.ndarray, p: float) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    return float(np.percentile(x, p))


def _robust_mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 1e9
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)) + 1e-9)


def _finite_percentile(x: np.ndarray, p: float, default: float = 0.0) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float(default)
    return float(np.percentile(x, p))


def _vis(lm: list, idx: int) -> float:
    try:
        if idx >= len(lm) or lm[idx] is None:
            return 0.0
        return float(lm[idx].get("visibility", 0.0))
    except Exception:
        return 0.0


def _xy(lm: list, idx: int):
    try:
        if idx >= len(lm) or lm[idx] is None:
            return None
        x = lm[idx].get("x", None)
        y = lm[idx].get("y", None)
        if x is None or y is None:
            return None
        return float(x), float(y)
    except Exception:
        return None


def series_y(pose_frames: List[Dict], idx: int) -> np.ndarray:
    y = np.full(len(pose_frames), np.nan, dtype=float)
    for i, fr in enumerate(pose_frames):
        lm = fr.get("landmarks") or []
        if not isinstance(lm, list):
            continue
        p = _xy(lm, idx)
        if p is not None and np.isfinite(p[1]):
            y[i] = p[1]
    return _interp_nans(y)


def series_vis(pose_frames: List[Dict], idx: int) -> np.ndarray:
    vis = np.zeros(len(pose_frames), dtype=float)
    for i, fr in enumerate(pose_frames):
        lm = fr.get("landmarks") or []
        if not isinstance(lm, list):
            continue
        vis[i] = _vis(lm, idx)
    return vis


def _foot_ground_score(
    y_ank: np.ndarray,
    y_toe: np.ndarray,
    t: int,
    hold: int,
    win0: int,
    win1: int,
    dt: float,
) -> int:
    if t < 0 or t + hold >= len(y_ank):
        return 0

    a_seg = y_ank[t : t + hold]
    t_seg = y_toe[t : t + hold]
    if not np.all(np.isfinite(a_seg)) or not np.all(np.isfinite(t_seg)):
        return 0

    w0 = max(win0, t - max(hold * 3, 6))
    w1 = min(win1, t + max(hold, 3))
    hist_a = y_ank[w0 : w1 + 1]
    hist_t = y_toe[w0 : w1 + 1]
    if not np.any(np.isfinite(hist_a)) or not np.any(np.isfinite(hist_t)):
        return 0

    low_a = robust_percentile(hist_a, 85)
    low_t = robust_percentile(hist_t, 85)
    near_low = (np.median(a_seg) >= low_a - 0.01) and (np.median(t_seg) >= low_t - 0.01)

    rng_a = max(robust_percentile(hist_a, 90) - robust_percentile(hist_a, 10), 1e-6)
    rng_t = max(robust_percentile(hist_t, 90) - robust_percentile(hist_t, 10), 1e-6)

    dy_a = np.diff(a_seg) / max(dt, 1e-6)
    dy_t = np.diff(t_seg) / max(dt, 1e-6)

    v_ok = (np.median(np.abs(dy_a)) <= 0.18 * (rng_a / max(dt, 1e-6))) and \
           (np.median(np.abs(dy_t)) <= 0.18 * (rng_t / max(dt, 1e-6)))

    jit_ok = (_robust_mad(a_seg) <= 0.15 * rng_a) and (_robust_mad(t_seg) <= 0.15 * rng_t)

    score = 0
    score += 1 if near_low else 0
    score += 1 if v_ok else 0
    score += 1 if jit_ok else 0
    return score


def is_grounded(
    y_ank: np.ndarray,
    y_toe: np.ndarray,
    t: int,
    hold: int,
    win0: int,
    win1: int,
    dt: float,
) -> bool:
    return _foot_ground_score(y_ank, y_toe, t, hold, win0, win1, dt) >= 2


def _recently_grounded(
    y_ank: np.ndarray,
    y_toe: np.ndarray,
    t: int,
    hold: int,
    win0: int,
    win1: int,
    dt: float,
    lookback: int,
) -> bool:
    j0 = max(win0, t - lookback)
    for j in range(t - 1, j0 - 1, -1):
        if is_grounded(y_ank, y_toe, j, hold, win0, win1, dt):
            return True
    return False


def _candidate_case(
    *,
    frame: int,
    front_label: str,
    front_score: int,
    back_score: int,
    method: str,
    back_recent: bool = False,
    score_bonus: float = 0.0,
):
    back_support = max(back_score, 1 if back_recent else 0)
    confidence = min(0.9, 0.42 + 0.08 * front_score + 0.05 * back_support + score_bonus)
    return build_candidate(
        frame=frame,
        method=method,
        confidence=confidence,
        score=float(front_score + 0.5 * back_support + score_bonus),
        reason=f"{front_label}_front",
    )


def _ffc_rank_score(
    *,
    frame: int,
    search_start: int,
    pelvis_on: int,
    win_end: int,
    front_score: int,
    back_support: int,
    front_edge: float,
    stability: float,
) -> float:
    chain_anchor = min(max(pelvis_on, search_start), win_end)
    pre_span = max(1, chain_anchor - search_start)
    post_span = max(1, win_end - chain_anchor)
    if frame < chain_anchor:
        anchor_bias = -0.45 * ((chain_anchor - frame) / float(pre_span))
    else:
        anchor_bias = 0.10 * ((frame - chain_anchor) / float(post_span))
    return (
        (0.40 * float(front_score))
        + (0.10 * float(back_support))
        + (1.60 * float(front_edge))
        + (0.06 * float(stability))
        + anchor_bias
    )


def _ground_window_strength(
    y_ank: np.ndarray,
    y_toe: np.ndarray,
    *,
    frame: int,
    hold: int,
    win_start: int,
    win_end: int,
    dt: float,
    radius: int,
) -> float:
    strength = 0.0
    for offset in range(-radius, radius + 1):
        idx = frame + offset
        if idx < win_start or idx + hold >= len(y_ank):
            continue
        weight = 1.0 / (1.0 + abs(offset))
        strength += weight * _foot_ground_score(
            y_ank,
            y_toe,
            idx,
            hold,
            win_start,
            win_end,
            dt,
        )
    return strength


def _speed_scale(
    approach_speed: Optional[np.ndarray],
    *,
    start: int,
    end: int,
) -> float:
    if approach_speed is None:
        return 1.0
    segment = np.asarray(approach_speed[start : end + 1], dtype=float)
    segment = segment[np.isfinite(segment) & (segment > 0.0)]
    if segment.size < 3:
        return 1.0
    p50 = float(np.percentile(segment, 50))
    p85 = float(np.percentile(segment, 85))
    if p50 <= 1e-6:
        return 1.0
    ratio = p85 / p50
    return max(0.85, min(1.20, ratio))


def _contact_edge_strength(
    y_ank: np.ndarray,
    y_toe: np.ndarray,
    *,
    frame: int,
    hold: int,
    win_start: int,
    win_end: int,
    dt: float,
    pelvis_jerk: Optional[np.ndarray] = None,
) -> float:
    if frame < win_start or frame + hold >= len(y_ank):
        return 0.0

    current_score = _foot_ground_score(y_ank, y_toe, frame, hold, win_start, win_end, dt)
    if current_score < 2:
        return 0.0

    prev_score = 0
    if frame - 1 >= win_start:
        prev_score = _foot_ground_score(y_ank, y_toe, frame - 1, hold, win_start, win_end, dt)

    next_score = current_score
    if frame + 1 + hold < len(y_ank):
        next_score = _foot_ground_score(y_ank, y_toe, frame + 1, hold, win_start, win_end, dt)

    edge_ground = 1.0 if prev_score < 2 else max(0.0, min(1.0, (current_score - prev_score) / 3.0))

    pre_start = max(win_start, frame - hold)
    pre_ank = y_ank[pre_start : frame + 1]
    pre_toe = y_toe[pre_start : frame + 1]
    post_end = min(len(y_ank), frame + hold + 1)
    post_ank = y_ank[frame:post_end]
    post_toe = y_toe[frame:post_end]

    hist_a = y_ank[max(win_start, frame - hold * 3) : min(win_end + 1, frame + hold * 2)]
    hist_t = y_toe[max(win_start, frame - hold * 3) : min(win_end + 1, frame + hold * 2)]
    rng_a = max(_finite_percentile(hist_a, 90) - _finite_percentile(hist_a, 10), 1e-6)
    rng_t = max(_finite_percentile(hist_t, 90) - _finite_percentile(hist_t, 10), 1e-6)
    velocity_scale = max((rng_a + rng_t) / max(dt, 1e-6), 1e-6)

    pre_speed = 0.0
    post_speed = 0.0
    if len(pre_ank) >= 2 and len(pre_toe) >= 2:
        pre_speed = float(np.median(np.maximum(0.0, np.diff(pre_ank) / max(dt, 1e-6))))
        pre_speed += float(np.median(np.maximum(0.0, np.diff(pre_toe) / max(dt, 1e-6))))
    if len(post_ank) >= 2 and len(post_toe) >= 2:
        post_speed = float(np.median(np.abs(np.diff(post_ank) / max(dt, 1e-6))))
        post_speed += float(np.median(np.abs(np.diff(post_toe) / max(dt, 1e-6))))
    decel = max(0.0, min(1.0, (pre_speed - post_speed) / velocity_scale))

    stability = max(0.0, min(1.0, next_score / 3.0))

    jerk_cue = 0.0
    if pelvis_jerk is not None and 0 <= frame < len(pelvis_jerk):
        jerk_window = pelvis_jerk[max(win_start, frame - 2) : min(win_end + 1, frame + 3)]
        jerk_ref = max(_finite_percentile(jerk_window, 90, default=0.0), 1e-6)
        jerk_value = float(np.nan_to_num(pelvis_jerk[frame], nan=0.0))
        jerk_cue = max(0.0, min(1.0, jerk_value / jerk_ref))

    return float(max(0.0, min(1.0, (0.45 * edge_ground) + (0.25 * decel) + (0.15 * stability) + (0.15 * jerk_cue))))


def _refine_to_established_support(
    y_ank: np.ndarray,
    y_toe: np.ndarray,
    *,
    frame: int,
    hold: int,
    win_start: int,
    win_end: int,
    dt: float,
    pelvis_jerk: Optional[np.ndarray] = None,
    max_forward: Optional[int] = None,
) -> int:
    """
    Convert an initial contact-edge frame into the first clearly established
    support frame in the same grounded block.

    This keeps the detector aligned with a coach-visible landing frame instead
    of a just-barely-touching edge when the foot is still settling down.
    """
    grounded = _foot_ground_score(y_ank, y_toe, frame, hold, win_start, win_end, dt)
    if grounded < 2:
        return int(frame)

    start_edge = _contact_edge_strength(
        y_ank,
        y_toe,
        frame=frame,
        hold=hold,
        win_start=win_start,
        win_end=win_end,
        dt=dt,
        pelvis_jerk=pelvis_jerk,
    )
    if start_edge < 0.45:
        return int(frame)

    radius = max(1, hold // 2)
    max_step = max(1, int(max_forward or hold))
    limit = min(int(win_end), int(frame) + max_step)
    for idx in range(int(frame) + 1, limit + 1):
        score = _foot_ground_score(y_ank, y_toe, idx, hold, win_start, win_end, dt)
        if score < 2:
            break
        edge = _contact_edge_strength(
            y_ank,
            y_toe,
            frame=idx,
            hold=hold,
            win_start=win_start,
            win_end=win_end,
            dt=dt,
            pelvis_jerk=pelvis_jerk,
        )
        stability = _ground_window_strength(
            y_ank,
            y_toe,
            frame=idx,
            hold=hold,
            win_start=win_start,
            win_end=win_end,
            dt=dt,
            radius=radius,
        )
        if edge <= max(0.22, start_edge * 0.55) and stability >= 3.0:
            return int(idx)

    return int(frame)


def _pick_local_contact_frame(
    ranked: List[Tuple[float, int, str, Dict]],
    *,
    fps: Optional[float],
    hold: int,
) -> Tuple[float, int, str, Dict]:
    """
    Pick the highest-scoring contact first, then allow a slightly later frame
    only when it stays within the same local landing band.

    This prevents broad score-band tie-breaking from drifting into a much later
    grounded block near release while still allowing a 1-2 frame coach-visible
    establishment shift inside the same contact neighborhood.
    """
    ranked = list(ranked or [])
    if not ranked:
        raise ValueError("ranked candidates required")

    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    base = ranked[0]
    base_score, base_frame, _, _ = base
    timing = foot_contact_timing(float(fps or 0.0))
    local_post_band = max(int(hold), int(timing["ffc_local_post_band"]), 1)
    local_score_band = 0.18

    local = [
        item
        for item in ranked
        if item[1] >= base_frame
        and (item[1] - base_frame) <= local_post_band
        and (base_score - item[0]) <= local_score_band
    ]
    if not local:
        return base

    local.sort(key=lambda item: item[1], reverse=True)
    return local[0]


def pick_ffc_backward_from_release(
    *,
    search_start: int,
    pelvis_on: int,
    preferred_front_side: Optional[str],
    win_end: int,
    hold: int,
    win_start: int,
    dt: float,
    back_recent: int,
    y_LA: np.ndarray,
    y_RA: np.ndarray,
    y_LFI: np.ndarray,
    y_RFI: np.ndarray,
    fps: Optional[float] = None,
    approach_speed: Optional[np.ndarray] = None,
    pelvis_jerk: Optional[np.ndarray] = None,
) -> Tuple[Optional[int], Optional[str], List[Dict], float]:
    candidates: List[Dict] = []
    ranked: List[Tuple[float, int, str, Dict]] = []
    radius = max(1, hold // 2)
    edge_only_threshold = 0.55

    for i in range(win_end, search_start - 1, -1):
        left_score = _foot_ground_score(y_LA, y_LFI, i, hold, win_start, win_end, dt)
        right_score = _foot_ground_score(y_RA, y_RFI, i, hold, win_start, win_end, dt)
        left_grounded = left_score >= 2
        right_grounded = right_score >= 2
        right_recent = _recently_grounded(
            y_RA, y_RFI, i, hold, win_start, win_end, dt, back_recent
        )
        left_recent = _recently_grounded(
            y_LA, y_LFI, i, hold, win_start, win_end, dt, back_recent
        )

        left_edge = _contact_edge_strength(
            y_LA,
            y_LFI,
            frame=i,
            hold=hold,
            win_start=win_start,
            win_end=win_end,
            dt=dt,
            pelvis_jerk=pelvis_jerk,
        )
        right_edge = _contact_edge_strength(
            y_RA,
            y_RFI,
            frame=i,
            hold=hold,
            win_start=win_start,
            win_end=win_end,
            dt=dt,
            pelvis_jerk=pelvis_jerk,
        )

        left_front_ok = left_grounded and (right_grounded or right_recent or left_edge >= edge_only_threshold)
        right_front_ok = right_grounded and (left_grounded or left_recent or right_edge >= edge_only_threshold)

        if left_front_ok:
            left_stability = _ground_window_strength(
                y_LA,
                y_LFI,
                frame=i,
                hold=hold,
                win_start=win_start,
                win_end=win_end,
                dt=dt,
                radius=radius,
            )
            back_support = max(right_score, 1 if right_recent else 0)
            rank_score = _ffc_rank_score(
                frame=i,
                search_start=search_start,
                pelvis_on=pelvis_on,
                win_end=win_end,
                front_score=left_score,
                back_support=back_support,
                front_edge=left_edge,
                stability=left_stability,
            )
            candidate = _candidate_case(
                frame=i,
                front_label="left",
                front_score=left_score,
                back_score=right_score,
                back_recent=right_recent,
                method="release_backward_chain_grounding",
                score_bonus=rank_score,
            )
            if candidate is not None:
                candidate["score"] = round(rank_score, 4)
                candidate["contact_edge_strength"] = round(left_edge, 3)
                candidates.append(candidate)
                ranked.append(
                    (
                        rank_score,
                        i,
                        "left",
                        candidate,
                    )
                )

        if right_front_ok:
            right_stability = _ground_window_strength(
                y_RA,
                y_RFI,
                frame=i,
                hold=hold,
                win_start=win_start,
                win_end=win_end,
                dt=dt,
                radius=radius,
            )
            back_support = max(left_score, 1 if left_recent else 0)
            rank_score = _ffc_rank_score(
                frame=i,
                search_start=search_start,
                pelvis_on=pelvis_on,
                win_end=win_end,
                front_score=right_score,
                back_support=back_support,
                front_edge=right_edge,
                stability=right_stability,
            )
            candidate = _candidate_case(
                frame=i,
                front_label="right",
                front_score=right_score,
                back_score=left_score,
                back_recent=left_recent,
                method="release_backward_chain_grounding",
                score_bonus=rank_score,
            )
            if candidate is not None:
                candidate["score"] = round(rank_score, 4)
                candidate["contact_edge_strength"] = round(right_edge, 3)
                candidates.append(candidate)
                ranked.append(
                    (
                        rank_score,
                        i,
                        "right",
                        candidate,
                    )
                )

    if ranked:
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        best_score = ranked[0][0]
        close_ranked = [item for item in ranked if (best_score - item[0]) <= 0.18]
        if preferred_front_side in {"left", "right"}:
            preferred = [
                item for item in ranked
                if item[2] == preferred_front_side and (best_score - item[0]) <= 0.65
            ]
            if preferred:
                max_edge = max(
                    float((item[3] or {}).get("contact_edge_strength") or 0.0)
                    for item in preferred
                )
                strong_preferred = [
                    item for item in preferred
                    if float((item[3] or {}).get("contact_edge_strength") or 0.0)
                    >= max(0.25, max_edge * 0.60)
                ]
                if strong_preferred:
                    preferred = strong_preferred
                picked = _pick_local_contact_frame(
                    preferred,
                    fps=fps,
                    hold=hold,
                )
                if picked[1] == pelvis_on:
                    preferred_pre_chain = [item for item in preferred if item[1] < pelvis_on]
                    if preferred_pre_chain:
                        picked = _pick_local_contact_frame(
                            preferred_pre_chain,
                            fps=fps,
                            hold=hold,
                        )
                _, frame, side, candidate = picked
                frame = _refine_to_established_support(
                    y_LA if side == "left" else y_RA,
                    y_LFI if side == "left" else y_RFI,
                    frame=frame,
                    hold=hold,
                    win_start=win_start,
                    win_end=win_end,
                    dt=dt,
                    pelvis_jerk=pelvis_jerk,
                )
                return (
                    frame,
                    side,
                    candidates,
                    float(candidate.get("confidence") or 0.62),
                )
        _, frame, front_side, candidate = _pick_local_contact_frame(
            close_ranked,
            fps=fps,
            hold=hold,
        )
        frame = _refine_to_established_support(
            y_LA if front_side == "left" else y_RA,
            y_LFI if front_side == "left" else y_RFI,
            frame=frame,
            hold=hold,
            win_start=win_start,
            win_end=win_end,
            dt=dt,
            pelvis_jerk=pelvis_jerk,
        )
        return (
            frame,
            front_side,
            candidates,
            float(candidate.get("confidence") or 0.62),
        )

    return None, None, candidates, 0.0


def detection_context(
    *,
    fps: float,
    fps_was_defaulted: bool,
    window_was_clipped: bool,
    win_start: int,
    win_end: int,
) -> Dict[str, object]:
    return {
        "fps": float(fps),
        "fps_was_defaulted": bool(fps_was_defaulted),
        "window_was_clipped": bool(window_was_clipped),
        "window": [int(win_start), int(win_end)],
    }


def annotated_result(payload: Dict[str, Dict]) -> Dict[str, Dict]:
    return annotate_detection_contract(payload)


def ffc_search_start(
    *,
    win_start: int,
    win_end: int,
    pelvis_on: int,
    fps: float,
    hold: int,
) -> int:
    timing = foot_contact_timing(fps)
    min_ffc_release_band = max(hold + 1, int(timing["min_ffc_release_band"]))
    pre_pelvis_lead = max(hold + 2, int(timing["pre_pelvis_lead"]))
    lower_bound = pelvis_on - pre_pelvis_lead
    return max(win_start, min(lower_bound, win_end - min_ffc_release_band))


def pick_bfc_backward_from_ffc(
    *,
    ffc: int,
    front_side: Optional[str],
    hold: int,
    win_start: int,
    win_end: int,
    dt: float,
    fps: float,
    y_LA: np.ndarray,
    y_RA: np.ndarray,
    y_LFI: np.ndarray,
    y_RFI: np.ndarray,
    vis_LA: np.ndarray,
    vis_RA: np.ndarray,
    vis_LFI: np.ndarray,
    vis_RFI: np.ndarray,
    approach_speed: Optional[np.ndarray] = None,
    pelvis_jerk: Optional[np.ndarray] = None,
) -> Tuple[int, float, str]:
    if front_side == "left":
        back_ank, back_toe = y_RA, y_RFI
        back_ank_vis, back_toe_vis = vis_RA, vis_RFI
        front_ank, front_toe = y_LA, y_LFI
    elif front_side == "right":
        back_ank, back_toe = y_LA, y_LFI
        back_ank_vis, back_toe_vis = vis_LA, vis_LFI
        front_ank, front_toe = y_RA, y_RFI
    else:
        return clamp(ffc - 1, win_start, ffc), 0.0, "context_pre_ffc"

    timing = foot_contact_timing(fps)
    speed_scale = _speed_scale(
        approach_speed,
        start=max(win_start, max(0, ffc - int(timing["recent_support_band"]))),
        end=max(win_start, min(win_end, ffc - 1)),
    )
    min_gap = max(1, int(round(float(timing["bfc_min_ffc_gap"]) * speed_scale)))
    max_gap = max(hold + 1, int(round(float(timing["bfc_max_ffc_gap"]) * speed_scale)))
    band_start = max(win_start, ffc - max_gap)
    band_end = max(band_start, min(ffc - min_gap, ffc - 1))
    recent_band = max(hold + 2, int(timing["recent_support_band"]))
    edge_search_start = max(band_start, ffc - recent_band)
    best_edge: Optional[Tuple[float, int, float]] = None
    recent_vis = []
    for idx in range(edge_search_start, band_end + 1):
        if idx < len(back_ank_vis):
            recent_vis.append(float(min(back_ank_vis[idx], back_toe_vis[idx])))
    median_recent_vis = float(np.median(recent_vis)) if recent_vis else 0.0

    if median_recent_vis >= MIN_VIS:
        denom = max(1, ffc - edge_search_start)
        for idx in range(edge_search_start, band_end + 1):
            back_score = _foot_ground_score(back_ank, back_toe, idx, hold, win_start, win_end, dt)
            if back_score < 2:
                continue
            edge_strength = _contact_edge_strength(
                back_ank,
                back_toe,
                frame=idx,
                hold=hold,
                win_start=win_start,
                win_end=win_end,
                dt=dt,
                pelvis_jerk=pelvis_jerk,
            )
            if edge_strength < 0.30:
                continue
            support_after = 0.0
            valid_after = 0
            for j in range(idx, min(ffc, idx + hold + 1)):
                valid_after += 1
                if _foot_ground_score(back_ank, back_toe, j, hold, win_start, win_end, dt) >= 2:
                    support_after += 1.0
            support_ratio = support_after / max(1, valid_after)
            front_penalty = _foot_ground_score(front_ank, front_toe, idx, hold, win_start, win_end, dt) / 3.0
            late_bias = float(idx - edge_search_start) / float(denom)
            score = edge_strength + (0.18 * support_ratio) + (0.10 * late_bias) - (0.10 * front_penalty)
            if best_edge is None or score > best_edge[0]:
                best_edge = (score, idx, edge_strength)

    earliest = max(band_start, ffc - max(3, hold + 1))
    seed_frame: Optional[int] = None

    for idx in range(band_end, earliest - 1, -1):
        back_score = _foot_ground_score(back_ank, back_toe, idx, hold, win_start, win_end, dt)
        if back_score >= 2:
            seed_frame = idx
            continue
        if seed_frame is not None:
            break

    if seed_frame is None:
        return clamp(ffc - 1, win_start, ffc), 0.0, "no_ground_confirmed"

    chosen_frame = seed_frame
    for idx in range(seed_frame + 1, min(ffc, win_end + 1)):
        back_score = _foot_ground_score(back_ank, back_toe, idx, hold, win_start, win_end, dt)
        if back_score < 2:
            break
        chosen_frame = idx

    if best_edge is not None:
        _, edge_frame, edge_strength = best_edge
        edge_override_gap = min(
            int(timing["edge_override_gap"]),
            max(1, hold),
        )
        if int(chosen_frame) - int(edge_frame) <= edge_override_gap:
            chosen_frame = int(edge_frame)
            chosen_confidence = round(min(0.85, 0.35 + (0.55 * edge_strength)), 2)
            chosen_method = "back_foot_support_edge"
        else:
            chosen_confidence = 0.0
            chosen_method = "simple_grounded_bfc"
    else:
        chosen_confidence = 0.0
        chosen_method = "simple_grounded_bfc"

    if approach_speed is not None and band_end > band_start:
        speed = np.asarray(approach_speed, dtype=float)
        speed = np.nan_to_num(speed, nan=0.0)
        speed_ref = float(np.percentile(speed[band_start : band_end + 1], 60))
        for idx in range(band_end, band_start - 1, -1):
            back_score = _foot_ground_score(back_ank, back_toe, idx, hold, win_start, win_end, dt)
            front_score = _foot_ground_score(front_ank, front_toe, idx, hold, win_start, win_end, dt)
            if back_score < 1 or front_score >= 2:
                continue
            if float(speed[idx]) < speed_ref:
                continue
            if idx > int(chosen_frame):
                chosen_frame = int(idx)
                chosen_confidence = max(float(chosen_confidence), 0.45)
            break

    return int(chosen_frame), float(chosen_confidence), chosen_method


def sanitize_bfc_frame(
    *,
    bfc: int,
    ffc: int,
    front_side: Optional[str],
    hold: int,
    win_start: int,
    win_end: int,
    dt: float,
    y_LA: np.ndarray,
    y_RA: np.ndarray,
    y_LFI: np.ndarray,
    y_RFI: np.ndarray,
) -> Tuple[int, bool]:
    if front_side == "left":
        front_ank, front_toe = y_LA, y_LFI
        back_ank, back_toe = y_RA, y_RFI
    elif front_side == "right":
        front_ank, front_toe = y_RA, y_RFI
        back_ank, back_toe = y_LA, y_LFI
    else:
        return int(bfc), False

    bfc = int(clamp(bfc, win_start, max(win_start, ffc - 1)))
    if (int(ffc) - int(bfc)) > max(1, hold - 1):
        return bfc, False
    front_score = _foot_ground_score(front_ank, front_toe, bfc, hold, win_start, win_end, dt)
    if front_score < 2:
        return bfc, False

    earliest = max(win_start, bfc - max(5, hold + 2))
    for idx in range(bfc - 1, earliest - 1, -1):
        back_score = _foot_ground_score(back_ank, back_toe, idx, hold, win_start, win_end, dt)
        candidate_front = _foot_ground_score(front_ank, front_toe, idx, hold, win_start, win_end, dt)
        if back_score >= 2 and candidate_front < 2:
            return idx, True

    return bfc, False
