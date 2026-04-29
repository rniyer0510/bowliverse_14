# app/workers/events/ffc_bfc.py
"""
PHASE-1 INVARIANTS (UPDATED FOR REAL POSE DATA):

1. Pelvis kinematics define WHEN braking begins.
2. Geometry defines WHEN contact actually occurs.
3. FFC = latest plausible landing frame before release, found by searching backward from release
   and confirmed with front-foot grounding plus back-foot support/unloading heuristics.
4. Kinematics and geometry never compete on the same frame.
5. If uncertain, return a conservative fallback with low confidence (never return {}).
"""

from typing import Dict, List, Optional, Tuple
import math
import numpy as np

from app.common.logger import get_logger
from app.workers.events.event_confidence import build_candidate, chain_quality, compact_candidates, annotate_detection_contract
from app.workers.events.signal_cache import build_signal_cache
from app.workers.events.timing_constants import foot_contact_timing

logger = get_logger(__name__)

# ------------------------------------------------------------
# MediaPipe landmark indices (LOCKED)
# ------------------------------------------------------------
LS, RS = 11, 12
LH, RH = 23, 24
LA, RA = 27, 28
LFI, RFI = 31, 32

MIN_VIS = 0.25
EPS = 1e-9


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _as_int(v) -> Optional[int]:
    try:
        return int(v)
    except Exception:
        return None


def _interp_nans(x: np.ndarray) -> np.ndarray:
    """Linear interpolation over NaNs for numerical stability."""
    x = x.astype(float)
    idx = np.arange(len(x))
    good = np.isfinite(x)
    if good.sum() < 2:  # Need at least 2 points for interp
        return np.full_like(x, np.nan)
    x[~good] = np.interp(idx[~good], idx[good], x[good])
    return x


def _moving_average(x: np.ndarray, k: int) -> np.ndarray:
    k = max(1, int(k))
    if k <= 1:
        return x
    if k % 2 == 0:
        k += 1
    w = np.ones(k, dtype=float) / float(k)
    return np.convolve(x, w, mode="same")


def _robust_percentile(x: np.ndarray, p: float) -> float:
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


def _midpoint(a, b):
    if a is None or b is None:
        return None
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _angle_of_vec(v):
    return math.atan2(v[1], v[0])


def _safe_angle(lm: list, i1: int, i2: int) -> float:
    p1 = _xy(lm, i1)
    p2 = _xy(lm, i2)
    if p1 is None or p2 is None:
        return float("nan")
    return _angle_of_vec((p2[0] - p1[0], p2[1] - p1[1]))


def _pelvis_xy(lm: list):
    return _midpoint(_xy(lm, LH), _xy(lm, RH))


def _series_y(pose_frames: List[Dict], idx: int) -> np.ndarray:
    y = np.full(len(pose_frames), np.nan, dtype=float)
    for i, fr in enumerate(pose_frames):
        lm = fr.get("landmarks") or []
        if not isinstance(lm, list):
            continue
        p = _xy(lm, idx)
        if p is not None and np.isfinite(p[1]):
            y[i] = p[1]
    return _interp_nans(y)


def _series_vis(pose_frames: List[Dict], idx: int) -> np.ndarray:
    vis = np.zeros(len(pose_frames), dtype=float)
    for i, fr in enumerate(pose_frames):
        lm = fr.get("landmarks") or []
        if not isinstance(lm, list):
            continue
        vis[i] = _vis(lm, idx)
    return vis


# ------------------------------------------------------------
# Geometry: grounded scoring (robust to pose noise)
# ------------------------------------------------------------
def _foot_ground_score(
    y_ank: np.ndarray,
    y_toe: np.ndarray,
    t: int,
    hold: int,
    win0: int,
    win1: int,
    dt: float,
) -> int:
    """
    Return grounded score in {0,1,2,3} using three cues:
      1) near-low position
      2) low vertical velocity
      3) low jitter (MAD)
    We accept grounded if score >= 2 (NOT all 3), to tolerate pose noise.
    """
    n = len(y_ank)
    if t < 0 or t + hold >= n:
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

    # 1) Near-low (y increases downward)
    # Use 85th percentile as "near-ground" reference; small slack for normalization noise.
    low_a = _robust_percentile(hist_a, 85)
    low_t = _robust_percentile(hist_t, 85)
    near_low = (np.median(a_seg) >= low_a - 0.01) and (np.median(t_seg) >= low_t - 0.01)

    # 2) Low vertical velocity (but avoid over-tight gating)
    rng_a = max(_robust_percentile(hist_a, 90) - _robust_percentile(hist_a, 10), 1e-6)
    rng_t = max(_robust_percentile(hist_t, 90) - _robust_percentile(hist_t, 10), 1e-6)

    dy_a = np.diff(a_seg) / max(dt, 1e-6)
    dy_t = np.diff(t_seg) / max(dt, 1e-6)

    v_ok = (np.median(np.abs(dy_a)) <= 0.18 * (rng_a / max(dt, 1e-6))) and \
           (np.median(np.abs(dy_t)) <= 0.18 * (rng_t / max(dt, 1e-6)))

    # 3) Low jitter (MAD)
    jit_ok = (_robust_mad(a_seg) <= 0.15 * rng_a) and (_robust_mad(t_seg) <= 0.15 * rng_t)

    score = 0
    score += 1 if near_low else 0
    score += 1 if v_ok else 0
    score += 1 if jit_ok else 0
    return score


def _is_grounded(
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
    """
    True if foot is grounded in any of the previous `lookback` frames.
    Used to allow natural unloading of back foot during front-foot braking.
    """
    j0 = max(win0, t - lookback)
    for j in range(t - 1, j0 - 1, -1):
        if _is_grounded(y_ank, y_toe, j, hold, win0, win1, dt):
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
    """
    Reward local temporal stability around a candidate frame.
    Stronger candidates stay plausibly grounded for a few neighboring frames,
    not just on one isolated pose sample.
    """
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
    """
    Reward the contact edge rather than the settled grounded state.

    A strong edge looks like:
      1) the foot becomes grounded here, not several frames earlier
      2) downward motion decelerates sharply into the grounded window
      3) the grounded state remains stable for a short band after contact
      4) pelvis jerk provides a small kinematic confirmation when available
    """
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


def _pick_ffc_backward_from_release(
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
    pelvis_jerk: Optional[np.ndarray] = None,
) -> Tuple[Optional[int], Optional[str], List[Dict], float]:
    """
    Search backward from the release-side window for the latest plausible front-foot contact.
    Kinetic-chain timing defines the neighborhood, but the search must keep a
    minimum release-side band so late pelvis_on does not exclude real contact.
    grounding heuristics pick the exact frame.
    """
    candidates: List[Dict] = []
    ranked: List[Tuple[float, int, str, Dict]] = []
    radius = max(1, hold // 2)

    for i in range(win_end - hold, search_start - 1, -1):
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

        left_front_ok = left_grounded and (right_grounded or right_recent)
        right_front_ok = right_grounded and (left_grounded or left_recent)

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
            chain_anchor = min(max(pelvis_on, search_start), win_end)
            release_bias = 0.06 * ((i - chain_anchor) / max(1, win_end - chain_anchor))
            candidate = _candidate_case(
                frame=i,
                front_label="left",
                front_score=left_score,
                back_score=right_score,
                back_recent=right_recent,
                method="release_backward_chain_grounding",
                score_bonus=(0.02 * left_stability) + (0.04 * left_edge) + release_bias,
            )
            if candidate is not None:
                candidate["contact_edge_strength"] = round(left_edge, 3)
                candidates.append(candidate)
                ranked.append(
                    (
                        float(candidate.get("score") or 0.0) + (0.06 * left_edge),
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
            chain_anchor = min(max(pelvis_on, search_start), win_end)
            release_bias = 0.06 * ((i - chain_anchor) / max(1, win_end - chain_anchor))
            candidate = _candidate_case(
                frame=i,
                front_label="right",
                front_score=right_score,
                back_score=left_score,
                back_recent=left_recent,
                method="release_backward_chain_grounding",
                score_bonus=(0.02 * right_stability) + (0.04 * right_edge) + release_bias,
            )
            if candidate is not None:
                candidate["contact_edge_strength"] = round(right_edge, 3)
                candidates.append(candidate)
                ranked.append(
                    (
                        float(candidate.get("score") or 0.0) + (0.06 * right_edge),
                        i,
                        "right",
                        candidate,
                    )
                )

    if ranked:
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        if (
            preferred_front_side in {"left", "right"}
            and len(ranked) >= 2
            and ranked[0][1] == ranked[1][1]
            and ranked[0][2] != ranked[1][2]
            and abs(ranked[0][0] - ranked[1][0]) <= 0.08
        ):
            for score, frame, side, candidate in ranked[:2]:
                if side == preferred_front_side:
                    return (
                        frame,
                        side,
                        candidates,
                        float(candidate.get("confidence") or 0.62),
                    )
        _, frame, front_side, candidate = ranked[0]
        return (
            frame,
            front_side,
            candidates,
            float(candidate.get("confidence") or 0.62),
        )

    return None, None, candidates, 0.0


def _detection_context(
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


def _annotated_result(payload: Dict[str, Dict]) -> Dict[str, Dict]:
    return annotate_detection_contract(payload)


def _ffc_search_start(
    *,
    win_start: int,
    win_end: int,
    pelvis_on: int,
    fps: float,
    hold: int,
) -> int:
    """
    Pelvis activation is a soft prior, not the true lower bound for FFC.

    Keep a meaningful pre-pelvis lead band so contact-edge candidates can still
    be found when pelvis_on fires late in behind-camera or distant clips.
    """
    timing = foot_contact_timing(fps)
    min_ffc_release_band = max(hold + 1, int(timing["min_ffc_release_band"]))
    pre_pelvis_lead = max(hold + 2, int(timing["pre_pelvis_lead"]))
    lower_bound = pelvis_on - pre_pelvis_lead
    return max(win_start, min(lower_bound, win_end - min_ffc_release_band))


def _pick_bfc_backward_from_ffc(
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
    approach_speed: np.ndarray,
    pelvis_jerk: Optional[np.ndarray] = None,
) -> Tuple[int, float, str]:
    """
    Prefer a recent back-foot support edge near FFC when one is visible.
    Otherwise, fall back to the legacy latest-grounded-support rule.
    """
    if front_side == "left":
        back_ank, back_toe = y_RA, y_RFI
        back_ank_vis, back_toe_vis = vis_RA, vis_RFI
        front_ank, front_toe = y_LA, y_LFI
    elif front_side == "right":
        back_ank, back_toe = y_LA, y_LFI
        back_ank_vis, back_toe_vis = vis_LA, vis_LFI
        front_ank, front_toe = y_RA, y_RFI
    else:
        return _clamp(ffc - 1, win_start, ffc), 0.0, "context_pre_ffc"

    timing = foot_contact_timing(fps)
    recent_band = max(hold + 2, int(timing["recent_support_band"]))
    edge_search_start = max(win_start, ffc - recent_band)
    best_edge: Optional[Tuple[float, int, float]] = None
    recent_vis = []
    for idx in range(edge_search_start, ffc):
        if idx < len(back_ank_vis):
            recent_vis.append(float(min(back_ank_vis[idx], back_toe_vis[idx])))
    median_recent_vis = float(np.median(recent_vis)) if recent_vis else 0.0

    if median_recent_vis >= MIN_VIS:
        denom = max(1, ffc - edge_search_start)
        for idx in range(edge_search_start, ffc):
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

    earliest = max(win_start, ffc - max(3, hold + 1))
    seed_frame: Optional[int] = None

    for idx in range(ffc - 1, earliest - 1, -1):
        back_score = _foot_ground_score(back_ank, back_toe, idx, hold, win_start, win_end, dt)
        if back_score >= 2:
            seed_frame = idx
            continue
        if seed_frame is not None:
            break

    if seed_frame is None:
        return _clamp(ffc - 1, win_start, ffc), 0.0, "no_ground_confirmed"

    chosen_frame = seed_frame
    for idx in range(seed_frame + 1, min(ffc, win_end + 1)):
        back_score = _foot_ground_score(back_ank, back_toe, idx, hold, win_start, win_end, dt)
        if back_score < 2:
            break
        chosen_frame = idx

    if best_edge is not None:
        _, edge_frame, edge_strength = best_edge
        edge_override_gap = max(hold + 1, int(timing["edge_override_gap"]))
        if int(chosen_frame) - int(edge_frame) <= edge_override_gap:
            confidence = min(0.85, 0.35 + (0.55 * edge_strength))
            return int(edge_frame), round(confidence, 2), "back_foot_support_edge"

    return int(chosen_frame), 0.0, "simple_grounded_bfc"


def _sanitize_bfc_frame(
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
    """
    BFC must not be a frame that already looks like FFC.
    If the chosen frame shows clear front-foot contact, step backward one frame
    at a time until we find the latest earlier frame with grounded back-foot
    support and no clear front-foot contact.
    """
    if front_side == "left":
        front_ank, front_toe = y_LA, y_LFI
        back_ank, back_toe = y_RA, y_RFI
    elif front_side == "right":
        front_ank, front_toe = y_RA, y_RFI
        back_ank, back_toe = y_LA, y_LFI
    else:
        return int(bfc), False

    bfc = int(_clamp(bfc, win_start, max(win_start, ffc - 1)))
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


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def detect_ffc_bfc(
    pose_frames: List[Dict],
    hand: str,
    release_frame: int,
    delivery_window: Tuple[int, int],  # ignored for FFC windowing
    fps: Optional[float] = None,
    **_ignored,
) -> Dict[str, Dict]:

    n = len(pose_frames)
    if n < 10:
        logger.warning("[FFC/BFC] Too few frames")
        return {}

    rel = _as_int(release_frame)
    if rel is None:
        logger.error("[FFC/BFC] Missing release frame")
        return {}

    # Robust FPS handling
    try:
        fps_f = float(fps) if fps else 0.0
    except Exception:
        fps_f = 0.0
    fps_was_defaulted = False
    if fps_f <= 1e-3:
        fps_f = 30.0
        fps_was_defaulted = True
    fps_f = max(1.0, fps_f)
    timing = foot_contact_timing(fps_f)
    fps_f = float(timing["fps"])
    dt = float(timing["dt"])

    # ------------------------------------------------------------
    # Window upstream of release
    # ------------------------------------------------------------
    lookback = int(timing["lookback"])
    hold = int(timing["hold"])
    smooth_k = int(timing["smooth_k"])

    unclamped_start = rel - lookback
    unclamped_end = rel - 2
    win_start = _clamp(unclamped_start, 0, n - 1)
    win_end   = _clamp(unclamped_end, win_start, n - 1)
    window_was_clipped = win_start != unclamped_start or win_end != unclamped_end
    detection_context = _detection_context(
        fps=fps_f,
        fps_was_defaulted=fps_was_defaulted,
        window_was_clipped=window_was_clipped,
        win_start=win_start,
        win_end=win_end,
    )

    logger.info(f"[FFC/BFC][WINDOW] derived=[{win_start}..{win_end}] release={rel} fps={fps_f:.2f}")

    if win_end <= win_start + 2:
        logger.warning("[FFC/BFC] Degenerate window")
        return _annotated_result({
            "ffc": {"frame": win_start, "confidence": 0.15, "method": "degenerate_window"},
            "detection_context": detection_context,
        })

    cache = build_signal_cache(
        pose_frames=pose_frames,
        hand=hand,
        fps=fps_f,
    )

    # ------------------------------------------------------------
    # Shared kinematic cache
    # ------------------------------------------------------------
    pelvis_xy = cache["pelvis_centre_xy"]
    px = pelvis_xy[:, 0]
    py = pelvis_xy[:, 1]
    vis_ok = np.asarray(cache["hips_vis_raw"] >= MIN_VIS, dtype=bool)
    valid_pelvis_count = int(np.sum(np.isfinite(px)))

    if valid_pelvis_count < max(hold, 6):
        logger.warning("[FFC/BFC] Insufficient valid pelvis landmarks")
        return _annotated_result({
            "ffc": {"frame": win_start, "confidence": 0.10, "method": "insufficient_landmarks"},
            "detection_context": detection_context,
        })

    v_lin = _moving_average(
        np.nan_to_num(cache["pelvis_linear_speed"], nan=0.0),
        smooth_k,
    )
    w_rot = _moving_average(
        np.nan_to_num(cache["hip_line_angular_velocity"], nan=0.0),
        smooth_k,
    )
    pelvis_jerk = _moving_average(
        np.nan_to_num(cache["pelvis_jerk"], nan=0.0),
        smooth_k,
    )

    R = w_rot / (v_lin + EPS)

    # ------------------------------------------------------------
    # Pelvis activity onset (kinematics)
    # ------------------------------------------------------------
    R_on = _robust_percentile(R[win_start:win_end + 1], 70)
    pelvis_on = None

    # Backward scan: find last region where R exceeds threshold and hips readable
    for i in range(win_end, win_start, -1):
        if vis_ok[i] and R[i] > R_on:
            pelvis_on = i
        elif pelvis_on is not None:
            break

    if pelvis_on is None:
        logger.warning("[FFC/BFC] Pelvis never activated; using win_start")
        pelvis_on = win_start

    logger.info(f"[FFC/BFC][PELVIS_ON] idx={pelvis_on}")

    # ------------------------------------------------------------
    # Geometry forward lock (RELAXED): front grounded + back grounded OR recently grounded
    # ------------------------------------------------------------
    y_LA = _series_y(pose_frames, LA)
    y_RA = _series_y(pose_frames, RA)
    y_LFI = _series_y(pose_frames, LFI)
    y_RFI = _series_y(pose_frames, RFI)
    vis_LA = _series_vis(pose_frames, LA)
    vis_RA = _series_vis(pose_frames, RA)
    vis_LFI = _series_vis(pose_frames, LFI)
    vis_RFI = _series_vis(pose_frames, RFI)

    # If ankle data is missing, fall back conservatively to pelvis_on (low conf)
    if (not np.any(np.isfinite(y_LA)) and not np.any(np.isfinite(y_RA))) or (not np.any(np.isfinite(y_LFI)) and not np.any(np.isfinite(y_RFI))):
        logger.warning("[FFC/BFC] No valid foot landmarks; pelvis fallback")
        ffc = pelvis_on
        bfc = _clamp(ffc - max(3, hold), win_start, ffc)
        logger.info(f"[FFC/BFC][RESULT] CHOSEN_FFC_FRAME={ffc}")
        return _annotated_result({
            "ffc": {"frame": int(ffc), "confidence": 0.20, "method": "no_foot_data_fallback"},
            "bfc": {"frame": int(bfc), "confidence": 0.20, "method": "no_foot_data_fallback"},
            "detection_context": detection_context,
        })

    # Search backward from release for the latest plausible landing frame.
    # Kinetic-chain timing localizes the neighborhood; grounding heuristics
    # choose the exact FFC frame inside that neighborhood.
    back_recent = int(timing["back_recent"])
    ffc = None
    front_side = None
    ffc_candidates: List[Dict] = []
    ffc_search_start = _ffc_search_start(
        win_start=win_start,
        win_end=win_end,
        pelvis_on=pelvis_on,
        fps=fps_f,
        hold=hold,
    )
    if ffc_search_start < pelvis_on:
        logger.info(
            "[FFC/BFC][FFC_SEARCH] widening lower bound from pelvis_on=%s to search_start=%s",
            pelvis_on,
            ffc_search_start,
        )

    preferred_front_side = "left" if str(hand or "").upper().startswith("R") else "right"
    ffc, front_side, ffc_candidates, ffc_confidence = _pick_ffc_backward_from_release(
        search_start=ffc_search_start,
        pelvis_on=pelvis_on,
        preferred_front_side=preferred_front_side,
        win_end=win_end,
        hold=hold,
        win_start=win_start,
        dt=dt,
        back_recent=back_recent,
        y_LA=y_LA,
        y_RA=y_RA,
        y_LFI=y_LFI,
        y_RFI=y_RFI,
        pelvis_jerk=pelvis_jerk,
    )

    # ------------------------------------------------------------
    # Fallback ladder (never return empty)
    # ------------------------------------------------------------
    if ffc is None:
        # 1) Relax further: any single grounded foot after pelvis_on
        for i in range(win_end - hold, ffc_search_start - 1, -1):
            if _is_grounded(y_LA, y_LFI, i, hold, win_start, win_end, dt) or _is_grounded(y_RA, y_RFI, i, hold, win_start, win_end, dt):
                ffc = i
                candidate = build_candidate(
                    frame=i,
                    method="single_foot_fallback",
                    confidence=0.22,
                    score=1.0,
                )
                if candidate is not None:
                    ffc_candidates.append(candidate)
                logger.warning(f"[FFC/BFC][FALLBACK] single_foot frame={ffc}")
                bfc = _clamp(ffc - max(3, hold), win_start, ffc)
                corrected_bfc, corrected = _sanitize_bfc_frame(
                    bfc=bfc,
                    ffc=ffc,
                    front_side=preferred_front_side,
                    hold=hold,
                    win_start=win_start,
                    win_end=win_end,
                    dt=dt,
                    y_LA=y_LA,
                    y_RA=y_RA,
                    y_LFI=y_LFI,
                    y_RFI=y_RFI,
                )
                if corrected:
                    logger.info("[FFC/BFC][BFC_CORRECT] single_foot %s -> %s", bfc, corrected_bfc)
                    bfc = corrected_bfc
                chain = chain_quality(
                    bfc_frame=bfc,
                    ffc_frame=ffc,
                    uah_frame=None,
                    release_frame=rel,
                    bfc_confidence=0.22,
                    ffc_confidence=0.22,
                )
                logger.info(f"[FFC/BFC][RESULT] CHOSEN_FFC_FRAME={ffc}")
                return _annotated_result({
                    "ffc": {
                        "frame": int(ffc),
                        "confidence": 0.22,
                        "method": "single_foot_fallback",
                        "candidates": compact_candidates(ffc_candidates),
                        "window": [int(win_start), int(win_end)],
                    },
                    "bfc": {
                        "frame": int(bfc),
                        "confidence": 0.22,
                        "method": "single_foot_fallback",
                    },
                    "event_chain": chain,
                    "detection_context": detection_context,
                })

        # 2) Ultimate fallback: 3/4 into pelvis->release window (must obey win_end fence)
        ffc = pelvis_on + int(0.75 * (win_end - pelvis_on))
        ffc = _clamp(ffc, pelvis_on, win_end)
        candidate = build_candidate(
            frame=ffc,
            method="ultimate_fallback",
            confidence=0.15,
            score=0.0,
        )
        if candidate is not None:
            ffc_candidates.append(candidate)
        logger.warning(f"[FFC/BFC][FALLBACK] ultimate_3quarter frame={ffc}")

        bfc = _clamp(ffc - max(3, hold), win_start, ffc)
        corrected_bfc, corrected = _sanitize_bfc_frame(
            bfc=bfc,
            ffc=ffc,
            front_side=preferred_front_side,
            hold=hold,
            win_start=win_start,
            win_end=win_end,
            dt=dt,
            y_LA=y_LA,
            y_RA=y_RA,
            y_LFI=y_LFI,
            y_RFI=y_RFI,
        )
        if corrected:
            logger.info("[FFC/BFC][BFC_CORRECT] ultimate_fallback %s -> %s", bfc, corrected_bfc)
            bfc = corrected_bfc
        chain = chain_quality(
            bfc_frame=bfc,
            ffc_frame=ffc,
            uah_frame=None,
            release_frame=rel,
            bfc_confidence=0.15,
            ffc_confidence=0.15,
        )
        logger.info(f"[FFC/BFC][RESULT] CHOSEN_FFC_FRAME={ffc}")
        return _annotated_result({
            "ffc": {
                "frame": int(ffc),
                "confidence": 0.15,
                "method": "ultimate_fallback",
                "candidates": compact_candidates(ffc_candidates),
                "window": [int(win_start), int(win_end)],
            },
            "bfc": {"frame": int(bfc), "confidence": 0.15, "method": "ultimate_fallback"},
            "event_chain": chain,
            "detection_context": detection_context,
        })

    # ------------------------------------------------------------
    # Result
    # ------------------------------------------------------------
    logger.info(f"[FFC/BFC][RESULT] CHOSEN_FFC_FRAME={ffc}")

    bfc, bfc_confidence, bfc_method = _pick_bfc_backward_from_ffc(
        ffc=ffc,
        front_side=front_side,
        hold=hold,
        win_start=win_start,
        win_end=win_end,
        dt=dt,
        fps=fps_f,
        y_LA=y_LA,
        y_RA=y_RA,
        y_LFI=y_LFI,
        y_RFI=y_RFI,
        vis_LA=vis_LA,
        vis_RA=vis_RA,
        vis_LFI=vis_LFI,
        vis_RFI=vis_RFI,
        approach_speed=v_lin,
        pelvis_jerk=pelvis_jerk,
    )
    corrected_bfc, corrected = _sanitize_bfc_frame(
        bfc=bfc,
        ffc=ffc,
        front_side=front_side or preferred_front_side,
        hold=hold,
        win_start=win_start,
        win_end=win_end,
        dt=dt,
        y_LA=y_LA,
        y_RA=y_RA,
        y_LFI=y_LFI,
        y_RFI=y_RFI,
    )
    if corrected:
        logger.info("[FFC/BFC][BFC_CORRECT] %s %s -> %s", bfc_method, bfc, corrected_bfc)
        bfc = corrected_bfc
        bfc_method = f"{bfc_method}_front_contact_corrected"
        bfc_confidence = max(0.0, float(bfc_confidence) - 0.05)
    ffc_confidence = max(0.45, ffc_confidence or 0.62)
    chain = chain_quality(
        bfc_frame=bfc,
        ffc_frame=ffc,
        uah_frame=None,
        release_frame=rel,
        bfc_confidence=bfc_confidence,
        ffc_confidence=ffc_confidence,
    )

    return _annotated_result({
        "ffc": {
            "frame": int(ffc),
            "confidence": ffc_confidence,
            "method": "release_backward_chain_grounding",
            "candidates": compact_candidates(ffc_candidates),
            "window": [int(win_start), int(win_end)],
        },
        "bfc": {"frame": int(bfc), "confidence": bfc_confidence, "method": bfc_method},
        "event_chain": chain,
        "detection_context": detection_context,
    })
