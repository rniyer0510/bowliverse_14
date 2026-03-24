"""
Elbow legality evaluation — ActionLab V14.

Primary rule:
- Use measured elbow extension whenever a release-anchored elbow window is dense enough.

Fallback rule:
- If extension cannot be measured reliably, judge legality from the smoothness of the
  bowling-arm flow across FFC -> UAH -> release.
"""

from typing import Any, Dict, List, Optional, Tuple
import math

THRESH_LEGAL = 18.0
THRESH_BORDERLINE = 22.0

RELEASE_TRIM_FRAMES = 2
MIN_SAMPLES = 5
BASELINE_MAX_SAMPLES = 4
PEAK_Q = 90
PRE_RELEASE_LOOKBACK = 10
POST_RELEASE_GRACE = 12

BASELINE_VEL_MAX = 3.0  # deg/frame
FLOW_MIN_SAMPLES = 6
FLOW_MIN_SPAN_FRAMES = 5
FLOW_SIGN_EPS = 1.0
FLOW_MAX_RATE = 12.0
FLOW_MAX_JERK = 10.0
FLOW_MAX_SIGN_FLIPS = 1

LOW_VIS_RESCUE_MIN_VIS = 0.10
LOW_VIS_SHOULDER_MIN_VIS = 0.50
LOW_VIS_CLEAR_MIN_SAMPLES = 6
WEAK_WINDOW_MARGIN_DEG = 2.0

LS, LE, LW = 11, 13, 15
RS, RE, RW = 12, 14, 16
L_PINKY, R_PINKY = 17, 18
L_INDEX, R_INDEX = 19, 20
L_THUMB, R_THUMB = 21, 22


# -------------------------------------------------
# Utilities
# -------------------------------------------------

def _safe_int(x: Any) -> Optional[int]:
    try:
        return int(x) if x is not None else None
    except Exception:
        return None


def _finite(x: Any) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except Exception:
        return False


def _extract_event_frame(events: Dict[str, Any], key: str) -> Optional[int]:
    node = events.get(key) if isinstance(events, dict) else None
    if isinstance(node, dict):
        return _safe_int(node.get("frame"))
    return None


def _point_vis(pt: Any) -> float:
    try:
        if isinstance(pt, dict):
            return float(pt.get("visibility", 0.0))
        return float(getattr(pt, "visibility", 0.0))
    except Exception:
        return 0.0


def _point_xyz(pt: Any) -> Optional[Tuple[float, float, float]]:
    try:
        if isinstance(pt, dict):
            return (
                float(pt.get("x", 0.0)),
                float(pt.get("y", 0.0)),
                float(pt.get("z", 0.0)),
            )
        return (
            float(getattr(pt, "x", 0.0)),
            float(getattr(pt, "y", 0.0)),
            float(getattr(pt, "z", 0.0)),
        )
    except Exception:
        return None


def _angle_3pt(a: Tuple[float, float, float], b: Tuple[float, float, float], c: Tuple[float, float, float]) -> Optional[float]:
    bax, bay, baz = a[0] - b[0], a[1] - b[1], a[2] - b[2]
    bcx, bcy, bcz = c[0] - b[0], c[1] - b[1], c[2] - b[2]
    mag1 = math.sqrt(bax * bax + bay * bay + baz * baz)
    mag2 = math.sqrt(bcx * bcx + bcy * bcy + bcz * bcz)
    if mag1 < 1e-6 or mag2 < 1e-6:
        return None
    dot = bax * bcx + bay * bcy + baz * bcz
    cosv = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cosv))


def _weighted_centroid(points: List[Tuple[Tuple[float, float, float], float]]) -> Optional[Tuple[float, float, float]]:
    weight_sum = sum(w for _, w in points)
    if weight_sum < 1e-9:
        return None
    return (
        sum(p[0] * w for p, w in points) / weight_sum,
        sum(p[1] * w for p, w in points) / weight_sum,
        sum(p[2] * w for p, w in points) / weight_sum,
    )


def _percentile(vals: List[float], q: int) -> float:
    xs = sorted(vals)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (q / 100.0)
    f = int(math.floor(k))
    c = int(math.ceil(k))
    return xs[f] if f == c else xs[f] + (xs[c] - xs[f]) * (k - f)


def _mad_filter(vals: List[float], z: float = 6.0) -> List[float]:
    if len(vals) < 5:
        return vals
    med = sorted(vals)[len(vals) // 2]
    dev = [abs(v - med) for v in vals]
    mad = sorted(dev)[len(dev) // 2]
    if mad < 1e-9:
        return vals
    out = []
    for v in vals:
        mz = 0.6745 * (v - med) / mad
        if abs(mz) <= z:
            out.append(v)
    return out if len(out) >= 3 else vals


def _collect_rows(
    elbow_signal: List[Dict[str, Any]],
    start_frame: int,
    end_frame: int,
) -> List[tuple[int, float]]:
    rows: List[tuple[int, float]] = []
    for it in elbow_signal:
        f = _safe_int(it.get("frame")) if isinstance(it, dict) else None
        if f is None or f < start_frame or f > end_frame:
            continue
        if not it.get("valid", False):
            continue
        a = it.get("angle_deg")
        if not _finite(a):
            continue
        rows.append((f, float(a)))
    return rows


def _sign_flips(rates: List[float]) -> int:
    flips = 0
    prev_sign = 0
    for rate in rates:
        if abs(rate) < FLOW_SIGN_EPS:
            continue
        sign = 1 if rate > 0 else -1
        if prev_sign and sign != prev_sign:
            flips += 1
        prev_sign = sign
    return flips


def _flow_based_legality(
    elbow_signal: List[Dict[str, Any]],
    events: Dict[str, Any],
    *,
    primary_reason: str,
    primary_debug: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ffc = _extract_event_frame(events, "ffc")
    uah = _extract_event_frame(events, "uah")
    release = _extract_event_frame(events, "release")

    start_frame = ffc if ffc is not None else uah
    if start_frame is None or release is None or start_frame >= release:
        return {
            "verdict": "UNKNOWN",
            "confidence": 0.20,
            "reason": "insufficient_flow_window",
            "flow": {
                "source": "ffc_to_release",
                "start_frame": start_frame,
                "end_frame": release,
            },
            "debug": primary_debug or {},
        }

    rows = _collect_rows(elbow_signal, start_frame, release)
    span_frames = rows[-1][0] - rows[0][0] if rows else 0
    if len(rows) < FLOW_MIN_SAMPLES or span_frames < FLOW_MIN_SPAN_FRAMES:
        return {
            "verdict": "UNKNOWN",
            "confidence": 0.25,
            "reason": "insufficient_flow_signal",
            "flow": {
                "source": "ffc_to_release",
                "start_frame": start_frame,
                "end_frame": release,
                "valid_samples": len(rows),
                "span_frames": span_frames,
            },
            "debug": primary_debug or {},
        }

    rates: List[float] = []
    for (f0, a0), (f1, a1) in zip(rows, rows[1:]):
        gap = max(1, f1 - f0)
        rates.append((a1 - a0) / gap)

    if len(rates) < 2:
        return {
            "verdict": "UNKNOWN",
            "confidence": 0.25,
            "reason": "insufficient_flow_signal",
            "flow": {
                "source": "ffc_to_release",
                "start_frame": start_frame,
                "end_frame": release,
                "valid_samples": len(rows),
                "span_frames": span_frames,
            },
            "debug": primary_debug or {},
        }

    abs_rates = [abs(r) for r in rates]
    jerks = [abs(rates[i] - rates[i - 1]) for i in range(1, len(rates))]
    sign_flips = _sign_flips(rates)
    p90_rate = _percentile(abs_rates, 90)
    p90_jerk = _percentile(jerks, 90) if jerks else 0.0

    irregular = (
        p90_rate > FLOW_MAX_RATE
        or p90_jerk > FLOW_MAX_JERK
        or sign_flips > FLOW_MAX_SIGN_FLIPS
    )

    return {
        "verdict": "ILLEGAL" if irregular else "LEGAL",
        "confidence": 0.45 if irregular else 0.50,
        "reason": "flow_irregular_fallback" if irregular else "flow_consistent_fallback",
        "extension_deg": None,
        "flow": {
            "source": "ffc_to_release",
            "start_frame": start_frame,
            "end_frame": release,
            "valid_samples": len(rows),
            "p90_rate_deg_per_frame": round(p90_rate, 2),
            "p90_jerk_deg_per_frame": round(p90_jerk, 2),
            "sign_flips": sign_flips,
        },
        "debug": {
            "primary_reason": primary_reason,
            **(primary_debug or {}),
        },
    }


def _select_measurement_rows(
    elbow_signal: List[Dict[str, Any]],
    uah: Optional[int],
    release_used: int,
) -> tuple[int, int, List[float], Dict[str, Any]]:
    max_frame = max((_safe_int(it.get("frame")) for it in elbow_signal if isinstance(it, dict)), default=release_used)
    release_start = max(0, release_used - PRE_RELEASE_LOOKBACK)
    release_rows = [a for _, a in _collect_rows(elbow_signal, release_start, release_used)]
    if len(release_rows) >= MIN_SAMPLES:
        return (
            release_start,
            release_used,
            release_rows,
            {
                "window_mode": "release_anchored",
                "window_start": release_start,
                "window_end": release_used,
                "valid_samples": len(release_rows),
            },
        )

    grace_limit = min(max_frame or release_used, release_used + POST_RELEASE_GRACE)
    for candidate_end in range(release_used + 1, grace_limit + 1):
        candidate_start = max(0, candidate_end - PRE_RELEASE_LOOKBACK)
        candidate_rows = [a for _, a in _collect_rows(elbow_signal, candidate_start, candidate_end)]
        if len(candidate_rows) >= MIN_SAMPLES:
            return (
                candidate_start,
                candidate_end,
                candidate_rows,
                {
                    "window_mode": "release_grace_rescue",
                    "window_start": candidate_start,
                    "window_end": candidate_end,
                    "release_used": release_used,
                    "grace_frames": candidate_end - release_used,
                    "valid_samples": len(candidate_rows),
                },
            )

    primary_rows = (
        [a for _, a in _collect_rows(elbow_signal, uah, release_used)]
        if uah is not None and uah <= release_used
        else []
    )
    if len(primary_rows) >= MIN_SAMPLES:
        return (
            uah if uah is not None else release_start,
            release_used,
            primary_rows,
            {
                "window_mode": "uah_to_release",
                "window_start": uah,
                "window_end": release_used,
                "valid_samples": len(primary_rows),
            },
        )

    return (
        release_start,
        release_used,
        release_rows,
        {
            "window_mode": "release_anchored_sparse",
            "window_start": release_start,
            "window_end": release_used,
            "uah_window_start": uah,
            "uah_window_end": release_used,
            "uah_window_samples": len(primary_rows),
            "valid_samples": len(release_rows),
        },
    )


def _compute_low_visibility_signal(
    pose_frames: List[Dict[str, Any]],
    hand: str,
    *,
    min_vis: float = LOW_VIS_RESCUE_MIN_VIS,
) -> List[Dict[str, Any]]:
    hand = (hand or "R").upper()
    if hand == "R":
        s_idx, e_idx, w_idx = RS, RE, RW
        extras = [(R_INDEX, 0.20), (R_PINKY, 0.15), (R_THUMB, 0.10)]
    else:
        s_idx, e_idx, w_idx = LS, LE, LW
        extras = [(L_INDEX, 0.20), (L_PINKY, 0.15), (L_THUMB, 0.10)]

    signal: List[Dict[str, Any]] = []
    prev_angle: Optional[float] = None

    for i, item in enumerate(pose_frames or []):
        frame_idx = int(item.get("frame", i)) if isinstance(item, dict) else i
        landmarks = item.get("landmarks") if isinstance(item, dict) else None
        if not isinstance(landmarks, list) or len(landmarks) <= max(s_idx, e_idx, w_idx):
            signal.append({"frame": frame_idx, "angle_deg": None, "valid": False})
            continue

        shoulder = landmarks[s_idx]
        elbow = landmarks[e_idx]
        if _point_vis(shoulder) < LOW_VIS_SHOULDER_MIN_VIS or _point_vis(elbow) < min_vis:
            signal.append({"frame": frame_idx, "angle_deg": None, "valid": False})
            continue

        shoulder_xyz = _point_xyz(shoulder)
        elbow_xyz = _point_xyz(elbow)
        if shoulder_xyz is None or elbow_xyz is None:
            signal.append({"frame": frame_idx, "angle_deg": None, "valid": False})
            continue

        points: List[Tuple[Tuple[float, float, float], float]] = []
        wrist = landmarks[w_idx]
        if _point_vis(wrist) >= min_vis:
            wrist_xyz = _point_xyz(wrist)
            if wrist_xyz is not None:
                points.append((wrist_xyz, 0.55))

        for idx, weight in extras:
            if idx >= len(landmarks):
                continue
            pt = landmarks[idx]
            if _point_vis(pt) < min_vis:
                continue
            pt_xyz = _point_xyz(pt)
            if pt_xyz is not None:
                points.append((pt_xyz, weight))

        distal = _weighted_centroid(points)
        if distal is None:
            signal.append({"frame": frame_idx, "angle_deg": None, "valid": False})
            continue

        angle = _angle_3pt(shoulder_xyz, elbow_xyz, distal)
        if angle is None:
            signal.append({"frame": frame_idx, "angle_deg": None, "valid": False})
            continue

        if prev_angle is not None:
            delta = angle - prev_angle
            if abs(delta) > 25.0:
                angle = prev_angle + math.copysign(25.0, delta)

        prev_angle = angle
        signal.append({"frame": frame_idx, "angle_deg": angle, "valid": True})

    return signal


def _apply_low_visibility_rescue(
    pose_frames: List[Dict[str, Any]],
    hand: str,
    events: Dict[str, Any],
) -> Dict[str, Any]:
    rescue_signal = _compute_low_visibility_signal(pose_frames, hand)
    rescue = _compute_elbow_legality(rescue_signal, events)

    debug = dict(rescue.get("debug") or {})
    debug["rescue_min_vis"] = LOW_VIS_RESCUE_MIN_VIS
    debug["rescue_valid_samples"] = sum(1 for row in rescue_signal if row.get("valid"))

    extension = rescue.get("extension_deg")
    window_mode = debug.get("window_mode")

    if extension is not None:
        if rescue.get("verdict") in {"ILLEGAL", "BORDERLINE"}:
            return {
                "verdict": "SUSPECT",
                "confidence": 0.50,
                "reason": "low_visibility_rescue_suspect",
                "extension_deg": extension,
                "debug": {"rescue": rescue, **debug},
            }
        if (
            rescue.get("verdict") == "LEGAL"
            and window_mode == "release_anchored"
            and debug.get("valid_samples", 0) >= LOW_VIS_CLEAR_MIN_SAMPLES
        ):
            return {
                "verdict": "LEGAL",
                "confidence": 0.45,
                "reason": "low_visibility_rescue_legal",
                "extension_deg": extension,
                "debug": {"rescue": rescue, **debug},
            }
        return {
            "verdict": "SUSPECT",
            "confidence": 0.40,
            "reason": "low_visibility_rescue_inconclusive",
            "extension_deg": extension,
            "debug": {"rescue": rescue, **debug},
        }

    if rescue.get("verdict") == "ILLEGAL":
        return {
            "verdict": "SUSPECT",
            "confidence": 0.45,
            "reason": "low_visibility_flow_suspect",
            "debug": {"rescue": rescue, **debug},
        }

    if rescue.get("verdict") == "LEGAL" and rescue.get("reason") == "flow_consistent_fallback":
        flow = rescue.get("flow") or {}
        if flow.get("valid_samples", 0) >= FLOW_MIN_SAMPLES and flow.get("span_frames", 0) >= FLOW_MIN_SPAN_FRAMES:
            return {
                "verdict": "LEGAL",
                "confidence": 0.35,
                "reason": "low_visibility_flow_legal",
                "debug": {"rescue": rescue, **debug},
            }

    return {
        "verdict": "SUSPECT",
        "confidence": 0.35,
        "reason": "low_visibility_rescue_inconclusive",
        "debug": {"rescue": rescue, **debug},
    }


def _is_weak_primary_window(primary: Dict[str, Any]) -> bool:
    debug = primary.get("debug") or {}
    window_mode = debug.get("window_mode")
    valid_samples = int(debug.get("valid_samples") or 0)
    return window_mode == "release_grace_rescue" and valid_samples <= MIN_SAMPLES


def _review_weak_primary_window(
    primary: Dict[str, Any],
    pose_frames: List[Dict[str, Any]],
    hand: str,
    events: Dict[str, Any],
) -> Dict[str, Any]:
    rescue = _apply_low_visibility_rescue(pose_frames, hand, events)
    reviewed = dict(primary)
    reviewed_debug = dict(reviewed.get("debug") or {})
    reviewed_debug["secondary_rescue"] = rescue
    reviewed["debug"] = reviewed_debug

    extension = reviewed.get("extension_deg")
    if extension is None:
        return rescue

    # Strongly legal measured windows stay legal even if the rescue is cautious.
    if reviewed.get("verdict") == "LEGAL" and extension <= (THRESH_LEGAL - WEAK_WINDOW_MARGIN_DEG):
        reviewed["confidence"] = min(float(reviewed.get("confidence") or 0.0), 0.70)
        reviewed["reason"] = "weak_window_but_clear_margin"
        return reviewed

    # Weak near-threshold calls must be corroborated.
    if reviewed.get("verdict") == "LEGAL":
        if rescue.get("verdict") == "LEGAL":
            reviewed["confidence"] = min(float(reviewed.get("confidence") or 0.0), 0.65)
            reviewed["reason"] = "weak_window_confirmed"
            return reviewed
        return {
            "verdict": "SUSPECT",
            "confidence": 0.45,
            "reason": "weak_window_near_threshold",
            "extension_deg": extension,
            "debug": reviewed_debug,
        }

    if reviewed.get("verdict") in {"BORDERLINE", "ILLEGAL"} and rescue.get("verdict") == "LEGAL":
        return {
            "verdict": "SUSPECT",
            "confidence": 0.50,
            "reason": "weak_window_conflicted",
            "extension_deg": extension,
            "debug": reviewed_debug,
        }

    reviewed["confidence"] = min(float(reviewed.get("confidence") or 0.0), 0.60)
    reviewed["reason"] = "weak_window_unconfirmed"
    return reviewed


# -------------------------------------------------
# Core
# -------------------------------------------------

def _compute_elbow_legality(
    elbow_signal: List[Dict[str, Any]],
    events: Dict[str, Any],
) -> Dict[str, Any]:

    uah = _extract_event_frame(events, "uah")
    release = _extract_event_frame(events, "release")

    if uah is None or release is None:
        return _flow_based_legality(
            elbow_signal,
            events,
            primary_reason="events_missing",
            primary_debug={"uah": uah, "release": release},
        )

    release_used = release - RELEASE_TRIM_FRAMES
    if release_used < 0:
        return _flow_based_legality(
            elbow_signal,
            events,
            primary_reason="event_window_too_short",
            primary_debug={"uah": uah, "release": release, "release_used": release_used},
        )

    start_frame, end_frame, rows, window_debug = _select_measurement_rows(
        elbow_signal,
        uah,
        release_used,
    )

    # ---------------------------------------------
    # SIGNAL-SPARSE CASE (KEY FIX)
    # ---------------------------------------------
    if len(rows) < MIN_SAMPLES:
        return _flow_based_legality(
            elbow_signal,
            events,
            primary_reason="insufficient_signal_density",
            primary_debug={
                "valid_samples": len(rows),
                "uah": uah,
                "release": release,
                "release_used": release_used,
                **window_debug,
                "note": "Event window sufficient; elbow landmarks sparse",
            },
        )

    rows = _mad_filter(rows)

    velocities = [abs(rows[i] - rows[i-1]) for i in range(1, len(rows))]
    baseline_candidates = []

    for i, v in enumerate(velocities):
        if v <= BASELINE_VEL_MAX:
            baseline_candidates.append(rows[i])
        if len(baseline_candidates) >= BASELINE_MAX_SAMPLES:
            break

    if len(baseline_candidates) < 2:
        baseline_candidates = rows[:BASELINE_MAX_SAMPLES]

    baseline = sorted(baseline_candidates)[len(baseline_candidates) // 2]
    peak = _percentile(rows, PEAK_Q)
    extension = peak - baseline

    if extension > THRESH_BORDERLINE:
        verdict = "ILLEGAL"
        confidence = 0.75
    elif extension > THRESH_LEGAL:
        verdict = "BORDERLINE"
        confidence = 0.60
    else:
        verdict = "LEGAL"
        confidence = 0.80

    return {
        "verdict": verdict,
        "confidence": round(confidence, 2),
        "extension_deg": round(extension, 2),
        "baseline_angle_deg": round(baseline, 2),
        "release_angle_deg": round(peak, 2),
        "window": {"start_frame": start_frame, "end_frame": end_frame},
        "debug": window_debug,
    }


# -------------------------------------------------
# Public API
# -------------------------------------------------

def evaluate_elbow_legality(
    elbow_signal: List[Dict[str, Any]],
    events: Dict[str, Any] = None,
    pose_frames: Optional[List[Dict[str, Any]]] = None,
    hand: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    if not isinstance(events, dict):
        return {
            "verdict": "UNKNOWN",
            "confidence": 0.20,
            "reason": "events_missing",
        }
    primary = _compute_elbow_legality(elbow_signal, events)
    if primary.get("verdict") == "UNKNOWN":
        if not pose_frames or not hand:
            return primary
        return _apply_low_visibility_rescue(pose_frames, hand, events)

    if not pose_frames or not hand:
        return primary

    if _is_weak_primary_window(primary):
        return _review_weak_primary_window(primary, pose_frames, hand, events)

    return primary
