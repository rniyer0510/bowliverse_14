"""
Elbow legality evaluation — ActionLab V14 (FINAL, ICC-SAFE)

LOCKED BEHAVIOUR:
- Never return INCONCLUSIVE if elbow landmarks exist
- LEGAL with confidence downgrade when signal is sparse
- ILLEGAL / BORDERLINE only when confidence is sufficient
"""

from typing import Any, Dict, List, Optional
import math

THRESH_LEGAL = 15.0
THRESH_BORDERLINE = 20.0

RELEASE_TRIM_FRAMES = 2
MIN_SAMPLES = 5
BASELINE_MAX_SAMPLES = 4
PEAK_Q = 90

BASELINE_VEL_MAX = 3.0  # deg/frame


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
        return {
            "verdict": "LEGAL",
            "confidence": 0.25,
            "reason": "events_missing_assumed_legal"
        }

    release_used = release - RELEASE_TRIM_FRAMES
    if release_used <= uah:
        return {
            "verdict": "LEGAL",
            "confidence": 0.25,
            "reason": "event_window_too_short"
        }

    # Collect valid samples
    rows = []
    for it in elbow_signal:
        f = _safe_int(it.get("frame")) if isinstance(it, dict) else None
        if f is None or f < uah or f > release_used:
            continue
        if not it.get("valid", False):
            continue
        a = it.get("angle_deg")
        if not _finite(a):
            continue
        rows.append(float(a))

    # ---------------------------------------------
    # SIGNAL-SPARSE CASE (KEY FIX)
    # ---------------------------------------------
    if len(rows) < MIN_SAMPLES:
        return {
            "verdict": "LEGAL",
            "confidence": 0.30,
            "reason": "insufficient_signal_density",
            "debug": {
                "valid_samples": len(rows),
                "uah": uah,
                "release": release,
                "note": "Event window sufficient; elbow landmarks sparse"
            }
        }

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
        "window": {"start_frame": uah, "end_frame": release_used},
    }


# -------------------------------------------------
# Public API
# -------------------------------------------------

def evaluate_elbow_legality(
    elbow_signal: List[Dict[str, Any]],
    events: Dict[str, Any] = None,
    **kwargs,
) -> Dict[str, Any]:
    if not isinstance(events, dict):
        return {
            "verdict": "LEGAL",
            "confidence": 0.25,
            "reason": "events_missing_assumed_legal"
        }
    return _compute_elbow_legality(elbow_signal, events)

