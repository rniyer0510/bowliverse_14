import numpy as np

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24

VIS_THR = 0.5


def _lean_at_frame(pose_frames, idx):
    lms = pose_frames[idx].get("landmarks")
    if not lms:
        return None
    if min(
        lms[LEFT_SHOULDER]["visibility"],
        lms[RIGHT_SHOULDER]["visibility"],
        lms[LEFT_HIP]["visibility"],
        lms[RIGHT_HIP]["visibility"],
    ) < VIS_THR:
        return None

    mid_sh = (lms[LEFT_SHOULDER]["x"] + lms[RIGHT_SHOULDER]["x"]) / 2
    mid_hp = (lms[LEFT_HIP]["x"] + lms[RIGHT_HIP]["x"]) / 2
    return abs(mid_sh - mid_hp)


def compute_lateral_trunk_lean(
    pose_frames,
    bfc_frame,
    ffc_frame,   # retained for signature compatibility (not used)
    release_frame,
    fps,
    config,
):
    """
    Lateral trunk lean risk (V14 LOCKED)

    PRINCIPLE:
    - BFC is the structural baseline.
    - Risk is driven by CHANGE in lean after BFC (Δlean), not absolute lean.
    - High lean at BFC with small Δlean is NOT short-term risk
      (may be noted for long-term load monitoring only).
    """

    if None in (bfc_frame, release_frame) or fps <= 0:
        return None

    start = int(bfc_frame)
    end = int(release_frame)

    if end <= start:
        return None

    # --- baseline lean at BFC ---
    lean_bfc = _lean_at_frame(pose_frames, start)
    if lean_bfc is None:
        return None

    # --- scan lean from BFC -> Release ---
    leans = []
    used = 0

    for i in range(start, end + 1):
        lean = _lean_at_frame(pose_frames, i)
        if lean is None:
            continue
        leans.append(float(lean))
        used += 1

    if len(leans) < 3:
        return None

    leans = np.array(leans, dtype=float)

    lean_peak = float(np.max(leans))
    delta_lean = max(0.0, lean_peak - lean_bfc)

    # --- score driven ONLY by delta ---
    lean_ref = float(config["lean_ref"])
    signal = min(delta_lean / max(1e-6, lean_ref), 1.0)

    # Conservative floor handling
    signal = max(signal, float(config.get("floor", 0.15)))

    conf = min(used / max(1, (end - start + 1)), 1.0)

    out = {
        "risk_id": "lateral_trunk_lean",
        "signal_strength": round(float(signal), 3),
        "confidence": round(float(conf), 3),
        "debug": {
            "lean_bfc": round(float(lean_bfc), 3),
            "lean_peak": round(float(lean_peak), 3),
            "delta_lean": round(float(delta_lean), 3),
            "baseline_frame": int(start),
        },
    }

    # --- interpretation hint for long-term monitoring ---
    if delta_lean < 0.03 and lean_bfc >= lean_ref:
        out["note"] = (
            "Stable lateral posture established early. "
            "Not a short-term risk, but may warrant long-term load monitoring."
        )

    return out
