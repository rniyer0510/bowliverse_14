import numpy as np

LEFT_HIP = 23
RIGHT_HIP = 24
VIS_THR = 0.50


def _pelvis_mid(lms):
    lh = lms[LEFT_HIP]
    rh = lms[RIGHT_HIP]
    if lh.get("visibility", 0.0) < VIS_THR or rh.get("visibility", 0.0) < VIS_THR:
        return None
    x = (lh["x"] + rh["x"]) / 2.0
    y = (lh["y"] + rh["y"]) / 2.0
    z = (lh.get("z", 0.0) + rh.get("z", 0.0)) / 2.0
    return x, y, z


def _axis_weights(action: dict):
    """
    Choose a deterministic forward-axis blend.
    - side_on: mostly x
    - front_on: mostly z
    - mixed/unknown: blended (robust default)
    If action confidence is low, bias toward blended.
    """
    action = action or {}
    intent = str(action.get("intent") or "").lower()
    conf = action.get("confidence", None)

    # robust default
    wx, wz = 0.25, 0.75

    if "side" in intent:
        wx, wz = 0.80, 0.20
    elif "front" in intent:
        wx, wz = 0.10, 0.90
    elif "mixed" in intent:
        wx, wz = 0.35, 0.65

    # Low confidence → blend more (avoid over-committing)
    try:
        if conf is not None and float(conf) < 0.55:
            wx, wz = 0.30, 0.70
            intent = intent + "_lowconf"
    except Exception:
        pass

    return wx, wz, intent


def compute_front_foot_braking_shock(pose_frames, ffc_frame, fps, config, action=None):
    """
    Front-foot braking shock (FFBS) — cricket-native, baseball-style.

    Fixes:
    - intent-aware forward axis (x vs z)
    - LOW-MOMENTUM GATE: if there is negligible forward travel in the window,
      braking shock is not applicable (cap severity, add note).
    """
    if ffc_frame is None or ffc_frame <= 0 or fps <= 0:
        return None

    pre = int(config.get("pre_window", 6))
    post = int(config.get("post_window", 4))
    jerk_ref = float(config.get("jerk_reference", 15.0))

    # forward travel gate (configurable; normalized units)
    travel_min = float(config.get("travel_min", 0.012))  # must be configurable later

    start = max(0, int(ffc_frame) - pre)
    end = min(len(pose_frames) - 1, int(ffc_frame) + post)

    if end - start < 3:
        return None

    wx, wz, axis_intent = _axis_weights(action or {})

    series = []
    used_frames = 0

    for i in range(start, end + 1):
        item = pose_frames[i]
        lms = item.get("landmarks")
        if not lms:
            continue
        mid = _pelvis_mid(lms)
        if mid is None:
            continue
        x, y, z = mid

        forward = (wx * x) + (wz * z)
        series.append(float(forward))
        used_frames += 1

    if len(series) < 4:
        return None

    s = np.array(series, dtype=float)

    # Robust "travel" estimate in window (ignore extremes)
    p5 = float(np.percentile(s, 5))
    p95 = float(np.percentile(s, 95))
    travel = abs(p95 - p5)

    vel = np.gradient(s) * fps
    acc = np.gradient(vel) * fps
    jerk = np.gradient(acc) * fps

    peak_jerk = float(np.max(np.abs(jerk)))
    raw_strength = peak_jerk / max(1e-6, jerk_ref)
    signal_strength = float(min(raw_strength, 1.0))

    n = len(vel)
    mid_idx = max(1, n // 2)
    v_pre = float(np.mean(vel[:mid_idx])) if mid_idx > 0 else 0.0
    v_post = float(np.mean(vel[mid_idx:])) if (n - mid_idx) > 0 else 0.0

    note = None

    # ----------------------------------------------------------
    # Momentum/travel gate:
    # If there is negligible translation, braking shock is not a
    # meaningful label. Cap severity to "monitor".
    # ----------------------------------------------------------
    if travel < travel_min:
        signal_strength = min(signal_strength, 0.25)
        note = "Low forward travel in FFC window; braking shock not applicable (monitor only)."

    # Continuity discriminator (only meaningful when travel exists)
    if travel >= travel_min and peak_jerk > jerk_ref and abs(v_pre) > 1e-6:
        if abs(v_post) >= 0.9 * abs(v_pre):
            signal_strength = min(signal_strength, 0.35)

    confidence = min(used_frames / max(1, (pre + post + 1)), 1.0)

    out = {
        "risk_id": "front_foot_braking_shock",
        "signal_strength": round(float(signal_strength), 3),
        "confidence": round(float(confidence), 3),
        "window": {"start_frame": start, "end_frame": end},
        "debug": {
            "axis_intent": axis_intent,
            "axis_weights": {"wx": round(wx, 3), "wz": round(wz, 3)},
            "peak_jerk": round(float(peak_jerk), 3),
            "jerk_ref": float(jerk_ref),
            "samples": int(used_frames),
            "travel_p95_p5": round(float(travel), 4),
            "travel_min": float(travel_min),
            "forward_velocity": {
                "pre_mean": round(v_pre, 3),
                "post_mean": round(v_post, 3),
            },
        },
    }
    if note:
        out["note"] = note
    return out
