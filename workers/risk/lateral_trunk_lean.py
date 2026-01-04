import numpy as np

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
VIS_THR = 0.5


def compute_lateral_trunk_lean(pose_frames, ffc_frame, release_frame, fps, config):
    if None in (ffc_frame, release_frame) or fps <= 0:
        return None

    start = int(ffc_frame)
    end = int(release_frame)

    leans = []
    used = 0

    for i in range(start, end + 1):
        lms = pose_frames[i].get("landmarks")
        if not lms:
            continue

        if min(
            lms[LEFT_SHOULDER]["visibility"],
            lms[RIGHT_SHOULDER]["visibility"],
            lms[LEFT_HIP]["visibility"],
            lms[RIGHT_HIP]["visibility"],
        ) < VIS_THR:
            continue

        mid_sh = (lms[LEFT_SHOULDER]["x"] + lms[RIGHT_SHOULDER]["x"]) / 2
        mid_hp = (lms[LEFT_HIP]["x"] + lms[RIGHT_HIP]["x"]) / 2

        lean = abs(mid_sh - mid_hp)
        leans.append(lean)
        used += 1

    if len(leans) < 3:
        return None

    leans = np.array(leans)
    peak = float(np.max(leans))
    signal = min(peak / config["lean_ref"], 1.0)
    conf = min(used / (end - start + 1), 1.0)

    return {
        "risk_id": "lateral_trunk_lean",
        "signal_strength": round(signal, 3),
        "confidence": round(conf, 3),
        "debug": {"peak_lateral_offset": round(peak, 3)},
    }
