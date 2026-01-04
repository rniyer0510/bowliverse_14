import numpy as np

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
VIS_THR = 0.5


def compute_trunk_rotation_snap(pose_frames, ffc_frame, uah_frame, fps, config):
    if None in (ffc_frame, uah_frame) or fps <= 0:
        return None

    start = int(ffc_frame)
    end = int(uah_frame)

    angles = []
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

        sh = np.array([
            lms[RIGHT_SHOULDER]["x"] - lms[LEFT_SHOULDER]["x"],
            lms[RIGHT_SHOULDER]["y"] - lms[LEFT_SHOULDER]["y"],
        ])
        hp = np.array([
            lms[RIGHT_HIP]["x"] - lms[LEFT_HIP]["x"],
            lms[RIGHT_HIP]["y"] - lms[LEFT_HIP]["y"],
        ])

        ang = np.degrees(np.arctan2(sh[1], sh[0]) - np.arctan2(hp[1], hp[0]))
        angles.append(ang)
        used += 1

    if len(angles) < 4:
        return None

    angles = np.unwrap(np.radians(angles))
    vel = np.gradient(angles) * fps
    acc = np.gradient(vel) * fps

    peak = float(np.max(np.abs(acc)))
    signal = min(peak / config["snap_ref"], 1.0)
    conf = min(used / (end - start + 1), 1.0)

    return {
        "risk_id": "trunk_rotation_snap",
        "signal_strength": round(signal, 3),
        "confidence": round(conf, 3),
        "debug": {"peak_rot_acc": round(peak, 3)},
    }
