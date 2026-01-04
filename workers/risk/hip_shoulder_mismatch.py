import numpy as np

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
VIS_THR = 0.5


def compute_hip_shoulder_mismatch(pose_frames, ffc_frame, release_frame, fps, config):
    if None in (ffc_frame, release_frame) or fps <= 0:
        return None

    start = int(ffc_frame)
    end = int(release_frame)

    hip_vel = []
    sh_vel = []
    used = 0

    prev_hip = None
    prev_sh = None

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

        if prev_hip is not None:
            hip_vel.append(np.linalg.norm(hp - prev_hip) * fps)
            sh_vel.append(np.linalg.norm(sh - prev_sh) * fps)
            used += 1

        prev_hip = hp
        prev_sh = sh

    if len(hip_vel) < 3:
        return None

    hip_vel = np.array(hip_vel)
    sh_vel = np.array(sh_vel)

    mismatch = np.mean(np.abs(sh_vel - hip_vel))
    signal = min(mismatch / config["mismatch_ref"], 1.0)
    conf = min(used / (end - start), 1.0)

    return {
        "risk_id": "hip_shoulder_mismatch",
        "signal_strength": round(signal, 3),
        "confidence": round(conf, 3),
        "debug": {"mean_mismatch": round(float(mismatch), 3)},
    }
