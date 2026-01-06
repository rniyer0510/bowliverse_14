from typing import Any, Dict, List, Optional, Tuple
import math

LS, LE, LW = 11, 13, 15
RS, RE, RW = 12, 14, 16

MIN_VIS = 0.5
SMOOTH_WIN = 3
UAH_LOOKBACK_SEC = 0.9
UAH_MIN_SEP_SEC = 0.15

def _xyz(pt):
    try:
        return (float(pt["x"]), float(pt["y"]), float(pt.get("z", 0.0)))
    except Exception:
        return None

def _dist(a, b):
    return math.sqrt(sum((a[i] - b[i])**2 for i in range(3)))

def _smooth(vals):
    out = []
    last = 0.0
    for v in vals:
        if v is None:
            last *= 0.9
            out.append(last)
        else:
            last = v
            out.append(v)
    return out

def _upper_arm_tilt(s, e):
    vx, vy, vz = e[0]-s[0], e[1]-s[1], e[2]-s[2]
    mag = math.sqrt(vx*vx + vy*vy + vz*vz)
    if mag < 1e-6:
        return None
    cosv = abs(vy) / mag
    cosv = max(-1.0, min(1.0, cosv))
    return math.degrees(math.acos(cosv))

def detect_release_uah(pose_frames: List[Dict[str, Any]], hand: str, fps: float):
    s_idx, e_idx, w_idx = (RS, RE, RW) if hand == "R" else (LS, LE, LW)

    frames, wrist_xyz, pelvis_xyz, shoulder_xyz, elbow_xyz = [], [], [], [], []

    for it in pose_frames:
        f = it["frame"]
        lm = it.get("landmarks")
        frames.append(f)

        if not lm:
            wrist_xyz.append(None)
            pelvis_xyz.append(None)
            shoulder_xyz.append(None)
            elbow_xyz.append(None)
            continue

        w = _xyz(lm[w_idx]) if lm[w_idx]["visibility"] >= MIN_VIS else None
        s = _xyz(lm[s_idx]) if lm[s_idx]["visibility"] >= MIN_VIS else None
        e = _xyz(lm[e_idx]) if lm[e_idx]["visibility"] >= MIN_VIS else None

        hip_l = _xyz(lm[23]) if lm[23]["visibility"] >= MIN_VIS else None
        hip_r = _xyz(lm[24]) if lm[24]["visibility"] >= MIN_VIS else None
        pelvis = None
        if hip_l and hip_r:
            pelvis = ((hip_l[0]+hip_r[0])/2, (hip_l[1]+hip_r[1])/2, (hip_l[2]+hip_r[2])/2)

        wrist_xyz.append(w)
        pelvis_xyz.append(pelvis)
        shoulder_xyz.append(s)
        elbow_xyz.append(e)

    # Forward axis from pelvis velocity
    vels = []
    for i in range(1, len(pelvis_xyz)):
        if pelvis_xyz[i] and pelvis_xyz[i-1]:
            vels.append((
                pelvis_xyz[i][0]-pelvis_xyz[i-1][0],
                pelvis_xyz[i][1]-pelvis_xyz[i-1][1],
                pelvis_xyz[i][2]-pelvis_xyz[i-1][2],
            ))
    if not vels:
        forward = (1,0,0)
    else:
        fx = sum(v[0] for v in vels)/len(vels)
        fy = sum(v[1] for v in vels)/len(vels)
        fz = sum(v[2] for v in vels)/len(vels)
        mag = math.sqrt(fx*fx+fy*fy+fz*fz) or 1.0
        forward = (fx/mag, fy/mag, fz/mag)

    proj = [None]
    for i in range(1, len(wrist_xyz)):
        if wrist_xyz[i] and wrist_xyz[i-1]:
            vx = wrist_xyz[i][0]-wrist_xyz[i-1][0]
            vy = wrist_xyz[i][1]-wrist_xyz[i-1][1]
            vz = wrist_xyz[i][2]-wrist_xyz[i-1][2]
            proj.append(vx*forward[0] + vy*forward[1] + vz*forward[2])
        else:
            proj.append(None)

    proj_s = _smooth(proj)
    start = int(len(proj_s)*0.45)
    release_i = max(range(start, len(proj_s)), key=lambda i: proj_s[i])

    fps_eff = fps or 25.0
    min_sep = max(2, int(UAH_MIN_SEP_SEC*fps_eff))
    lookback = max(8, int(UAH_LOOKBACK_SEC*fps_eff))

    uah_i = None
    best = None
    for i in range(max(0, release_i-min_sep), max(0, release_i-lookback), -1):
        if shoulder_xyz[i] and elbow_xyz[i]:
            tilt = _upper_arm_tilt(shoulder_xyz[i], elbow_xyz[i])
            if tilt is not None and (best is None or tilt < best):
                best = tilt
                uah_i = i

    if uah_i is None:
        uah_i = max(0, release_i-min_sep)

    return {
        "uah": {"frame": frames[uah_i]},
        "release": {"frame": frames[release_i]},
    }
