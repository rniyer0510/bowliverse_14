"""
Basic Coaching Diagnostics — ActionLab V14
-----------------------------------------
This is NOT injury-risk. This is "what a club bowler can fix in nets".

Goals:
- Provide actionable feedback even when risk signals are low/occluded.
- Be robust (deterministic, visibility-aware, windowed).
- Do NOT change risk/flow logic. Output as separate 'basics' block.

Diagnostics:
1) Knee bracing proxy  : pelvis drop after FFC (COM sink = likely no brace)
2) Back-foot stability : back ankle jitter near UAH (unstable base = upper-body bowling)
3) Front-foot toe line : front foot angle vs batsman axis (toe drifting outside target line)
"""

from typing import Any, Dict, List, Optional, Tuple
import math
import statistics

from app.common.signal_quality import landmarks_visible
from app.workers.action.geometry import compute_batsman_axis
from app.workers.action.foot_orientation import compute_foot_intent

# MediaPipe indices
LH, RH = 23, 24
L_ANKLE, R_ANKLE = 27, 28

# -----------------------------
# Config (V14)
# NOTE: Move to config layer before commercial release.
# -----------------------------
CFG = {
    # Knee brace proxy: if pelvis drops too much after FFC -> likely collapse / no brace
    # units: normalized y in image space (MediaPipe)
    "knee": {
        "post_window": 8,          # frames after FFC to examine
        "drop_warn": 0.020,        # mild drop
        "drop_bad": 0.040,         # strong drop
        "min_visible": 5,          # minimum frames with visible hips
    },
    # Back-foot stability near UAH: ankle jitter
    "back_foot": {
        "pre_window": 8,           # frames before UAH (inclusive window end at UAH)
        "jitter_warn": 0.006,      # mild jitter (normalized)
        "jitter_bad": 0.012,       # high jitter
        "min_visible": 5,
    },
    # Front-foot toe line: uses compute_foot_intent angle bands
    "toe": {
        "semi_open_min": 35.0,
        "semi_open_max": 65.0,
        "front_on_min": 65.0,
    }
}


def _pelvis_y(frame: Dict[str, Any]) -> Optional[float]:
    if not frame or "landmarks" not in frame:
        return None
    if not landmarks_visible(frame, [LH, RH]):
        return None
    lm = frame["landmarks"]
    return 0.5 * (float(lm[LH]["y"]) + float(lm[RH]["y"]))


def _stdev(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0.0
    try:
        return float(statistics.pstdev(vals))
    except Exception:
        return 0.0


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _grade_from_thresholds(value: float, warn: float, bad: float) -> Tuple[str, float]:
    """
    Returns (grade, severity_strength) where:
      ok   -> 0.15
      warn -> ~0.45
      bad  -> ~0.75
    """
    if value >= bad:
        return "bad", 0.75
    if value >= warn:
        return "warn", 0.45
    return "ok", 0.15


def analyze_basics(
    pose_frames: List[Dict[str, Any]],
    hand: str,
    events: Dict[str, Any],
    action: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Returns a non-risk coaching block. Safe to show to kids.
    """
    hand = (hand or "").upper().strip()
    uah = (events.get("uah") or {}).get("frame")
    ffc = (events.get("ffc") or {}).get("frame")
    bfc = (events.get("bfc") or {}).get("frame")

    out: Dict[str, Any] = {
        "knee_brace_proxy": {"status": "unknown", "confidence": 0.0},
        "back_foot_stability": {"status": "unknown", "confidence": 0.0},
        "front_foot_toe_alignment": {"status": "unknown", "confidence": 0.0},
        "coach_cues": [],
        "user_cues": [],
    }

    # -------------------------------------------------------------------------
    # 1) Knee brace proxy: pelvis drop after FFC
    # -------------------------------------------------------------------------
    if ffc is not None and 0 <= ffc < len(pose_frames):
        post = int(CFG["knee"]["post_window"])
        start = int(ffc)
        end = min(len(pose_frames) - 1, int(ffc) + post)

        ys: List[float] = []
        used = 0

        for i in range(start, end + 1):
            y = _pelvis_y(pose_frames[i])
            if y is None:
                continue
            ys.append(float(y))
            used += 1

        if used >= int(CFG["knee"]["min_visible"]) and ys:
            y0 = ys[0]
            y_peak = max(ys)  # larger y means pelvis "dropped" (down the image)
            drop = float(y_peak - y0)

            grade, strength = _grade_from_thresholds(
                drop,
                float(CFG["knee"]["drop_warn"]),
                float(CFG["knee"]["drop_bad"]),
            )

            conf = _clip01(used / max(1, (end - start + 1)))

            out["knee_brace_proxy"] = {
                "status": grade,
                "confidence": round(conf, 2),
                "debug": {"pelvis_drop": round(drop, 4), "frames_used": used},
            }

            if grade in ("warn", "bad"):
                out["coach_cues"].append("Front knee brace: reduce pelvis sink after front-foot contact (stronger block).")
                out["user_cues"].append("At front-foot landing, try to stay tall and stop your front knee from collapsing.")

    # -------------------------------------------------------------------------
    # 2) Back-foot stability near UAH: ankle jitter
    # -------------------------------------------------------------------------
    if uah is not None and 0 <= uah < len(pose_frames):
        pre = int(CFG["back_foot"]["pre_window"])
        start = max(0, int(uah) - pre)
        end = int(uah)

        xs: List[float] = []
        zs: List[float] = []
        used = 0

        back_ankle_idx = R_ANKLE if hand == "R" else L_ANKLE

        for i in range(start, end + 1):
            fr = pose_frames[i]
            if not fr or "landmarks" not in fr:
                continue
            if not landmarks_visible(fr, [back_ankle_idx]):
                continue
            lm = fr["landmarks"][back_ankle_idx]
            xs.append(float(lm["x"]))
            zs.append(float(lm.get("z", 0.0)))
            used += 1

        if used >= int(CFG["back_foot"]["min_visible"]) and xs:
            jitter = math.sqrt((_stdev(xs) ** 2) + (_stdev(zs) ** 2))
            grade, strength = _grade_from_thresholds(
                jitter,
                float(CFG["back_foot"]["jitter_warn"]),
                float(CFG["back_foot"]["jitter_bad"]),
            )
            conf = _clip01(used / max(1, (end - start + 1)))

            out["back_foot_stability"] = {
                "status": grade,
                "confidence": round(conf, 2),
                "debug": {"ankle_jitter": round(float(jitter), 5), "frames_used": used},
            }

            if grade in ("warn", "bad"):
                out["coach_cues"].append("Back-foot stability: get the back foot planted before upper-body rotation (better GRF transfer).")
                out["user_cues"].append("Try to land your back foot firmly before you spin and pull your arm through.")

    # -------------------------------------------------------------------------
    # 3) Front-foot toe alignment vs batsman axis @ BFC
    # -------------------------------------------------------------------------
    if bfc is not None and 0 <= bfc < len(pose_frames):
        axis = compute_batsman_axis(pose_frames, bfc, ffc)
        if axis is not None:
            foot = compute_foot_intent(pose_frames=pose_frames, hand=hand, bfc_frame=bfc, axis=axis)
            if foot and isinstance(foot.get("angle"), (int, float)):
                ang = float(foot["angle"])
                intent = str(foot.get("intent") or "UNKNOWN")

                # Interpret angle into coaching-friendly status
                if ang >= float(CFG["toe"]["front_on_min"]):
                    status = "open"
                elif float(CFG["toe"]["semi_open_min"]) <= ang <= float(CFG["toe"]["semi_open_max"]):
                    status = "semi_open"
                else:
                    status = "aligned"

                out["front_foot_toe_alignment"] = {
                    "status": status,
                    "confidence": 1.0,
                    "debug": {"toe_angle_deg": round(ang, 2), "band": intent},
                }

                if status in ("semi_open", "open"):
                    out["coach_cues"].append("Front-foot toe line: guide toe closer to the target line to reduce rotational leak.")
                    out["user_cues"].append("Point your front toe a bit more towards the batsman/target as you land.")

    # De-duplicate cues while preserving order
    def _dedupe(items: List[str]) -> List[str]:
        seen = set()
        outl = []
        for it in items:
            if it in seen:
                continue
            seen.add(it)
            outl.append(it)
        return outl

    out["coach_cues"] = _dedupe(out["coach_cues"])
    out["user_cues"] = _dedupe(out["user_cues"])

    return out
