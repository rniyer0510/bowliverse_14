import os
from typing import Any, Dict, Optional, Tuple

OUT_DIR = "/tmp/actionlab_frames"

def _ensure_dir() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

def _safe_import_cv2():
    try:
        import cv2  # type: ignore
        return cv2
    except Exception:
        return None

def extract_frame(video_path: str, frame_idx: int):
    cv2 = _safe_import_cv2()
    if cv2 is None:
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()

def save_visual(
    risk_id: str,
    frame_idx: int,
    frame,
    region: str = "generic",
    direction: str = "load",
    severity: str = "low",
) -> Optional[str]:
    """
    Backward-compatible helper: saves the frame as PNG.
    (Older code imported save_visual; we keep it to prevent import crashes.)
    """
    cv2 = _safe_import_cv2()
    if cv2 is None:
        return None

    try:
        _ensure_dir()
        out_path = os.path.join(OUT_DIR, f"{risk_id}_{frame_idx}.png")
        ok = cv2.imwrite(out_path, frame)
        return out_path if ok else None
    except Exception:
        return None

def _lm_xy(lm: Dict[str, Any], key: str) -> Optional[Tuple[float, float, float]]:
    if not isinstance(lm, dict):
        return None
    p = lm.get(key)
    if not isinstance(p, dict):
        return None
    x = p.get("x")
    y = p.get("y")
    v = p.get("v", 0.0)
    if x is None or y is None:
        return None
    try:
        return float(x), float(y), float(v)
    except Exception:
        return None

def _pick_best_of(lm: Dict[str, Any], keys):
    best = None
    best_v = -1.0
    for k in keys:
        t = _lm_xy(lm, k)
        if t is None:
            continue
        _, _, v = t
        if v > best_v:
            best_v = v
            best = (k, t)
    return best

def _to_px(frame, x_norm: float, y_norm: float) -> Tuple[int, int]:
    h, w = frame.shape[:2]
    x = int(max(0, min(w - 1, x_norm * w)))
    y = int(max(0, min(h - 1, y_norm * h)))
    return x, y

def _anchor_point_for_risk(risk_id: str, lm: Dict[str, Any]):
    if risk_id == "knee_brace_failure":
        pick = _pick_best_of(lm, ["LEFT_KNEE", "RIGHT_KNEE"])
        if pick:
            _, (x, y, _) = pick
            return x, y, "knee"

    if risk_id == "front_foot_braking_shock":
        pick = _pick_best_of(lm, ["LEFT_ANKLE", "RIGHT_ANKLE"])
        if pick:
            _, (x, y, _) = pick
            return x, y, "ankle"

    if risk_id == "lateral_trunk_lean":
        pick = _pick_best_of(lm, ["LEFT_SHOULDER", "RIGHT_SHOULDER"])
        if pick:
            _, (x, y, _) = pick
            return x, y, "shoulder"

    if risk_id == "trunk_rotation_snap":
        ls = _lm_xy(lm, "LEFT_SHOULDER")
        rs = _lm_xy(lm, "RIGHT_SHOULDER")
        if ls and rs:
            x = (ls[0] + rs[0]) / 2.0
            y = (ls[1] + rs[1]) / 2.0
            return x, y, "torso"

    if risk_id == "hip_shoulder_mismatch":
        lh = _lm_xy(lm, "LEFT_HIP")
        rh = _lm_xy(lm, "RIGHT_HIP")
        if lh and rh:
            x = (lh[0] + rh[0]) / 2.0
            y = (lh[1] + rh[1]) / 2.0
            return x, y, "hip"

    return None

def draw_and_save_visual(
    *,
    risk: Dict[str, Any],
    video_path: str,
    frame_idx: int,
    pose_frames,
) -> Optional[Dict[str, Any]]:
    """
    Creates PNG with a visible arrow+dot and returns {"image_path", "frame", "anchor"}.
    Never raises.
    """
    try:
        if not video_path or frame_idx is None or frame_idx < 0:
            return None

        frame = extract_frame(video_path, frame_idx)
        if frame is None:
            return None

        lm = {}
        try:
            if pose_frames and 0 <= frame_idx < len(pose_frames):
                lm = (pose_frames[frame_idx] or {}).get("landmarks") or {}
        except Exception:
            lm = {}

        anchor = _anchor_point_for_risk(risk.get("risk_id", ""), lm)
        if anchor is None:
            h, w = frame.shape[:2]
            px, py = w // 2, h // 2
            label = "center"
        else:
            x_norm, y_norm, label = anchor
            px, py = _to_px(frame, x_norm, y_norm)

        cv2 = _safe_import_cv2()
        if cv2 is None:
            return None

        h, w = frame.shape[:2]
        sx = max(0, min(w - 1, px - int(0.12 * w)))
        sy = max(0, min(h - 1, py - int(0.12 * h)))

        cv2.arrowedLine(frame, (sx, sy), (px, py), (0, 165, 255), 3, tipLength=0.25)
        cv2.circle(frame, (px, py), 10, (0, 165, 255), -1)

        out_path = save_visual(
            risk_id=risk.get("risk_id", "risk"),
            frame_idx=int(frame_idx),
            frame=frame,
            region=risk.get("risk_region", "generic"),
            direction=risk.get("risk_direction", "load"),
            severity="low",
        )
        if not out_path:
            return None

        return {
            "image_path": out_path,
            "frame": int(frame_idx),
            "anchor": label,
        }
    except Exception:
        return None
