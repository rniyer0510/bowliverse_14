from __future__ import annotations
from .shared import *
from .tracks import *
from .bubble_base import _draw_top_risk_panel

def _trunk_lean_caption(risk: Optional[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    if not isinstance(risk, dict):
        return None
    signal = float(risk.get("signal_strength") or 0.0)
    if signal >= 0.65:
        return {
            "title": "Body is falling away here.",
            "body": "The body moves too far away as the ball comes out.",
        }
    if signal >= 0.35:
        return {
            "title": "Body leans away here.",
            "body": "The body starts to drift away as the ball comes out.",
        }
    return {
        "title": "Body stays tall here.",
        "body": "The body stays more over the front leg as the ball comes out.",
    }
def _draw_trunk_lean_callout(
    frame: np.ndarray,
    *,
    tracks: Dict[int, Dict[str, Any]],
    frame_idx: int,
    hand: Optional[str] = None,
    risk: Optional[Dict[str, Any]],
    proof_step: Optional[Dict[str, Any]] = None,
) -> None:
    caption = _trunk_lean_caption(risk)
    if not caption and not proof_step:
        return
    left_shoulder = _track_point(tracks, LEFT_SHOULDER, frame_idx)
    right_shoulder = _track_point(tracks, RIGHT_SHOULDER, frame_idx)
    left_hip = _track_point(tracks, LEFT_HIP, frame_idx)
    right_hip = _track_point(tracks, RIGHT_HIP, frame_idx)
    if not (left_shoulder and right_shoulder and left_hip and right_hip):
        return

    shoulder_mid = ((left_shoulder[0] + right_shoulder[0]) // 2, (left_shoulder[1] + right_shoulder[1]) // 2)
    hip_mid = ((left_hip[0] + right_hip[0]) // 2, (left_hip[1] + right_hip[1]) // 2)
    dx = shoulder_mid[0] - hip_mid[0]
    signal = float((risk or {}).get("signal_strength") or 0.0)
    accent = (90, 220, 255) if signal < 0.35 else ((0, 196, 255) if signal < 0.65 else (0, 126, 255))
    scale = min(frame.shape[0], frame.shape[1])
    thickness = max(3, scale // 160)
    offset_x = int(round(scale * 0.04))

    cv2.line(frame, hip_mid, shoulder_mid, accent, thickness, cv2.LINE_AA)
    arrow_dir = 1 if dx >= 0 else -1
    arrow_start = (shoulder_mid[0], shoulder_mid[1] - int(scale * 0.02))
    arrow_end = (shoulder_mid[0] + arrow_dir * offset_x, shoulder_mid[1] - int(scale * 0.08))
    cv2.arrowedLine(frame, arrow_start, arrow_end, accent, thickness, cv2.LINE_AA, tipLength=0.18)
    cv2.circle(frame, shoulder_mid, max(12, scale // 18), accent, thickness, cv2.LINE_AA)

    _draw_top_risk_panel(
        frame,
        title=str((proof_step or {}).get("title") or RISK_TITLE_BY_ID["lateral_trunk_lean"]),
        headline=str((proof_step or {}).get("headline") or (caption or {}).get("title") or ""),
        body=str((proof_step or {}).get("body") or (caption or {}).get("body") or ""),
        accent=accent,
        anchor=shoulder_mid,
        bowler_hand=hand,
    )
