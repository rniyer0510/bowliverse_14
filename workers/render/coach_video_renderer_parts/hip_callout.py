from __future__ import annotations
from .shared import *
from .tracks import *
from .bubble_base import _draw_top_risk_panel
from .geometry_helpers import _midpoint

def _hip_shoulder_caption(risk: Optional[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    if not isinstance(risk, dict):
        return None
    debug = risk.get("debug") or {}
    sequence_pattern = str(debug.get("sequence_pattern") or "").lower()
    signal = float(risk.get("signal_strength") or 0.0)
    if sequence_pattern == "shoulders_lead":
        if signal >= 0.45:
            return {
                "title": "Shoulders turn too early.",
                "body": "The shoulders get ahead of the hips too soon.",
            }
        return {
            "title": "Shoulders start a bit early.",
            "body": "The top half gets ahead a little too soon.",
        }
    if sequence_pattern == "hips_lead":
        if signal >= 0.45:
            return {
                "title": "Hips turn too early.",
                "body": "The hips get ahead of the shoulders too soon.",
            }
        return {
            "title": "Hips lead a bit early.",
            "body": "The hips start getting ahead of the shoulders.",
        }
    if signal >= 0.45:
        return {
            "title": "Hips and shoulders are not together here.",
            "body": "The middle of the body is not moving together.",
        }
    return None
def _draw_hip_shoulder_callout(
    frame: np.ndarray,
    *,
    tracks: Dict[int, Dict[str, Any]],
    frame_idx: int,
    risk: Optional[Dict[str, Any]],
    proof_step: Optional[Dict[str, Any]] = None,
) -> None:
    caption = _hip_shoulder_caption(risk)
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
    signal = float((risk or {}).get("signal_strength") or 0.0)
    accent = (90, 220, 255) if signal < 0.45 else (0, 196, 255)
    scale = min(frame.shape[0], frame.shape[1])
    thickness = max(3, scale // 170)

    cv2.line(frame, left_shoulder, right_shoulder, accent, thickness, cv2.LINE_AA)
    cv2.line(frame, left_hip, right_hip, accent, thickness, cv2.LINE_AA)
    cv2.line(frame, hip_mid, shoulder_mid, accent, max(2, thickness - 1), cv2.LINE_AA)

    debug = (risk or {}).get("debug") or {}
    sequence_pattern = str(debug.get("sequence_pattern") or "").lower()
    arrow_dx = int(round(scale * 0.05))
    if sequence_pattern == "shoulders_lead":
        cv2.arrowedLine(
            frame,
            (shoulder_mid[0] - arrow_dx, shoulder_mid[1] - arrow_dx),
            (shoulder_mid[0] + arrow_dx, shoulder_mid[1] - arrow_dx),
            accent,
            thickness,
            cv2.LINE_AA,
            tipLength=0.18,
        )
    elif sequence_pattern == "hips_lead":
        cv2.arrowedLine(
            frame,
            (hip_mid[0] - arrow_dx, hip_mid[1] + arrow_dx),
            (hip_mid[0] + arrow_dx, hip_mid[1] + arrow_dx),
            accent,
            thickness,
            cv2.LINE_AA,
            tipLength=0.18,
        )

    _draw_top_risk_panel(
        frame,
        title=str((proof_step or {}).get("title") or RISK_TITLE_BY_ID["hip_shoulder_mismatch"]),
        headline=str((proof_step or {}).get("headline") or (caption or {}).get("title") or ""),
        body=str((proof_step or {}).get("body") or (caption or {}).get("body") or ""),
        accent=accent,
        anchor=_midpoint(hip_mid, shoulder_mid) or shoulder_mid,
    )
