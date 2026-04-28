from __future__ import annotations
from .shared import *
from .joints import *
from .tracks import *
from .bubble_base import _draw_top_risk_panel

def _front_leg_support_caption(risk: Optional[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    if not isinstance(risk, dict):
        return None
    signal = float(risk.get("signal_strength") or 0.0)
    if signal >= 0.65:
        return {
            "title": "Front leg doesn't hold strong here.",
            "body": "The front leg is not giving a strong base at landing.",
        }
    if signal >= 0.35:
        return {
            "title": "Front leg needs a stronger base here.",
            "body": "The front leg softens a bit at landing.",
        }
    return {
        "title": "Front leg holds strong here.",
        "body": "The front leg gives a steady base at landing.",
    }
def _draw_front_leg_support_callout(
    frame: np.ndarray,
    *,
    tracks: Dict[int, Dict[str, Any]],
    frame_idx: int,
    hand: Optional[str],
    risk: Optional[Dict[str, Any]],
    proof_step: Optional[Dict[str, Any]] = None,
) -> None:
    caption = _front_leg_support_caption(risk)
    if not caption and not proof_step:
        return

    hip_idx, knee_idx, ankle_idx = _front_leg_joints(hand)
    hip = _track_point(tracks, hip_idx, frame_idx)
    knee = _track_point(tracks, knee_idx, frame_idx)
    ankle = _track_point(tracks, ankle_idx, frame_idx)
    if knee is None:
        return

    scale = min(frame.shape[0], frame.shape[1])
    signal = float((risk or {}).get("signal_strength") or 0.0)
    accent = (90, 220, 255) if signal < 0.35 else ((0, 196, 255) if signal < 0.65 else (0, 126, 255))
    radius = max(16, scale // 14)
    thickness = max(3, scale // 160)

    cv2.circle(frame, knee, radius + 4, SKELETON_SHADOW, thickness + 2, cv2.LINE_AA)
    cv2.circle(frame, knee, radius, accent, thickness, cv2.LINE_AA)

    if hip is not None and ankle is not None:
        direction = (ankle[0] - hip[0], ankle[1] - hip[1])
        arrow_start = (int(round(knee[0] + direction[0] * 0.10)), int(round(knee[1] - radius * 1.50)))
    else:
        arrow_start = (knee[0], knee[1] - int(radius * 1.75))
    cv2.line(
        frame,
        arrow_start,
        knee,
        accent,
        thickness,
        cv2.LINE_AA,
    )
    _draw_top_risk_panel(
        frame,
        title=str((proof_step or {}).get("title") or RISK_TITLE_BY_ID["knee_brace_failure"]),
        headline=str((proof_step or {}).get("headline") or (caption or {}).get("title") or ""),
        body=str((proof_step or {}).get("body") or (caption or {}).get("body") or ""),
        accent=accent,
        anchor=knee,
    )
def _draw_foot_line_overlay(
    frame: np.ndarray,
    *,
    pose_frames: List[Dict[str, Any]],
    frame_idx: int,
    events: Optional[Dict[str, Any]],
    hand: Optional[str],
    risk: Optional[Dict[str, Any]],
    proof_step: Optional[Dict[str, Any]] = None,
) -> None:
    if not isinstance(risk, dict):
        return
    width = frame.shape[1]
    height = frame.shape[0]
    front_toe_idx, front_heel_idx, back_toe_idx = _foot_indices(hand)
    bfc_frame = _safe_int(((events or {}).get("bfc") or {}).get("frame"))
    back_toe = _frame_point(pose_frames, frame_idx=bfc_frame if bfc_frame is not None else frame_idx, joint_idx=back_toe_idx, width=width, height=height)
    front_toe = _frame_point(pose_frames, frame_idx=frame_idx, joint_idx=front_toe_idx, width=width, height=height)
    front_heel = _frame_point(pose_frames, frame_idx=frame_idx, joint_idx=front_heel_idx, width=width, height=height)
    if not (back_toe and front_toe and front_heel):
        return

    signal = float(risk.get("signal_strength") or 0.0)
    accent = (120, 210, 255) if signal < 0.35 else ((0, 196, 255) if signal < 0.65 else (0, 126, 255))
    muted = (190, 202, 214)
    thickness = max(2, min(width, height) // 190)
    dx = front_toe[0] - back_toe[0]
    dy = front_toe[1] - back_toe[1]
    line_end = (back_toe[0] + int(round(dx * 1.10)), back_toe[1] + int(round(dy * 1.10)))

    cv2.line(frame, back_toe, line_end, muted, thickness, cv2.LINE_AA)
    cv2.line(frame, front_heel, front_toe, accent, thickness + 1, cv2.LINE_AA)
    cv2.circle(frame, front_toe, max(8, min(width, height) // 40), accent, thickness + 1, cv2.LINE_AA)
    _draw_top_risk_panel(
        frame,
        title=str((proof_step or {}).get("title") or "Where It Starts"),
        headline=str((proof_step or {}).get("headline") or "Front foot lands across here."),
        body=str((proof_step or {}).get("body") or "The landing foot is not lining force up cleanly toward target."),
        accent=accent,
        anchor=front_toe,
    )
