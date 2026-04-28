from __future__ import annotations
from .shared import *
from .tracks import *

def _draw_joint(frame: np.ndarray, point: Tuple[int, int], scale: int) -> None:
    outer = max(4, scale // 190)
    inner = max(2, outer - 1)
    cv2.circle(frame, point, outer + 1, SKELETON_SHADOW, -1, cv2.LINE_AA)
    cv2.circle(frame, point, outer, JOINT_OUTER, -1, cv2.LINE_AA)
    cv2.circle(frame, point, inner, SKELETON_COLOR, -1, cv2.LINE_AA)
def _draw_skeleton(frame: np.ndarray, tracks: Dict[int, Dict[str, Any]], frame_idx: int) -> None:
    scale = min(frame.shape[0], frame.shape[1])
    shadow_thickness = max(5, scale // 120)
    line_thickness = max(3, scale // 180)
    for start_idx, end_idx in SKELETON_EDGES:
        start = _track_point(tracks, start_idx, frame_idx)
        end = _track_point(tracks, end_idx, frame_idx)
        if start is None or end is None:
            continue
        cv2.line(frame, start, end, SKELETON_SHADOW, shadow_thickness, cv2.LINE_AA)
        cv2.line(frame, start, end, SKELETON_COLOR, line_thickness, cv2.LINE_AA)
    for joint_idx in TRACKED_JOINTS:
        point = _track_point(tracks, joint_idx, frame_idx)
        if point is None:
            continue
        _draw_joint(frame, point, scale)
def _overlay_panel(
    frame: np.ndarray,
    *,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    fill_color: Tuple[int, int, int],
    edge_color: Tuple[int, int, int],
    alpha: float = 0.88,
) -> None:
    x0 = max(0, min(frame.shape[1] - 1, x0))
    x1 = max(0, min(frame.shape[1], x1))
    y0 = max(0, min(frame.shape[0] - 1, y0))
    y1 = max(0, min(frame.shape[0], y1))
    if x1 <= x0 or y1 <= y0:
        return
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), fill_color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, frame)
    edge_thickness = 2 if min(frame.shape[0], frame.shape[1]) >= 520 else 1
    cv2.rectangle(frame, (x0, y0), (x1, y1), edge_color, edge_thickness, cv2.LINE_AA)
def _apply_bottom_scrim(
    frame: np.ndarray,
    *,
    start_y: int,
    end_y: Optional[int] = None,
    max_alpha: float = 0.48,
) -> None:
    height, width = frame.shape[:2]
    y0 = max(0, min(height - 1, int(start_y)))
    y1 = height if end_y is None else max(y0 + 1, min(height, int(end_y)))
    if y1 <= y0:
        return
    span = max(1, y1 - y0)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y0), (width, y1), (0, 0, 0), -1)
    alpha_mask = np.zeros((height, width), dtype=np.float32)
    gradient = np.linspace(0.0, float(max_alpha), span, dtype=np.float32)
    alpha_mask[y0:y1, :] = gradient[:, None]
    blended = (
        frame.astype(np.float32) * (1.0 - alpha_mask[..., None])
        + overlay.astype(np.float32) * alpha_mask[..., None]
    )
    frame[:, :] = np.clip(blended, 0, 255).astype(np.uint8)
