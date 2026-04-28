from __future__ import annotations
from .shared import *
from .font_utils import _load_theme_font, _pil_text_size
from .pil_context import _bgr_to_rgb, _frame_draw_context, _commit_frame_draw_context
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


def _draw_skeleton_legend(
    frame: np.ndarray,
    *,
    fps: float,
    frame_idx: int,
    legend_end_frame: int,
) -> None:
    if Image is None or ImageDraw is None or frame_idx >= legend_end_frame:
        return
    fade_frames = max(1, int(round(max(1.0, fps) * LEGEND_FADE_SECONDS)))
    fade_start = max(0, legend_end_frame - fade_frames)
    alpha_scale = 1.0
    if frame_idx >= fade_start:
        remaining = max(0, legend_end_frame - frame_idx)
        alpha_scale = max(0.0, min(1.0, float(remaining) / float(fade_frames)))
    width = frame.shape[1]
    height = frame.shape[0]
    scale = min(width, height)
    font = _load_theme_font(LABEL_FONT_FILE, max(18, int(round(scale * 0.038))))
    if font is None:
        return
    image, overlay, draw = _frame_draw_context(frame)
    rows = [("Skeleton", SKELETON_COLOR), ("Load / fault point", HOTSPOT_RING)]
    row_gap = max(8, int(round(scale * 0.014)))
    dot_r = max(7, int(round(scale * 0.014)))
    pad_x = max(16, int(round(scale * 0.032)))
    pad_y = max(12, int(round(scale * 0.024)))
    text_sizes = [_pil_text_size(draw, label, font) for label, _ in rows]
    max_text_w = max(size[0] for size in text_sizes)
    row_h = max(size[1] for size in text_sizes)
    panel_w = max_text_w + pad_x * 2 + dot_r * 2 + 12
    panel_h = pad_y * 2 + len(rows) * row_h + (len(rows) - 1) * row_gap
    x0 = 10
    y0 = 10
    x1 = min(width - 10, x0 + panel_w)
    y1 = min(height - 10, y0 + panel_h)
    fill_alpha = int(round(199 * alpha_scale))
    outline_alpha = int(round(44 * alpha_scale))
    text_alpha = int(round(255 * alpha_scale))
    draw.rounded_rectangle(
        (x0, y0, x1, y1),
        radius=max(8, int(round(scale * 0.016))),
        fill=(PANEL_BG[2], PANEL_BG[1], PANEL_BG[0], fill_alpha),
        outline=(255, 255, 255, outline_alpha),
    )
    current_y = y0 + pad_y
    for idx, (label, color) in enumerate(rows):
        dot_x = x0 + pad_x + dot_r
        dot_y = current_y + row_h // 2
        draw.ellipse(
            (dot_x - dot_r, dot_y - dot_r, dot_x + dot_r, dot_y + dot_r),
            fill=_bgr_to_rgb(color, text_alpha),
        )
        draw.text(
            (dot_x + dot_r + 8, current_y),
            label,
            font=font,
            fill=(255, 255, 255, text_alpha),
        )
        current_y += row_h + (row_gap if idx < len(rows) - 1 else 0)
    _commit_frame_draw_context(frame, image, overlay)
