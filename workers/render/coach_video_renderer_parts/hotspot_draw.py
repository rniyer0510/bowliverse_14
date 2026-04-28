from __future__ import annotations
from .shared import *
from .drawing_base import _overlay_panel
from .font_utils import _load_theme_font, _pil_text_size
from .pil_context import _bgr_to_rgb, _frame_draw_context, _commit_frame_draw_context

def _draw_hotspot_marker(
    frame: np.ndarray,
    *,
    center: Tuple[int, int],
    scale: int,
    weight: float,
    pulse_phase: float,
) -> None:
    pulse = 0.5 - 0.5 * np.cos(float(pulse_phase) * np.pi * 2.0)
    pulse_weight = max(0.0, min(1.0, weight))
    base = max(5, scale // 84)
    ring_step = max(4, scale // 72)
    inner_ring = int(round(base + ring_step * (0.86 + pulse_weight * 0.08)))
    outer_ring = int(round(base + ring_step * (1.92 + pulse_weight * 0.16 + pulse * 0.10)))
    ring_thickness = max(2, scale // 260)
    cv2.circle(frame, center, outer_ring, HOTSPOT_RING, ring_thickness + 1, cv2.LINE_AA)
    cv2.circle(frame, center, inner_ring, HOTSPOT_RING, ring_thickness, cv2.LINE_AA)
    cv2.circle(frame, center, base, HOTSPOT_DOT, -1, cv2.LINE_AA)
def _draw_hotspot_pointer_line(
    frame: np.ndarray,
    *,
    center: Tuple[int, int],
    direction: Tuple[float, float],
    scale: int,
) -> None:
    dx, dy = direction
    mag = max(1e-6, float(np.hypot(dx, dy)))
    ndx = dx / mag
    ndy = dy / mag
    length = max(26, scale // 7)
    start = (
        int(round(center[0] - ndx * length)),
        int(round(center[1] - ndy * length)),
    )
    end = (
        int(round(center[0] - ndx * max(8, scale // 32))),
        int(round(center[1] - ndy * max(8, scale // 32))),
    )
    thickness = max(2, scale // 200)
    cv2.line(frame, start, end, HOTSPOT_RING, thickness + 3, cv2.LINE_AA)
    cv2.line(frame, start, end, HOTSPOT_SOFT, thickness + 1, cv2.LINE_AA)
def _draw_hotspot_compact_label(
    frame: np.ndarray,
    *,
    center: Tuple[int, int],
    direction: Tuple[float, float],
    label: str,
    scale: int,
    occupied_rects: Optional[List[Tuple[int, int, int, int]]] = None,
) -> Optional[Tuple[int, int, int, int]]:
    if not label:
        return None
    width = frame.shape[1]
    height = frame.shape[0]
    if Image is None or ImageDraw is None:
        return None
    font = _load_theme_font(LABEL_FONT_FILE, max(22, int(round(scale * 0.046))))
    if font is None:
        return None
    measure_image = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    measure_draw = ImageDraw.Draw(measure_image)
    text_w, text_h = _pil_text_size(measure_draw, label, font)
    pad_x = max(14, scale // 46)
    pad_y = max(8, scale // 68)
    dx, dy = direction
    candidates = [
        (center[0] + int(scale * 0.10), center[1] - int(scale * 0.05)),
        (center[0] - int(scale * 0.10) - text_w - pad_x * 2, center[1] - int(scale * 0.05)),
        (center[0] + int(scale * 0.10), center[1] + int(scale * 0.05)),
        (center[0] - int(scale * 0.10) - text_w - pad_x * 2, center[1] + int(scale * 0.05)),
        (center[0] - text_w // 2 - pad_x, center[1] - int(scale * 0.11)),
        (center[0] - text_w // 2 - pad_x, center[1] + int(scale * 0.08)),
    ]
    best = None
    best_score = float("-inf")
    occupied_rects = occupied_rects or []
    for cand_x0, cand_y0_base in candidates:
        cand_x0 = int(cand_x0)
        cand_y0 = int(cand_y0_base - text_h // 2 - pad_y)
        cand_x0 = max(12, min(width - text_w - pad_x * 2 - 12, cand_x0))
        cand_y0 = max(12, min(height - text_h - pad_y * 2 - 12, cand_y0))
        x_center = cand_x0 + (text_w + pad_x * 2) / 2.0
        y_center = cand_y0 + (text_h + pad_y * 2) / 2.0
        vec_x = x_center - center[0]
        vec_y = y_center - center[1]
        dist_score = min(1.0, float(np.hypot(vec_x, vec_y)) / max(1.0, scale * 0.16))
        dir_score = 0.0
        if abs(dx) >= abs(dy):
            dir_score = 1.0 if (dx >= 0 and vec_x >= 0) or (dx < 0 and vec_x <= 0) else 0.0
        else:
            dir_score = 1.0 if (dy >= 0 and vec_y >= 0) or (dy < 0 and vec_y <= 0) else 0.0
        top_panel_penalty = 0.0
        if cand_y0 < int(height * 0.24) and cand_x0 < int(width * 0.72):
            top_panel_penalty = 1.0
        overlap_penalty = 0.0
        cand_rect = (
            cand_x0,
            cand_y0,
            cand_x0 + text_w + pad_x * 2,
            cand_y0 + text_h + pad_y * 2,
        )
        for other_x0, other_y0, other_x1, other_y1 in occupied_rects:
            if not (
                cand_rect[2] <= other_x0
                or cand_rect[0] >= other_x1
                or cand_rect[3] <= other_y0
                or cand_rect[1] >= other_y1
            ):
                overlap_penalty += 2.5
        score = dir_score * 1.4 + dist_score * 0.6 - top_panel_penalty * 2.0 - overlap_penalty
        if score > best_score:
            best_score = score
            best = (cand_x0, cand_y0)
    x0, y0 = best if best is not None else (12, 12)
    x1 = x0 + text_w + pad_x * 2
    y1 = y0 + text_h + pad_y * 2
    _overlay_panel(
        frame,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        fill_color=PANEL_BG,
        edge_color=HOTSPOT_RING,
        alpha=0.72,
    )
    image, overlay, draw = _frame_draw_context(frame)
    draw.text(
        (x0 + pad_x, y0 + pad_y),
        label,
        font=font,
        fill=_bgr_to_rgb(TEXT_COLOR, 255),
    )
    _commit_frame_draw_context(frame, image, overlay)
    return (x0, y0, x1, y1)
