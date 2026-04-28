from __future__ import annotations
from .shared import *

def _wrap_text_lines(
    text: str,
    *,
    max_width: int,
    font_scale: float,
    thickness: int,
    max_lines: Optional[int] = None,
) -> List[str]:
    if max_width <= 0:
        return [str(text or "").strip()]
    lines: List[str] = []
    for paragraph in str(text or "").splitlines() or [""]:
        words = paragraph.split()
        if not words:
            if lines:
                lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            width_px, _ = cv2.getTextSize(candidate, TEXT_FONT, font_scale, thickness)[0]
            if width_px <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        last = lines[-1].rstrip()
        while last:
            width_px, _ = cv2.getTextSize(f"{last}...", TEXT_FONT, font_scale, thickness)[0]
            if width_px <= max_width:
                break
            last = last[:-1].rstrip()
        lines[-1] = f"{last}..." if last else "..."
    return lines
def _fit_wrapped_text(
    text: str,
    *,
    max_width: int,
    max_lines: int,
    base_scale: float,
    min_scale: float,
    thickness: int,
) -> Tuple[List[str], float]:
    scale = float(base_scale)
    while scale >= float(min_scale):
        lines = _wrap_text_lines(
            text,
            max_width=max_width,
            font_scale=scale,
            thickness=thickness,
            max_lines=max_lines,
        )
        if len(lines) <= max_lines:
            return lines, scale
        scale -= 0.03
    return (
        _wrap_text_lines(
            text,
            max_width=max_width,
            font_scale=min_scale,
            thickness=thickness,
            max_lines=max_lines,
        ),
        float(min_scale),
    )
