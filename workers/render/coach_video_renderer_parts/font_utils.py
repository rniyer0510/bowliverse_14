from __future__ import annotations
from .shared import *

_THEME_FONT_CACHE: Dict[Tuple[str, int], Any] = {}

def _theme_font_dirs() -> List[Path]:
    candidates: List[Path] = []
    explicit_font_dir = str(os.getenv(THEME_FONT_DIR_ENV) or "").strip()
    if explicit_font_dir:
        candidates.append(Path(explicit_font_dir))
    frontend_repo = str(os.getenv(THEME_FRONTEND_REPO_ENV) or "").strip()
    if frontend_repo:
        candidates.append(Path(frontend_repo) / "assets" / "fonts")
    home = Path.home()
    candidates.extend(
        [
            home / "dev" / "bowliverse_android_smoke" / "assets" / "fonts",
            home / "Documents" / "bowliverse_android_smoke" / "assets" / "fonts",
            home / "Documents" / "New project" / "assets" / "fonts",
        ]
    )
    module_dir = Path(__file__).parent
    candidates.extend(
        [
            module_dir / "fonts",
            module_dir.parent / "fonts",
        ]
    )
    deduped: List[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.exists():
            deduped.append(path)
    return deduped
def _load_theme_font(font_file: str, size: int) -> Any:
    if ImageFont is None:
        return None
    safe_size = max(10, int(size))
    cache_key = (font_file, safe_size)
    cached = _THEME_FONT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    font_dirs = _theme_font_dirs()
    logger.info(
        "[font_utils] dirs=%s file=%s size=%s",
        [str(path) for path in font_dirs],
        font_file,
        safe_size,
    )
    for font_dir in font_dirs:
        font_path = font_dir / font_file
        if not font_path.exists():
            continue
        try:
            font = ImageFont.truetype(str(font_path), safe_size)
            _THEME_FONT_CACHE[cache_key] = font
            return font
        except Exception:
            continue
    try:
        fallback = ImageFont.load_default()
    except Exception:
        fallback = None
    logger.warning(
        "[font_utils] fallback font file=%s size=%s dirs=%s",
        font_file,
        safe_size,
        [str(path) for path in font_dirs],
    )
    _THEME_FONT_CACHE[cache_key] = fallback
    return fallback
def _pil_text_size(draw: Any, text: str, font: Any) -> Tuple[int, int]:
    bbox = draw.textbbox((0, 0), str(text or ""), font=font)
    return max(0, int(bbox[2] - bbox[0])), max(0, int(bbox[3] - bbox[1]))
def _phase_label_font_size(scale: int) -> int:
    return max(22, int(round(max(1, int(scale)) * 0.046)))
def _legend_font_size(scale: int) -> int:
    return max(18, int(round(max(1, int(scale)) * 0.038)))
def _wrap_pil_text(
    draw: Any,
    text: str,
    *,
    font: Any,
    max_width: int,
    max_lines: int,
) -> List[str]:
    source = " ".join(str(text or "").split())
    if not source:
        return []
    words = source.split(" ")
    lines: List[str] = []
    current = ""
    overflow = False
    for word in words:
        candidate = word if not current else f"{current} {word}"
        width_px, _ = _pil_text_size(draw, candidate, font)
        if width_px <= max_width or not current:
            current = candidate
            continue
        lines.append(current)
        current = word
        if len(lines) >= max_lines:
            overflow = True
            break
    if current and not overflow:
        lines.append(current)
    if overflow:
        if current and len(lines) < max_lines:
            lines.append(current)
        lines = lines[:max_lines]
        tail = lines[-1]
        while tail:
            candidate = f"{tail}..."
            width_px, _ = _pil_text_size(draw, candidate, font)
            if width_px <= max_width:
                lines[-1] = candidate
                break
            tail = tail.rsplit(" ", 1)[0].strip()
        if not tail:
            lines[-1] = "..."
    return lines
def _fit_pil_wrapped_text(
    draw: Any,
    text: str,
    *,
    font_file: str,
    base_size: int,
    min_size: int,
    max_width: int,
    max_lines: int,
) -> Tuple[Any, List[str]]:
    for font_size in range(max(base_size, min_size), min_size - 1, -1):
        font = _load_theme_font(font_file, font_size)
        if font is None:
            break
        lines = _wrap_pil_text(
            draw,
            text,
            font=font,
            max_width=max_width,
            max_lines=max_lines,
        )
        if not lines:
            return font, []
        if len(lines) <= max_lines:
            return font, lines
    fallback_font = _load_theme_font(font_file, min_size)
    return fallback_font, _wrap_pil_text(
        draw,
        text,
        font=fallback_font,
        max_width=max_width,
        max_lines=max_lines,
    ) if fallback_font is not None else []
def _pil_text_block_height(
    draw: Any,
    lines: List[str],
    font: Any,
    *,
    line_gap: int,
) -> int:
    if font is None or not lines:
        return 0
    total = 0
    for idx, line in enumerate(lines):
        _, line_h = _pil_text_size(draw, line, font)
        total += line_h
        if idx < len(lines) - 1:
            total += line_gap
    return total
