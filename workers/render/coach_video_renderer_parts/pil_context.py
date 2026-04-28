from __future__ import annotations
from .shared import *

def _bgr_to_rgb(color: Tuple[int, int, int], alpha: Optional[int] = None) -> Tuple[int, ...]:
    rgb = (int(color[2]), int(color[1]), int(color[0]))
    if alpha is None:
        return rgb
    return rgb + (int(alpha),)
def _frame_draw_context(frame: np.ndarray) -> Tuple[Any, Any, Any]:
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    return image, overlay, draw
def _commit_frame_draw_context(frame: np.ndarray, image: Any, overlay: Any) -> None:
    composited = Image.alpha_composite(image, overlay).convert("RGB")
    frame[:, :] = cv2.cvtColor(np.array(composited), cv2.COLOR_RGB2BGR)
