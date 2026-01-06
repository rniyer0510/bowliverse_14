"""
Geometry helpers — ActionLab V14 (action)

We compute a single "batsman axis" (forward direction) using hip-centre
displacement over a short window around BFC. This is camera-agnostic in
the sense that it does not assume a fixed axis; it infers forward from motion.

Frame shape expected (V14 loader):
  pose_frames[i] = {"frame": i, "landmarks": [33 landmarks]}

Landmark indices:
  LH=23, RH=24
"""

import math
from typing import Any, Dict, List, Optional, Tuple

LH, RH = 23, 24
VIS_MIN = 0.5


def _get_vis(pt: Any) -> float:
    try:
        if isinstance(pt, dict):
            return float(pt.get("visibility", 1.0))
        return float(getattr(pt, "visibility", 1.0))
    except Exception:
        return 1.0


def _xy(pt: Any) -> Optional[Tuple[float, float]]:
    if pt is None:
        return None
    if isinstance(pt, dict) and "x" in pt and "y" in pt:
        try:
            return (float(pt["x"]), float(pt["y"]))
        except Exception:
            return None
    try:
        return (float(pt.x), float(pt.y))
    except Exception:
        return None


def vec(a: Any, b: Any) -> Optional[Tuple[float, float]]:
    aa = _xy(a)
    bb = _xy(b)
    if aa is None or bb is None:
        return None
    return (bb[0] - aa[0], bb[1] - aa[1])


def norm(v: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    if not v:
        return None
    x, y = float(v[0]), float(v[1])
    mag = math.hypot(x, y)
    if mag < 1e-9:
        return None
    return (x / mag, y / mag)


def angle_deg(v1: Optional[Tuple[float, float]], v2: Optional[Tuple[float, float]]) -> Optional[float]:
    """
    Acute angle between v1 and v2 in degrees (0..90),
    using abs(dot) so direction sign doesn't flip the intent.
    """
    if not v1 or not v2:
        return None
    a = norm(v1)
    b = norm(v2)
    if not a or not b:
        return None
    dot = max(-1.0, min(1.0, abs(a[0] * b[0] + a[1] * b[1])))
    return float(math.degrees(math.acos(dot)))


def compute_batsman_axis(
    pose_frames: List[Dict[str, Any]],
    bfc_frame: int,
    ffc_frame: Optional[int] = None,
) -> Optional[Tuple[float, float]]:
    """
    Infer "forward" (towards batsman) from hip-centre displacement.

    We take a short window ending at BFC (and slightly after),
    compute per-step displacement vectors of pelvis centre,
    and take median displacement.

    Returns a unit vector (dx, dy) or None if insufficient data.
    """
    if bfc_frame is None:
        return None
    if not isinstance(pose_frames, list) or not pose_frames:
        return None

    # window: mostly before BFC, a couple frames after (to stabilize)
    start = max(0, int(bfc_frame) - 12)
    end = min(len(pose_frames) - 1, int(bfc_frame) + 2)

    centres: List[Tuple[int, Tuple[float, float]]] = []

    for f in range(start, end + 1):
        fr = pose_frames[f]
        if not isinstance(fr, dict):
            continue
        lm = fr.get("landmarks")
        if not isinstance(lm, list) or len(lm) <= RH:
            continue

        lh = lm[LH]
        rh = lm[RH]
        if _get_vis(lh) < VIS_MIN or _get_vis(rh) < VIS_MIN:
            continue

        lh_xy = _xy(lh)
        rh_xy = _xy(rh)
        if lh_xy is None or rh_xy is None:
            continue

        cx = 0.5 * (lh_xy[0] + rh_xy[0])
        cy = 0.5 * (lh_xy[1] + rh_xy[1])
        centres.append((f, (cx, cy)))

    if len(centres) < 4:
        return None

    # displacement between consecutive valid centres
    dxs: List[float] = []
    dys: List[float] = []
    for i in range(1, len(centres)):
        _, (x0, y0) = centres[i - 1]
        _, (x1, y1) = centres[i]
        dx = x1 - x0
        dy = y1 - y0
        if math.hypot(dx, dy) < 1e-6:
            continue
        dxs.append(dx)
        dys.append(dy)

    if len(dxs) < 2:
        return None

    dxs.sort()
    dys.sort()
    mdx = dxs[len(dxs) // 2]
    mdy = dys[len(dys) // 2]

    ax = norm((mdx, mdy))
    return ax
