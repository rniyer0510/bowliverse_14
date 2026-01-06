"""
Signal quality helpers — ActionLab V14

This module must support BOTH frame shapes:

A) Landmark-list frames (current V14 loader):
   pose_frame = {"frame": i, "landmarks": [33 mediapipe landmarks]}

B) Legacy named-key frames (older experiments):
   pose_frame = {"left_hip": {...}, "right_hip": {...}, ...}

We keep one API: landmarks_visible(pose_frame, required, min_vis=0.5)
- required can be a list[int] (MediaPipe indices) OR list[str] (named keys)
"""

from typing import Any, Iterable, List, Sequence, Union

VIS_DEFAULT = 1.0


def _get_vis(pt: Any) -> float:
    try:
        if isinstance(pt, dict):
            return float(pt.get("visibility", VIS_DEFAULT))
        return float(getattr(pt, "visibility", VIS_DEFAULT))
    except Exception:
        return VIS_DEFAULT


def landmarks_visible(
    pose_frame: Any,
    required: Sequence[Union[int, str]],
    min_vis: float = 0.5,
) -> bool:
    if pose_frame is None or not required:
        return False

    # Case 1: required are indices => expect "landmarks" list
    if isinstance(required[0], int):
        if not isinstance(pose_frame, dict):
            return False
        lm = pose_frame.get("landmarks")
        if not isinstance(lm, list):
            return False

        for idx in required:  # type: ignore[assignment]
            if not isinstance(idx, int):
                return False
            if idx < 0 or idx >= len(lm):
                return False
            pt = lm[idx]
            if pt is None:
                return False
            if _get_vis(pt) < min_vis:
                return False
        return True

    # Case 2: required are names => expect dict keys
    if not isinstance(pose_frame, dict):
        return False

    for name in required:  # type: ignore[assignment]
        if not isinstance(name, str):
            return False
        pt = pose_frame.get(name)
        if pt is None:
            return False
        if _get_vis(pt) < min_vis:
            return False

    return True
