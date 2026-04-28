from __future__ import annotations
from .shared import *

def _front_leg_joints(hand: Optional[str]) -> Tuple[int, int, int]:
    is_left_handed = str(hand or "R").upper().startswith("L")
    if is_left_handed:
        return RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE
    return LEFT_HIP, LEFT_KNEE, LEFT_ANKLE
def _foot_indices(hand: Optional[str]) -> Tuple[int, int, int]:
    is_left_handed = str(hand or "R").upper().startswith("L")
    if is_left_handed:
        return RIGHT_FOOT_INDEX, RIGHT_HEEL, LEFT_FOOT_INDEX
    return LEFT_FOOT_INDEX, LEFT_HEEL, RIGHT_FOOT_INDEX


__all__ = [name for name in globals() if name != "__builtins__"]
