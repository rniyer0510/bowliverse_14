from app.workers.action.intent_state import (
    IntentState,
    soften_open,
    soften_closed,
)
from app.common.geometry import project_to_screen_plane
from app.common.signal_quality import landmarks_visible


def classify_action(ctx):
    """
    Final action classification.
    - BFC-only
    - Single snapshot
    - No averaging
    - No angle thresholds
    """

    bfc = ctx.events.bfc
    pose = ctx.pose.frames[bfc.frame]

    # --------------------------------------------------------------
    # Guardrail: visibility (ONLY allowed UNKNOWN case)
    # --------------------------------------------------------------
    if not landmarks_visible(pose, ["left_heel", "left_toe"]):
        return {
            "action": "UNKNOWN",
            "intent": None,
        }

    # --------------------------------------------------------------
    # Step 1: Toe-based primary intent
    # --------------------------------------------------------------
    heel = pose["left_heel"]
    toe = pose["left_toe"]

    toe_vec = project_to_screen_plane(toe - heel)

    if toe_vec.is_parallel_to_pitch():
        toe_state = IntentState.CLOSED
    elif toe_vec.is_mostly_parallel():
        toe_state = IntentState.SEMI_CLOSED
    elif toe_vec.is_diagonal():
        toe_state = IntentState.SEMI_OPEN
    else:
        toe_state = IntentState.OPEN

    # --------------------------------------------------------------
    # Step 2: Hip softening (secondary, non-authoritative)
    # --------------------------------------------------------------
    hip_vec = project_to_screen_plane(
        pose["left_hip"] - pose["right_hip"]
    )

    intent = toe_state
    possible_contradiction = False

    if hip_vec.is_more_open_than(toe_vec):
        intent = soften_open(intent)
    elif hip_vec.is_more_closed_than(toe_vec):
        intent = soften_closed(intent)

    if toe_state == IntentState.OPEN and intent in (
        IntentState.CLOSED,
        IntentState.SEMI_CLOSED,
    ):
        possible_contradiction = True

    # --------------------------------------------------------------
    # Step 3: Shoulder phase (context only)
    # --------------------------------------------------------------
    shoulder_vec = project_to_screen_plane(
        pose["left_shoulder"] - pose["right_shoulder"]
    )

    if shoulder_vec.lags(intent):
        shoulder_phase = "LAGGING"
    elif shoulder_vec.leads(intent):
        shoulder_phase = "LEADING"
    else:
        shoulder_phase = "SYNCED"

    # --------------------------------------------------------------
    # Step 4: Final Action Mapping
    # --------------------------------------------------------------
    if possible_contradiction:
        action = "MIXED"

    elif intent in (IntentState.CLOSED, IntentState.SEMI_CLOSED):
        if shoulder_phase == "SYNCED":
            action = "SIDE_ON"
        else:
            action = "SIDE_ON_OPENING"

    elif intent == IntentState.SEMI_OPEN:
        if shoulder_phase == "LAGGING":
            action = "SIDE_ON_OPENING"
        else:
            action = "FRONT_ON_CLOSING"

    else:  # OPEN
        if shoulder_phase == "SYNCED":
            action = "FRONT_ON"
        else:
            action = "FRONT_ON_CLOSING"

    return {
        "action": action,
        "intent": intent.value,
    }
