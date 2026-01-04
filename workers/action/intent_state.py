from enum import Enum

class IntentState(Enum):
    CLOSED = "closed"
    SEMI_CLOSED = "semi_closed"
    SEMI_OPEN = "semi_open"
    OPEN = "open"


def soften_open(state: IntentState) -> IntentState:
    if state == IntentState.CLOSED:
        return IntentState.SEMI_CLOSED
    if state == IntentState.SEMI_CLOSED:
        return IntentState.SEMI_OPEN
    if state == IntentState.SEMI_OPEN:
        return IntentState.OPEN
    return state


def soften_closed(state: IntentState) -> IntentState:
    if state == IntentState.OPEN:
        return IntentState.SEMI_OPEN
    if state == IntentState.SEMI_OPEN:
        return IntentState.SEMI_CLOSED
    return state
