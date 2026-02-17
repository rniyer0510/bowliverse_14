"""
Basic rule-based moderation for coach notes.
Upgradeable to LLM moderation later without changing API contract.
"""

from typing import Tuple

# Expand cautiously over time
PROFANITY = {
    "idiot", "stupid", "useless", "worthless",
    "loser", "shut up"
}

THREATS = {
    "kill", "hurt you", "destroy you"
}

SEXUAL_CONTENT = {
    "sex", "nude", "explicit"
}

SELF_HARM = {
    "kill yourself", "suicide", "self harm"
}

HATE_SPEECH = {
    # Add region-specific slurs carefully
}

MAX_LENGTH = 1000


def _contains_phrase(text: str, phrases: set) -> bool:
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in phrases)


def moderate_coach_note(text: str) -> Tuple[bool, str]:
    """
    Returns:
        (is_allowed, reason_if_blocked)
    """

    if not text or not text.strip():
        return False, "Note cannot be empty."

    if len(text) > MAX_LENGTH:
        return False, "Note exceeds maximum allowed length."

    if _contains_phrase(text, PROFANITY):
        return False, "Please keep feedback respectful."

    if _contains_phrase(text, THREATS):
        return False, "Threatening language is not allowed."

    if _contains_phrase(text, SEXUAL_CONTENT):
        return False, "Inappropriate content detected."

    if _contains_phrase(text, SELF_HARM):
        return False, "Self-harm related content is not allowed."

    if _contains_phrase(text, HATE_SPEECH):
        return False, "Discriminatory language is not allowed."

    return True, ""
