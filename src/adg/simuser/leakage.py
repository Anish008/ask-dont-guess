"""Leakage detection for simulated user replies.

Per spec §11.3: Reject replies that contain gold answer strings.
"""

import re

from adg.core.logging import get_logger
from adg.data.schema import GoldInterpretation

log = get_logger(__name__)

# Minimum answer length to check for leakage (avoid false positives on "yes"/"no")
MIN_ANSWER_LENGTH_FOR_LEAKAGE_CHECK = 3


def normalize_for_leakage(text: str) -> str:
    """Normalize text for leakage detection.

    Args:
        text: Text to normalize

    Returns:
        Normalized text (lowercase, no punctuation)
    """
    normalized = text.lower()
    normalized = re.sub(r"[^\w\s]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def check_leakage(reply_text: str, intended: GoldInterpretation) -> bool:
    """Check if reply leaks gold answers.

    Per spec §11.3: Reply must not contain gold answer strings.

    Args:
        reply_text: User's reply
        intended: Intended interpretation with gold answers

    Returns:
        True if leakage detected
    """
    normalized_reply = normalize_for_leakage(reply_text)

    # Check each gold answer
    for answer in intended.answers:
        normalized_answer = normalize_for_leakage(answer)

        # Skip very short answers (avoid false positives on "yes"/"no")
        if len(normalized_answer) <= MIN_ANSWER_LENGTH_FOR_LEAKAGE_CHECK:
            continue

        if normalized_answer in normalized_reply:
            log.warning(
                "leakage_detected",
                reply=reply_text,
                leaked_answer=answer,
            )
            return True

    return False


__all__ = [
    "check_leakage",
    "normalize_for_leakage",
]
