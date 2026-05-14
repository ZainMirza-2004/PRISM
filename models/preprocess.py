"""Lightweight normalization for social text: lowercase, noise removal, keep punctuation."""

from __future__ import annotations

import re
import unicodedata


# Remove control/surrogate noise; keep letters, numbers, common punctuation and apostrophes.
_NOISE_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
# Collapse whitespace; keep single spaces between tokens.
_WS = re.compile(r"\s+")


def preprocess_social_post(text: str) -> str:
    """Lowercase, strip weird characters, normalize whitespace; punctuation retained for tone."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", str(text))
    t = _NOISE_CTRL.sub("", t)
    t = t.lower().strip()
    t = _WS.sub(" ", t)
    return t
