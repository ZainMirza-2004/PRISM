"""Shared critique / reporting markers: fusion safeguards + rule-layer false-positive reduction."""

from __future__ import annotations

import re

# Used by fusion_engine.contains_negation_or_critique and rule_signals gating.
_CRITIQUE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(?:wrong|incorrect|false|myth|harmful|offensive|problematic)\b", re.I),
    re.compile(r"\b(?:shouldn't|should\s+not|mustn't|must\s+not)\b", re.I),
    re.compile(r"\bnot\s+(?:true|okay|ok|acceptable|accurate|fair)\b", re.I),
    re.compile(r"\b(?:calling\s+out|call\s+out|condemn|reject|pushback)\b", re.I),
    re.compile(r"\bdon'?t\s+(?:say|believe|think|assume)\b", re.I),
    re.compile(r"\b(?:stereotype|bias)\s+(?:is\s+)?(?:wrong|harmful|bad)\b", re.I),
    re.compile(r"\bis\s+wrong\b", re.I),
    re.compile(r"\bsaying\s+.+\s+is\s+(?:wrong|harmful|bad|false)\b", re.I),
    # Reporting / meta-discourse (quoted bias, studies, journalism)
    re.compile(r"\b(?:stud(?:y|ies)|research)\s+(?:shows?|found|suggests?|say)\b", re.I),
    re.compile(r"\baccording\s+to\s+(?:a\s+)?(?:study|research|survey|report)\b", re.I),
    re.compile(r"\b(?:article|paper|report)\s+(?:claims?|argues?|states?)\b", re.I),
    re.compile(r"\bchallenge\s+(?:the\s+)?(?:idea|notion|claim)\b", re.I),
    re.compile(r"\b(?:push\s+back|debunk|refute)\s+(?:on|against)?\b", re.I),
]


def contains_negation_or_critique(text: str) -> bool:
    """True if text likely negates bias, critiques stereotypes, or reports third-party claims."""
    t = (text or "").strip()
    if not t:
        return False
    return any(p.search(t) for p in _CRITIQUE_PATTERNS)
