"""Lightweight linguistic hints (assist model only; not a stand-alone decision engine)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple

from models.rule_pattern_banks import COMP_PATTERNS as _COMP
from models.rule_pattern_banks import GEN_PATTERNS as _GEN
from models.rule_pattern_banks import INEQ_TALK_PATTERNS as _INEQ_TALK

# Organisational / leadership *discourse* (criticism of systems, not demographic stereotype)
_STRUCTURAL = [
    re.compile(r"\bservant leadership\b", re.I),
    re.compile(r"\btransformational leadership\b", re.I),
    re.compile(r"\bleadership (model|theory|framework|style|philosophy|narrative)\b", re.I),
    re.compile(
        r"\b(?:organi[sz]ational|bureaucratic) (theory|design|culture|model|narrative|overhead|bloat|drag)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:bureaucracy|red tape|meeting[- ]?heavy|process (bloat|overhead|noise|drag)|middle management|management layer|management overhead)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:stakeholder (narrative|story)|value (chain|stream)|operating (model|rhythm|cadence))\b",
        re.I,
    ),
    re.compile(r"\b(?:critique|critics|criticise|criticize|lament) (of|that|how)\b", re.I),
    re.compile(r"\bover(?:engineering|optimiz|manag)ed\b", re.I),
]
# Mentions of roles as *workplace* constructs (weak signal; use with _STRUCTURAL)
_WORKPLACE_ROLE = re.compile(
    r"\b(managers?|executives?|pms?|project managers?|hr\b|people ops|leadership team)\b",
    re.I,
)


@dataclass
class HeuristicSignals:
    has_comparison: bool
    has_generalization: bool
    strength: float
    match_terms: List[str]
    is_inequality_discussion: bool
    # True = likely *workplace/institutional* commentary, not “X people are Y”
    is_structural_workplace_criticism: bool = False
    structural_phrase_hits: List[str] = field(default_factory=list)


def _first_span(pat: re.Pattern, text: str) -> str | None:
    m = pat.search(text)
    return m.group(0)[:64].strip() if m else None


def _structural_workplace_hits(text: str) -> List[str]:
    t = (text or "").strip()
    hits: List[str] = []
    for p in _STRUCTURAL:
        m = p.search(t)
        if m:
            hits.append(m.group(0)[:80].strip())
    return list(dict.fromkeys(hits))[:5]


def _is_structural_workplace_criticism(text: str) -> Tuple[bool, List[str]]:
    """Distinguish org/leadership *discourse* from group stereotype. Assist-only."""
    t = (text or "").strip()
    hits = _structural_workplace_hits(t)
    if not hits:
        return False, []
    lo = t.lower()
    if "servant leadership" in lo or "transformational leadership" in lo:
        return True, hits
    if re.search(r"\b(bureaucracy|bureaucratic|red tape|organi[sz]ational)\b", t, re.I):
        return True, hits
    if len(hits) >= 2:
        return True, hits
    if _WORKPLACE_ROLE.search(t):
        return True, hits
    if re.search(r"\b(leadership|stakeholder|narrative|overhead|process bloat|middle management)\b", t, re.I):
        return True, hits
    return False, hits


def analyze_linguistic_signals(text: str) -> HeuristicSignals:
    t = (text or "").strip()
    mcomp: List[str] = []
    mgen: List[str] = []
    mneq = False
    for p in _COMP:
        s = _first_span(p, t)
        if s:
            mcomp.append(s)
    for p in _GEN:
        s = _first_span(p, t)
        if s:
            mgen.append(s)
    for p in _INEQ_TALK:
        if p.search(t):
            mneq = True
            break

    has_c = bool(mcomp)
    has_g = bool(mgen)
    s = 0.0
    if has_c:
        s += 0.5
    if has_g:
        s += 0.4
    s = min(1.0, s)
    if has_c and has_g:
        s = min(1.0, s + 0.1)

    terms: List[str] = []
    for g in mcomp + mgen:
        w = re.sub(r"\s+", " ", g)
        if w and w not in terms:
            terms.append(w)
    terms = terms[:4]

    sw, sw_h = _is_structural_workplace_criticism(t)
    if sw and not has_c and not has_g:
        for x in sw_h:
            if x and x not in terms:
                terms.append(x)
    terms = terms[:5]

    return HeuristicSignals(
        has_comparison=has_c,
        has_generalization=has_g,
        strength=s,
        match_terms=terms,
        is_inequality_discussion=mneq,
        is_structural_workplace_criticism=sw,
        structural_phrase_hits=sw_h,
    )
