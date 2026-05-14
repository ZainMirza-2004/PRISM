"""Single source of truth for regex banks shared by rule_signals and interpretation_signals."""

from __future__ import annotations

import re

# Group-oriented comparisons and preferences (HeuristicSignals + fusion rule overlap)
COMP_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(?:more|less)\s+suited\b", re.I),
    re.compile(r"\bprefer(?:s|red|ring)?\b", re.I),
    re.compile(r"\b(?:shortlist|prioriti[sz]e|prioritise)\b", re.I),
    re.compile(r"\bbetter\s+than\b", re.I),
    re.compile(r"\bworse\s+than\b", re.I),
    re.compile(r"\boutperform(?:s|ed)?\b", re.I),
    re.compile(r"\b(?:faster|slower|cheaper|safer)\s+than\b", re.I),
    re.compile(r"\b(?:reliable|reliability|dependable|trust\w*)\b.*\b(than|over|versus|vs)\b", re.I),
    re.compile(r"\blow(?:er|est)(?:-|\s)risk\s+(?:hire|bet|option)\b", re.I),
    re.compile(r"\bover\s+others\b", re.I),
    re.compile(r"\b(?:in\s+my\s+experience|i\s+usually|we\s+usually|we\s+often)\b", re.I),
    re.compile(
        r"\b(?:not\s+(?:a\s+)?(?:leadership|manager|executive)\s+type|"
        r"more\s+of\s+a\s+(?:technical|people)\s+track|"
        r"vocational\s+vs\s+(?:academic|university)|"
        r"trades?\s+vs\s+(?:college|degree)|"
        r"blue[-\s]?collar\s+vs\s+white[-\s]?collar)\b",
        re.I,
    ),
]

# Broad generalizations (stereotyping cues)
GEN_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(?:\w+\s+){0,2}(?:are|is)\s+usually\b", re.I),
    re.compile(r"\b(?:\w+\s+){0,2}tend(?:s|ency)?\s+to\b", re.I),
    re.compile(r"\bmost\s+\w+\s+(?:are|is)\b", re.I),
    re.compile(r"\bin\s+general\b", re.I),
    re.compile(r"\b(?:always|never)\s+.*\b(people|workers|candidates|staff|hires?)\b", re.I),
    re.compile(r"\b(?:naturally|inherently|obviously|clearly).{0,50}\b(people|workers|group)\b", re.I),
    re.compile(
        r"\bnot\s+(?:saying|trying|looking)\b.{0,80}\b(but|still|usually|often|pattern)\b",
        re.I,
    ),
]

# Structural inequality discussion (often neutral)
INEQ_TALK_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bunfair(ly|ness)?\b", re.I),
    re.compile(r"\bdispari(?:ty|ties)\b", re.I),
    re.compile(r"\bbarrier|underrepresent|inequit", re.I),
    re.compile(r"\bbias in\b|\bbias in feedback\b", re.I),
    re.compile(r"\bminority|marginali[sz]ed|systemic|structural\b", re.I),
    re.compile(r"\bface\s+barrier|slower promotion|harsh(er)?\s+judg", re.I),
]
