"""Structural rule layer: binary flags for fusion (precision structural signal)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from models.critique_lexicon import contains_negation_or_critique
from models.rule_pattern_banks import COMP_PATTERNS, GEN_PATTERNS, INEQ_TALK_PATTERNS
from models.stance_features import extract_stance_features

# ---------------------------------------------------------------------------
# Core pattern banks (staged detection)
# ---------------------------------------------------------------------------

# Tight “X are / is” + group head (use when text also matches critique markers)
_STRONG_EXPLICIT_GEN: list[re.Pattern[str]] = [
    re.compile(
        r"\b(?:people|women|men|they|immigrants?|locals?|foreigners?|workers|"
        r"candidates?|employees?|engineers?|managers?|staff|hires?)\s+(?:are|is)\b",
        re.I,
    ),
    re.compile(r"\bmost\s+\w+\s+(?:are|is)\b", re.I),
    re.compile(
        r"\b\w+(?:\s+\w+){0,3}\s+(?:are|is)\s+(?:not\s+)?(?:usually|often|always|never)\b",
        re.I,
    ),
]

# Additional generalisation shapes (only when not in “critique gate” mode)
_EXTRA_GENERAL_NON_CRITIQUE: list[re.Pattern[str]] = [
    re.compile(
        r"\b(?:detail-oriented|big-picture|leader\s+type|people\s+person)\b"
        r".{0,50}\b(?:versus|vs|compared|rather\s+than|not)\b",
        re.I,
    ),
    re.compile(
        r"\bnot\s+a\s+(?:leader|manager|people)\s+type\b",
        re.I,
    ),
]

# “Hard” comparison / rank (overlaps with COMP_PATTERNS; OR-merged for one bit)
_COMPARISON_EXPLICIT: list[re.Pattern[str]] = [
    re.compile(r"\bbetter\s+than\b", re.I),
    re.compile(r"\bworse\s+than\b", re.I),
    re.compile(r"\boutperform(?:s|ed|ing)?\b", re.I),
    re.compile(r"\b(?:more|less)\s+suited\b", re.I),
    re.compile(r"\bover\s+others\b", re.I),
    re.compile(
        r"\b(?:detail[-\s]?oriented|big[-\s]?picture|soft\s+skills)\b"
        r".{0,40}\b(?:versus|vs|compared|than|over)\b",
        re.I,
    ),
]

# Hard hiring / role preference
_HARD_PREFERENCE: list[re.Pattern[str]] = [
    re.compile(r"\bwe\s+prefer\b", re.I),
    re.compile(r"\bi\s+prefer\b", re.I),
    re.compile(r"\bour\s+preference\b", re.I),
    re.compile(
        r"\bprefer(?:s|red|ring)?\s+(?:to|over|locals?|immigrants?|candidates?|foreign|domestic)\b",
        re.I,
    ),
    re.compile(r"\b(?:shortlist|prioriti[sz]e|prioritise)\b", re.I),
    re.compile(r"\b(?:hire|hiring)\s+(?:locals?|local\s+candidates?)\b", re.I),
]

_ROLE_ASSIGN: list[re.Pattern[str]] = [
    re.compile(
        r"\b(?:people|women|men|they|immigrants?|locals?)\s+should\b",
        re.I,
    ),
    re.compile(
        r"\bshould\s+(?:not\s+)?(?:be|do|have|work|stay)\b",
        re.I,
    ),
]

# Soft fit / evaluative preference language (own dimension)
_SOFT_PREFERENCE: list[re.Pattern[str]] = [
    re.compile(r"\bgravitate\b", re.I),
    re.compile(r"\blean\s+towards?\b", re.I),
    re.compile(r"\bstronger\s+fit\b", re.I),
    re.compile(r"\bsmoother\b", re.I),
    re.compile(r"\bless\s+friction\b", re.I),
    re.compile(r"\bbetter\s+dynamic\b", re.I),
    re.compile(r"\baligned\s+with\s+how\s+we\s+work\b", re.I),
    re.compile(r"\bnatural\s+fit\b", re.I),
    re.compile(r"\blow[-\s]friction\b", re.I),
    re.compile(r"\bmore\s+polish\b", re.I),
    re.compile(r"\bexecutive\s+presence\b", re.I),
    re.compile(r"\bclient[-\s]facing\s+english\b", re.I),
    re.compile(r"\btime[-\s]zone\s+alignment\b", re.I),
    re.compile(r"\bsafe\s+pair\s+of\s+hands\b", re.I),
    re.compile(r"\bpedigree\s*(?:mismatch|issues?)?\b", re.I),
    re.compile(
        r"\b(?:strong|weaker)\s+(?:on|at)\s+(?:soft|hard)\s+skills?\b",
        re.I,
    ),
]

# Coded corporate / evaluative euphemism (extends legacy list)
_CODED_BIAS: list[re.Pattern[str]] = [
    re.compile(r"\bculture\s+fit\b", re.I),
    re.compile(r"\bculture\s+add\b", re.I),
    re.compile(r"\bpractical\s+choice\b", re.I),
    re.compile(r"\blow(?:er|est)(?:-|\s)risk\s+(?:hire|bet|option)\b", re.I),
    re.compile(r"\brealistic\s+(?:hire|choice)\b", re.I),
    re.compile(r"\bsafe\s+(?:pair|bet|hire)\b", re.I),
    re.compile(r"\barticulate\b", re.I),
    re.compile(r"\bpolish\b", re.I),
    re.compile(r"\bpedigree\b", re.I),
    re.compile(r"\bpolished\s+communicator\b", re.I),
]

# Implicit tendency markers must co-occur with a group/proxy term in a local window
_IMPLICIT_MARKERS = re.compile(
    r"\b(?:tend(?:s|ency)?\s+to|often|usually|typically|pattern\s+i\s+see|"
    r"i\s+(?:often|usually|typically)|we\s+(?:often|usually))\b",
    re.I,
)

_GROUP_OR_PROXY = re.compile(
    r"\b(?:women|men|woman|man|female|male|she|her|him|he|girls?|guys?|"
    r"immigrants?|locals?|foreign(?:ers)?|visa\s+candidates?|"
    r"candidates?|employees?|workers?|engineers?|managers?|leaders?|"
    r"hires?|people|staff|team(?:mates?)?|applicants?)\b",
    re.I,
)

_HIRING_CONTEXT = re.compile(
    r"\b(?:hire|hiring|shortlist|offer|role|position|open\s+role|req|requisition|"
    r"interview|candidate|apply|sponsorship|visa)\b",
    re.I,
)

_MIGRATION_SUBTEXT = re.compile(
    r"\b(?:visa|sponsorship|timezone|time\s*zone|client-facing\s+english|"
    r"english\s+proficiency|relocation)\b",
    re.I,
)

_DEBIAS_GUARDS: list[re.Pattern[str]] = [
    re.compile(
        r"\b(?:women|men|immigrants?|locals?|foreigners?|workers?|employees?|engineers?|managers?|people|staff)\s+are\s+not\s+"
        r"(?:lazy|inferior|the\s+problem|stealing\s+jobs?|criminals?|unreliable|dangerous)\b",
        re.I,
    ),
    re.compile(r"\bnot\s+(?:stealing\s+jobs?|inferior|the\s+problem)\b", re.I),
    re.compile(r"\bit\s+is\s+a\s+(?:myth|stereotype)\s+that\b", re.I),
    re.compile(r"\bthe\s+(?:myth|stereotype)\s+that\b", re.I),
]

_NEG_EVAL_TOWARD_GROUP = re.compile(
    r"\b(?:lazy|inferior|weak|unreliable|aggressive|emotional|incompetent|unsafe|problem|burden|threat|steal(?:ing)?\s+jobs?)\b",
    re.I,
)

_PRESCRIPTIVE_OR_EXCLUSION = re.compile(
    r"\b(?:should\s+not\s+hire|do\s+not\s+hire|don't\s+hire|must\s+avoid|prefer\s+\w+\s+over|better\s+than|more\s+suited|less\s+suited|"
    r"not\s+a\s+fit|keep\s+out|favor\s+\w+)\b",
    re.I,
)


def _any_match(patterns: list[re.Pattern[str]], text: str) -> bool:
    return any(p.search(text) for p in patterns)


def _implicit_tendency_with_group(text: str) -> bool:
    """Epistemic tendency + group/proxy within a loose local span."""
    for m in _IMPLICIT_MARKERS.finditer(text):
        lo = max(0, m.start() - 90)
        hi = min(len(text), m.end() + 90)
        span = text[lo:hi]
        if _GROUP_OR_PROXY.search(span):
            return True
    # ‘Pattern’ / ‘in my experience’ style without implicit marker regex but with stereotype cue
    if re.search(r"\bin\s+my\s+experience\b", text, re.I):
        lo = text.lower().find("in my experience")
        if lo >= 0:
            span = text[max(0, lo - 60) : min(len(text), lo + 120)]
            if _GROUP_OR_PROXY.search(span):
                return True
    return False


def _migration_hiring_frame(text: str) -> bool:
    return bool(_HIRING_CONTEXT.search(text) and _MIGRATION_SUBTEXT.search(text))


def _match_explicit_generalisation(text: str, critique: bool) -> bool:
    if _any_match(_DEBIAS_GUARDS, text):
        return False
    if critique:
        # Quoted / meta references (“stereotype that women are…”) — not endorsement.
        if re.search(
            r"\b(?:stereotype|myth|strawman|the\s+idea\s+that|notion\s+that|trope)\b",
            text,
            re.I,
        ):
            return False
    has_gen_shape = (
        _any_match(GEN_PATTERNS, text)
        or _any_match(_EXTRA_GENERAL_NON_CRITIQUE, text)
        or _any_match(_STRONG_EXPLICIT_GEN, text)
    )
    if not has_gen_shape:
        return False
    st = extract_stance_features(text)
    harmful_cue = bool(
        _NEG_EVAL_TOWARD_GROUP.search(text)
        or _PRESCRIPTIVE_OR_EXCLUSION.search(text)
        or (
            st.group_target_present > 0
            and st.attribution_assertion > 0
            and st.attribution_denial == 0
            and st.attribution_critique_of_stereotype == 0
            and (st.sentiment_toward_group > 0 or st.attribution_endorsement > 0 or st.essentialist_claim_score > 0)
        )
    )
    return harmful_cue


def _match_implicit_generalisation(text: str, critique: bool) -> bool:
    if critique:
        return False
    return _implicit_tendency_with_group(text)


def _match_soft_preference(text: str) -> bool:
    return _any_match(_SOFT_PREFERENCE, text)


def _match_hard_preference(text: str) -> bool:
    if _any_match(_HARD_PREFERENCE, text) or _any_match(_ROLE_ASSIGN, text):
        return True
    if _migration_hiring_frame(text):
        return True
    return False


def _match_comparative(text: str) -> bool:
    return _any_match(_COMPARISON_EXPLICIT, text) or _any_match(COMP_PATTERNS, text)


def _match_coded_euphemism(text: str) -> bool:
    return _any_match(_CODED_BIAS, text)


def _match_inequality_discourse(text: str) -> bool:
    return _any_match(INEQ_TALK_PATTERNS, text)


@dataclass
class RuleFusionSignals:
    """Binary cues for fusion + meta-classifier (extended categorical channels)."""

    generalisation: int
    comparison: int
    preference: int  # hard preference / role assignment / migration-hiring frame
    coded_bias: int
    inequality_context: int
    soft_preference: int = 0
    implicit_generalisation: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "generalisation": self.generalisation,
            "comparison": self.comparison,
            "preference": self.preference,
            "coded_bias": self.coded_bias,
            "inequality_context": self.inequality_context,
            "soft_preference": self.soft_preference,
            "implicit_generalisation": self.implicit_generalisation,
        }


def extract_rule_fusion_signals(clean_text: str) -> RuleFusionSignals:
    """Detect structural cues on preprocessed text (staged; critique-gated implicit/loose gen)."""
    t = (clean_text or "").strip()
    if not t:
        return RuleFusionSignals(0, 0, 0, 0, 0, 0, 0)

    critique = contains_negation_or_critique(t)

    explicit_gen = _match_explicit_generalisation(t, critique)
    implicit_gen = _match_implicit_generalisation(t, critique)
    gen = 1 if (explicit_gen or implicit_gen) else 0
    implicit_bit = 1 if implicit_gen else 0

    comp = 1 if _match_comparative(t) else 0
    hard_pref = _match_hard_preference(t)
    soft_pref = _match_soft_preference(t)
    pref = 1 if hard_pref else 0
    soft_pref_bit = 1 if soft_pref else 0

    coded = 1 if _match_coded_euphemism(t) else 0
    ineq = 1 if _match_inequality_discourse(t) else 0

    return RuleFusionSignals(gen, comp, pref, coded, ineq, soft_pref_bit, implicit_bit)


def structure_score_from_rules(r: RuleFusionSignals) -> float:
    """0.4*G + 0.3*C + 0.2*P + 0.1*coded_bias (unchanged weights; inequality/soft/implicit excluded)."""
    return (
        0.4 * float(r.generalisation)
        + 0.3 * float(r.comparison)
        + 0.2 * float(r.preference)
        + 0.1 * float(r.coded_bias)
    )
