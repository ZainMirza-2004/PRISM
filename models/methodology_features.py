"""Lexical cues aligned with ANNOTATION_METHODOLOGY (precedence + borderline neutral).

Used only by the meta LR layer — does not update DistilBERT.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Nationality / origin / migration axis (methodology §2.2 precedence when dominant)
_NATIONALITY_CUES = re.compile(
    r"\b(?:immigr|emigrant|border|asylum|deport|visa|"
    r"sponsorship|nationalit|citizen|foreign|abroad|overseas|"
    r"country\s+of\s+origin|ethnic|integration|assimilat|"
    r"send\s+them\s+back|they\s+don'?t\s+integrat|native[-\s]?born|"
    r"migrant\s+workers?|guest\s+workers?)\b",
    re.I,
)

# Profession / role hierarchy stereotypes (methodology §2.3)
_PROFESSION_CUES = re.compile(
    r"\b(?:vocational|trade\s*school|apprentices?|blue[-\s]?collar|white[-\s]?collar|"
    r"MBA|pedigree|not\s+leadership|management\s+material|executive\s+presence|"
    r"people\s+person|detail[-\s]?oriented|big[-\s]?picture|"
    r"real\s+work|proper\s+degree|technical\s+track|nurses?|teachers?|caregivers?|"
    r"hire\s+for\s+culture|culture\s+fit|shortlist)\b",
    re.I,
)

# Gender axis (explicit markers — complements DistilBERT type logits)
_GENDER_CUES = re.compile(
    r"\b(?:women|men|woman|man|female|male|mother|father|"
    r"daughter|son|girls?|guys?|maternity|paternity|she|her|he|him)\b",
    re.I,
)

# Pure counter-speech / debunking without endorsement (methodology → often neutral)
_COUNTER_DEBUNK = re.compile(
    r"\b(?:myth|stereotype|strawman|trope|false\s+claim|wrong\s+to\s+say|"
    r"not\s+(?:true|accurate)|has\s+nothing\s+to\s+do|push\s+back\s+on)\b",
    re.I,
)

# Demeaning evaluation lexicon (overlap with rule_signals — compact scalar)
_SLUR_STRENGTH = re.compile(
    r"\b(?:lazy|inferior|stealing\s+jobs?|unreliable|burden|threat|"
    r"dangerous|incompetent|keep\s+out|don'?t\s+belong)\b",
    re.I,
)

# Civil / evidence-forward policy framing (often neutral per methodology)
_POLICY_EVIDENCE = re.compile(
    r"\b(?:policy|reform|evidence|data|study|statistic|research|"
    r"report\s+shows|labour\s+market|economic|demographic)\b",
    re.I,
)


@dataclass(frozen=True)
class MethodologyFeatureBundle:
    nationality_cue: float
    profession_cue: float
    gender_axis_cue: float
    counter_speech_strength: float
    nationality_minus_profession: float
    civil_policy_framing: float


def extract_methodology_features(clean_text: str) -> MethodologyFeatureBundle:
    """Return six [0,1] scalars for meta fusion."""
    t = (clean_text or "").strip()
    if not t:
        return MethodologyFeatureBundle(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    n_hits = len(_NATIONALITY_CUES.findall(t))
    p_hits = len(_PROFESSION_CUES.findall(t))
    g_hits = len(_GENDER_CUES.findall(t))
    max_span = max(len(t) / 120.0, 1.0)

    nat = min(1.0, n_hits / max_span)
    prof = min(1.0, p_hits / max_span)
    gen = min(1.0, g_hits / max_span)

    counter_hits = len(_COUNTER_DEBUNK.findall(t))
    counter = min(1.0, counter_hits / 2.0)

    diff = nat - prof
    nat_minus_prof = max(0.0, min(1.0, (diff + 1.0) / 2.0))

    slur = 1.0 if _SLUR_STRENGTH.search(t) else 0.0
    pol = 1.0 if _POLICY_EVIDENCE.search(t) else 0.0
    # Policy/statistics-forward posts without demeaning lexicon → often neutral in methodology
    civil = pol * (1.0 - slur) * (1.0 - 0.35 * counter)

    return MethodologyFeatureBundle(
        nationality_cue=nat,
        profession_cue=prof,
        gender_axis_cue=gen,
        counter_speech_strength=counter,
        nationality_minus_profession=nat_minus_prof,
        civil_policy_framing=min(1.0, civil),
    )
