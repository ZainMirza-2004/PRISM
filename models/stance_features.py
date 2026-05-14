"""Stance decomposition features to separate group mention from harmful attribution."""

from __future__ import annotations

import re
from dataclasses import dataclass

_GROUP_TERMS = re.compile(
    r"\b(?:women|men|woman|man|female|male|immigrants?|foreigners?|locals?|people|workers?|"
    r"employees?|engineers?|managers?|staff|hires?|candidates?)\b",
    re.I,
)
_NEG_WORDS = re.compile(
    r"\b(?:lazy|inferior|harmful|dangerous|problem|burden|threat|stealing\s+jobs?|unreliable|incompetent)\b",
    re.I,
)
_NORMATIVE = re.compile(r"\b(?:should|must|need\s+to|ought\s+to|have\s+to)\b", re.I)
_NEGATION = re.compile(r"\b(?:not|never|no|n't)\b", re.I)
_DENIAL = re.compile(r"\b(?:are|is)\s+not\s+\w+|\bnot\s+(?:stealing\s+jobs?|inferior|the\s+problem)\b", re.I)
_CRITIQUE = re.compile(r"\b(?:myth|stereotype|strawman|trope|false\s+claim|wrong\s+to\s+say)\b", re.I)
_ENDORSE = re.compile(r"\b(?:better\s+than|more\s+suited|less\s+suited|prefer|not\s+a\s+fit|keep\s+out)\b", re.I)
_ESSENTIALIST = re.compile(r"\b(?:naturally|inherently|by\s+nature|always|never|usually)\b", re.I)


@dataclass(frozen=True)
class StanceFeatures:
    group_target_present: float
    sentiment_toward_group: float
    attribution_assertion: float
    attribution_denial: float
    attribution_critique_of_stereotype: float
    attribution_endorsement: float
    normative_language_score: float
    negation_scope_over_group: float
    essentialist_claim_score: float


def extract_stance_features(clean_text: str) -> StanceFeatures:
    t = (clean_text or "").strip()
    if not t:
        return StanceFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    g = 1.0 if _GROUP_TERMS.search(t) else 0.0
    neg_hits = len(_NEG_WORDS.findall(t))
    sentiment = min(1.0, neg_hits / 2.0) if g else 0.0
    denial = 1.0 if _DENIAL.search(t) else 0.0
    critique = 1.0 if _CRITIQUE.search(t) else 0.0
    endorsement = 1.0 if _ENDORSE.search(t) else 0.0
    normative = min(1.0, len(_NORMATIVE.findall(t)) / 2.0)
    neg_scope = 1.0 if (g and _NEGATION.search(t)) else 0.0
    essentialist = 1.0 if _ESSENTIALIST.search(t) else 0.0
    assertion = 1.0 if (g and not denial and not critique) else 0.0

    return StanceFeatures(
        group_target_present=g,
        sentiment_toward_group=sentiment,
        attribution_assertion=assertion,
        attribution_denial=denial,
        attribution_critique_of_stereotype=critique,
        attribution_endorsement=endorsement,
        normative_language_score=normative,
        negation_scope_over_group=neg_scope,
        essentialist_claim_score=essentialist,
    )
