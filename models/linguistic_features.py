"""Continuous, interpretable linguistic probes for meta-classifier (Option 2 bundle)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from models.rule_signals import RuleFusionSignals

# Target / group-ish presence (normalized token hit rate)
_GROUP_LEX = re.compile(
    r"\b(?:women|men|woman|man|female|male|she|her|him|he|they|them|"
    r"girls?|guys?|mother|father|daughter|son|girl|boy|"
    r"immigrants?|immigration|locals?|foreign(?:ers)?|nationality|citizen|visa|"
    r"candidates?|employees?|workers?|engineers?|developers?|managers?|leaders?|"
    r"hires?|people|staff|team|applicant|coworkers?)\b",
    re.I,
)

_HEDGE_MODAL = re.compile(
    r"\b(?:might|may|could|would|should|seems?|appear(?:s)?|perhaps|maybe|arguably|"
    r"generally|typically|often|usually|probably|possibly|potentially|rather|quite)\b",
    re.I,
)

_SOFT_PREF_HIT = re.compile(
    r"\b(?:gravitate|lean\s+towards?|stronger\s+fit|smoother|less\s+friction|"
    r"better\s+dynamic|natural\s+fit|executive\s+presence|culture\s+fit|polish)\b",
    re.I,
)

_POS = frozenset(
    "good great excellent strong positive better best smoother easier effective solid confident".split()
)
_NEG = frozenset(
    "bad worse poor weak negative terrible awful difficult harsh risky problematic harmful".split()
)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", (text or "").lower())


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


@dataclass(frozen=True)
class LinguisticFeatureBundle:
    group_presence: float
    soft_preference_norm: float
    hedging_rate: float
    polarity_gap: float
    implicit_x_group: float
    target_negative_sentiment: float
    exclusion_intent: float
    anti_stereotype_cue: float


def compute_linguistic_features(clean_text: str, rules: RuleFusionSignals) -> LinguisticFeatureBundle:
    """Stable scalar features aligned with preprocessing (same string as rule extraction)."""
    t = (clean_text or "").strip()
    if not t:
        return LinguisticFeatureBundle(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    tokens = _tokenize(t)
    n = max(len(tokens), 1)

    g_hits = len(_GROUP_LEX.findall(t))
    group_presence = _clamp01(g_hits / max(12.0, float(n) * 0.25))

    sf_hits = len(_SOFT_PREF_HIT.findall(t))
    soft_pref_norm = _clamp01(sf_hits / max(8.0, float(n) * 0.2))

    hedge_words = sum(
        1
        for w in tokens
        if w
        in {
            "might",
            "may",
            "could",
            "would",
            "should",
            "seems",
            "perhaps",
            "maybe",
            "generally",
            "typically",
            "often",
            "usually",
            "probably",
            "possibly",
        }
    )
    phrase_hits = len(_HEDGE_MODAL.findall(t))
    hedging_rate = _clamp01(min((hedge_words + phrase_hits) / max(10.0, float(n)), 1.0))

    pos_c = sum(1 for w in tokens if w in _POS)
    neg_c = sum(1 for w in tokens if w in _NEG)
    polarity = (pos_c - neg_c) / float(n)
    structural = max(
        float(rules.generalisation),
        float(rules.comparison),
        float(rules.preference),
        float(rules.soft_preference),
        float(rules.coded_bias),
    )
    polarity_gap = _clamp01(max(0.0, polarity) * structural)

    ixg = float(rules.implicit_generalisation) * group_presence

    exclusion_patterns = (
        r"\b(?:should\s+not\s+hire|don't\s+trust|do\s+not\s+trust|avoid\s+hiring|not\s+a\s+fit|keep\s+out)\b"
    )
    anti_stereo_patterns = (
        r"\b(?:not\s+inferior|not\s+the\s+problem|not\s+stealing\s+jobs|myth\s+that|stereotype\s+that|reject\s+the\s+stereotype)\b"
    )
    exclusion_intent = 1.0 if re.search(exclusion_patterns, t, re.I) else 0.0
    anti_stereotype_cue = 1.0 if re.search(anti_stereo_patterns, t, re.I) else 0.0
    target_negative_sentiment = _clamp01((neg_c / float(n)) * group_presence)

    return LinguisticFeatureBundle(
        group_presence=group_presence,
        soft_preference_norm=soft_pref_norm,
        hedging_rate=hedging_rate,
        polarity_gap=polarity_gap,
        implicit_x_group=_clamp01(ixg),
        target_negative_sentiment=target_negative_sentiment,
        exclusion_intent=exclusion_intent,
        anti_stereotype_cue=anti_stereotype_cue,
    )


def bundle_as_vector(b: LinguisticFeatureBundle) -> tuple[float, float, float, float, float, float, float, float]:
    return (
        b.group_presence,
        b.soft_preference_norm,
        b.hedging_rate,
        b.polarity_gap,
        b.implicit_x_group,
        b.target_negative_sentiment,
        b.exclusion_intent,
        b.anti_stereotype_cue,
    )
