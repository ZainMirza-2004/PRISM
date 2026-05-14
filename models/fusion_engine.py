"""Multi-signal fusion: DistilBERT bias strength + rule structure + RoBERTa intent."""

from __future__ import annotations

from typing import Any, Dict, Tuple, Union

from models.critique_lexicon import contains_negation_or_critique
from models.label_config import LABELS, NEUTRAL_LABEL
from models.rule_signals import RuleFusionSignals, structure_score_from_rules


_BIAS_KEYS = ["gender_bias", "nationality_bias", "profession_bias"]


def interpret_distilbert(dist: Dict[str, float]) -> Tuple[float, str]:
    """bias_strength = max(gender, nationality, profession); candidate = argmax over those three."""
    d = {k: float(dist.get(k, 0.0)) for k in LABELS}
    bias_strength = max(d[k] for k in _BIAS_KEYS)
    bias_type_candidate = max(_BIAS_KEYS, key=lambda k: d[k])
    return bias_strength, bias_type_candidate


def intent_from_p_hate(p_hate: float) -> str:
    if p_hate > 0.75:
        return "hostile"
    if p_hate < 0.2:
        return "non_hostile"
    return "uncertain"


def fuse_scores(
    dist: Dict[str, float],
    rules: RuleFusionSignals,
    p_hate: float,
    original_clean_text: str,
) -> Tuple[
    Union[bool, str],
    str,
    float,
    float,
    Dict[str, Any],
]:
    """
    Returns:
      bias_detected (True | False | 'uncertain'),
      bias_type (winner or neutral),
      confidence in [0.5, 0.95],
      base_score final (after adjustments),
      trace dict for debugging/explanations.
    """
    bias_strength, bias_type_candidate = interpret_distilbert(dist)
    structure_score = structure_score_from_rules(rules)

    base_score = 0.65 * bias_strength + 0.35 * structure_score

    intent = intent_from_p_hate(p_hate)

    if intent == "hostile":
        base_score += 0.15
    elif intent == "non_hostile" and structure_score == 0.0:
        base_score -= 0.20

    safeguard_applied = False
    if structure_score > 0.0 and p_hate < 0.2 and contains_negation_or_critique(original_clean_text):
        base_score -= 0.25
        safeguard_applied = True

    base_score = max(0.0, min(1.0, base_score))

    if base_score >= 0.6:
        bias_detected: Union[bool, str] = True
    elif base_score <= 0.4:
        bias_detected = False
    else:
        bias_detected = "uncertain"

    if bias_detected is True:
        bias_type = bias_type_candidate
    else:
        bias_type = NEUTRAL_LABEL  # False or "uncertain"

    confidence = max(0.5, min(0.95, base_score))

    trace = {
        "bias_strength": bias_strength,
        "bias_type_candidate": bias_type_candidate,
        "structure_score": structure_score,
        "intent": intent,
        "p_hate": p_hate,
        "rule_flags": rules.as_dict(),
        "safeguard_applied": safeguard_applied,
        "base_score_final": base_score,
    }
    return bias_detected, bias_type, confidence, base_score, trace


def fusion_distribution_snapshot(dist: Dict[str, float]) -> Dict[str, float]:
    """Normalized 4-class snapshot for API/debug."""
    out = {k: float(dist.get(k, 0.0)) for k in LABELS}
    s = sum(max(0.0, v) for v in out.values())
    if s <= 0:
        return {k: 1.0 / len(LABELS) for k in LABELS}
    return {k: max(0.0, out[k]) / s for k in LABELS}
