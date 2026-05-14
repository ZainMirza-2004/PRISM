"""Fixed-order feature vector for the learned fusion (meta-classifier) layer."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from models.fusion_engine import fusion_distribution_snapshot
from models.label_config import LABELS
from models.linguistic_features import LinguisticFeatureBundle, compute_linguistic_features
from models.methodology_features import extract_methodology_features
from models.rule_signals import RuleFusionSignals, structure_score_from_rules
from models.stance_features import extract_stance_features

# Bump when META_FEATURE_NAMES length or order changes (requires retraining meta fusion joblib).
# v6 appends auxiliary hate-speech DistilBERT type-head probs (same 4 labels) for dual-head fusion.
FEATURE_SCHEMA_VERSION: int = 6

# Order is stable for training & inference (append-only within a schema version).
META_FEATURE_NAMES_CORE: List[str] = [
    "distilbert_gender_bias",
    "distilbert_nationality_bias",
    "distilbert_profession_bias",
    "distilbert_neutral",
    "p_hate_roberta",
    "rule_generalisation",
    "rule_comparison",
    "rule_preference",
    "rule_coded_bias",
    "structure_score",
    "rule_soft_preference",
    "rule_implicit_generalisation",
    "rule_inequality_context",
    "lex_group_presence",
    "lex_soft_preference_norm",
    "lex_hedging_rate",
    "lex_polarity_gap",
    "lex_implicit_x_group",
    "lex_target_negative_sentiment",
    "lex_exclusion_intent",
    "lex_anti_stereotype_cue",
    "stance_group_target_present",
    "stance_sentiment_toward_group",
    "stance_attribution_assertion",
    "stance_attribution_denial",
    "stance_attribution_critique",
    "stance_attribution_endorsement",
    "stance_normative_language_score",
    "stance_negation_scope_over_group",
    "stance_essentialist_claim_score",
    "meth_nationality_cue",
    "meth_profession_cue",
    "meth_gender_axis_cue",
    "meth_counter_speech_strength",
    "meth_nationality_minus_profession",
    "meth_civil_policy_framing",
]

META_FEATURE_NAMES_SOCIAL_AUX: List[str] = [
    "social_type_gender_bias",
    "social_type_nationality_bias",
    "social_type_profession_bias",
    "social_type_neutral",
]

META_FEATURE_NAMES: List[str] = META_FEATURE_NAMES_CORE + META_FEATURE_NAMES_SOCIAL_AUX


def build_meta_feature_row(
    dist: Dict[str, float],
    p_hate: float,
    rules: RuleFusionSignals,
    *,
    clean_text: str | None = None,
    linguistic: LinguisticFeatureBundle | None = None,
    structure_score: float | None = None,
    dist_social: Dict[str, float] | None = None,
) -> np.ndarray:
    """36-dim row without *dist_social*; 40-dim with auxiliary hate-speech type head (schema v6)."""
    struct = float(structure_score) if structure_score is not None else structure_score_from_rules(rules)
    rd = rules.as_dict()
    ling = linguistic if linguistic is not None else compute_linguistic_features(clean_text or "", rules)
    stance = extract_stance_features(clean_text or "")
    meth = extract_methodology_features(clean_text or "")

    row = [
        float(dist.get("gender_bias", 0.0)),
        float(dist.get("nationality_bias", 0.0)),
        float(dist.get("profession_bias", 0.0)),
        float(dist.get("neutral", 0.0)),
        float(p_hate),
        float(rd.get("generalisation", 0)),
        float(rd.get("comparison", 0)),
        float(rd.get("preference", 0)),
        float(rd.get("coded_bias", 0)),
        struct,
        float(rd.get("soft_preference", 0)),
        float(rd.get("implicit_generalisation", 0)),
        float(rd.get("inequality_context", 0)),
        float(ling.group_presence),
        float(ling.soft_preference_norm),
        float(ling.hedging_rate),
        float(ling.polarity_gap),
        float(ling.implicit_x_group),
        float(ling.target_negative_sentiment),
        float(ling.exclusion_intent),
        float(ling.anti_stereotype_cue),
        float(stance.group_target_present),
        float(stance.sentiment_toward_group),
        float(stance.attribution_assertion),
        float(stance.attribution_denial),
        float(stance.attribution_critique_of_stereotype),
        float(stance.attribution_endorsement),
        float(stance.normative_language_score),
        float(stance.negation_scope_over_group),
        float(stance.essentialist_claim_score),
        float(meth.nationality_cue),
        float(meth.profession_cue),
        float(meth.gender_axis_cue),
        float(meth.counter_speech_strength),
        float(meth.nationality_minus_profession),
        float(meth.civil_policy_framing),
    ]
    if len(row) != len(META_FEATURE_NAMES_CORE):
        raise ValueError("Core feature dimension mismatch.")
    if dist_social is not None:
        snap_s = fusion_distribution_snapshot(dist_social)
        row.extend(float(snap_s.get(k, 0.0)) for k in LABELS)
    if dist_social is None and len(row) != 36:
        raise ValueError("Expected 36 core features without dist_social.")
    if dist_social is not None and len(row) != 40:
        raise ValueError("Expected 40 features with dist_social.")
    return np.asarray(row, dtype=np.float64).reshape(1, -1)


def build_meta_feature_matrix(rows: List[np.ndarray]) -> np.ndarray:
    if not rows:
        return np.zeros((0, 0), dtype=np.float64)
    return np.vstack(rows)
