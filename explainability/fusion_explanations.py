"""Structured explanation lines for the multi-signal fusion layer."""

from __future__ import annotations

from typing import Any, Dict, List, Union

from models.label_config import NEUTRAL_LABEL
from models.rule_signals import RuleFusionSignals


def build_fusion_explanation(
    bias_detected: Union[bool, str],
    rules: RuleFusionSignals,
    p_hate: float,
    bias_type: str,
    dist: Dict[str, float],
    *,
    fusion_trace: Dict[str, Any] | None = None,
) -> List[str]:
    """Return human-readable explanation bullet strings (spec: list, not a single paragraph)."""
    if bias_detected is True:
        lines: List[str] = []
        if fusion_trace and fusion_trace.get("mode") == "meta_lr" and fusion_trace.get("posterior"):
            posterior = fusion_trace["posterior"]
            pred = fusion_trace.get("predicted_class", "")
            pp = float(posterior.get(str(pred), 0.0))
            lines.append(
                f"Learned fusion classifier (multinomial logistic regression) predicts '{str(pred).replace('_', ' ')}' "
                f"with posterior mass {pp:.2f}, combining DistilBERT, Cardiff RoBERTa (p_hate), and linguistic rule features."
            )
        if rules.generalisation:
            lines.append("Generalisation or group-level phrasing detected (e.g. 'X are …').")
        if rules.implicit_generalisation:
            lines.append(
                "Hedged tendency phrasing tied to a group or role proxy (implicit generalisation cue)."
            )
        if rules.comparison:
            lines.append("Comparative or ranking phrasing (e.g. 'better than' / performance claims).")
        if rules.preference:
            lines.append(
                "Hard preference or hiring-frame language (shortlist, locals-first, migration/hiring combos, role 'should')."
            )
        if rules.soft_preference:
            lines.append(
                "Soft evaluative or 'fit' language (smooth hire, polish, dynamics, executive presence)."
            )
        if rules.coded_bias:
            lines.append("Coded organisational euphemisms (culture fit, pedigree, low-risk hire, polish, etc.).")
        if not lines and (rules.inequality_context and p_hate < 0.35):
            lines.append("Discusses inequality or bias in context; model still points to a specific bias class here.")
        if not lines:
            lines.append("Semantic head indicates a group / role bias signal for this post.")
        # Model hint
        best = max(
            ("gender_bias", dist.get("gender_bias", 0.0)),
            ("nationality_bias", dist.get("nationality_bias", 0.0)),
            ("profession_bias", dist.get("profession_bias", 0.0)),
            key=lambda x: x[1],
        )
        if best[0] == bias_type and best[1] > 0.2:
            human = bias_type.replace("_", " ")
            lines.append(f"Model distribution is strongest for {human} (core semantic signal).")
        if p_hate > 0.75:
            lines.append("Hostile or highly charged tone (RoBERTa intent calibration) increases confidence in harm potential.")
        elif p_hate < 0.2 and lines:
            lines.append("Low hate-probability: primary signal is group/role stereotyping, not general toxicity.")
        return _dedupe_preserve_order(lines)[:6]

    if bias_detected == "uncertain":
        return [
            "Score fell in the border band (0.4–0.6): combined model and structure evidence is mixed.",
            "Treat as low confidence; consider context, author intent, and follow-up moderation review.",
        ]

    # Not detected (includes meta-classifier predicting neutral)
    out: List[str] = []
    if fusion_trace and fusion_trace.get("mode") == "meta_lr" and fusion_trace.get("posterior"):
        posterior = fusion_trace["posterior"]
        pn = float(posterior.get("neutral", 0.0))
        out.append(
            f"Learned fusion classifier assigns highest probability to neutral/non-bias (posterior neutral={pn:.2f}), "
            "combining DistilBERT logits, RoBERTa p_hate, and structural rule features."
        )
    else:
        out.extend(
            [
                "No strong generalisation, comparison, or preference pattern met the fusion threshold.",
                "Low combined score from the DistilBERT bias head and the linguistic structure layer.",
            ]
        )
    top_bias = max(
        ("gender_bias", dist.get("gender_bias", 0.0)),
        ("nationality_bias", dist.get("nationality_bias", 0.0)),
        ("profession_bias", dist.get("profession_bias", 0.0)),
        key=lambda x: x[1],
    )
    if top_bias[1] >= 0.35:
        out.append(f"Bias-class probability was non-trivial ({top_bias[0].replace('_', ' ')}) but below the fused decision cutoff.")
    if rules.inequality_context and p_hate < 0.35:
        out.append("Reads partly like inequality or bias-in-society discussion rather than endorsing a stereotype.")
    return out


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for x in items:
        k = x.strip()
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out


def important_words_for_fusion(
    bias_detected: Union[bool, str],
    clean_text: str,
    bias_type: str,
    type_head,
    rules: RuleFusionSignals,
    p_hate: float,
    *,
    top_k: int = 5,
) -> List[str]:
    """Important tokens: attributions when bias_detected True; otherwise empty list per spec."""
    from explainability.keywords import build_important_words
    from models.interpretation_signals import analyze_linguistic_signals

    if bias_detected is not True or bias_type == NEUTRAL_LABEL:
        return []
    heur = analyze_linguistic_signals(clean_text)
    return build_important_words(
        clean_text,
        bias_type,
        heur,
        p_hate,
        type_head,
        use_integrated_attribution=True,
        top_k=top_k,
    )
