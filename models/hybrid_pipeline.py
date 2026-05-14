"""Hybrid bias detection: DistilBERT type head + rule structure + Cardiff RoBERTa hate (intent)."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from explainability.fusion_explanations import (
    build_fusion_explanation,
    important_words_for_fusion,
)
from models.bias_type_head import BiasTypeHead
from models.fusion_engine import (
    fuse_scores,
    fusion_distribution_snapshot,
    intent_from_p_hate,
    interpret_distilbert,
)
from models.hf_hate_client import HateApiResult, classify_hate, classify_hate_batch, MODEL_ID as DEFAULT_HATE_ID
from models.meta_classifier import MetaFusionClassifier, resolve_hybrid_meta_path
from models.preprocess import preprocess_social_post
from models.linguistic_features import compute_linguistic_features
from models.rule_signals import RuleFusionSignals, extract_rule_fusion_signals, structure_score_from_rules
from models.stance_features import extract_stance_features

LOGGER = logging.getLogger(__name__)


class HybridBiasPipeline:
    """Multi-signal fusion pipeline (parallel DistilBERT + RoBERTa; rules on clean text)."""

    def __init__(
        self,
        type_model_dir: str,
        *,
        hf_token: Optional[str] = None,
        hate_model_id: Optional[str] = None,
        parallel_signals: bool = True,
        meta_classifier_path: Optional[str] = None,
        auxiliary_type_model_dir: Optional[str] = None,
        # PRISM eval defaults: slightly sharper primary + softer auxiliary ⇒ richer disagreement for dual meta.
        primary_type_temperature: float = 1.18,
        auxiliary_type_temperature: float = 1.42,
        calibrated: bool = False,
        thresholds_path: Optional[str] = None,
        use_thresholds: bool = False,
    ):
        # Slightly sharper primary + softer aux ⇒ meta sees a wider, more informative disagreement surface.
        self.type_head = BiasTypeHead(
            type_model_dir, calm_temperature=float(primary_type_temperature)
        )
        self.hf_token = hf_token
        self.hate_model_id = hate_model_id or DEFAULT_HATE_ID
        self.parallel_signals = parallel_signals
        effective_meta = meta_classifier_path
        if effective_meta is None:
            effective_meta = resolve_hybrid_meta_path(
                None, type_model_dir, auxiliary_type_model_dir
            )
        self.meta_fusion = MetaFusionClassifier(
            effective_meta,
            calibrated=calibrated,
            thresholds_path=thresholds_path,
            use_thresholds=use_thresholds,
        )
        md = getattr(self.meta_fusion, "_meta_input_dim", None)
        self.aux_type_head: Optional[BiasTypeHead] = None
        if md == 40:
            if not auxiliary_type_model_dir:
                raise ValueError(
                    "The selected meta fusion model expects 40 inputs (primary + auxiliary type heads). "
                    "Pass auxiliary_type_model_dir='models/distilbert_social_bias' (or --aux-type-model)."
                )
            self.aux_type_head = BiasTypeHead(
                auxiliary_type_model_dir, calm_temperature=float(auxiliary_type_temperature)
            )
        elif auxiliary_type_model_dir:
            LOGGER.warning(
                "Auxiliary type head %s was requested but the loaded meta expects %s features; "
                "aux head is skipped. Use train-meta --aux-type-model and a matching dual joblib.",
                auxiliary_type_model_dir,
                md or 36,
            )
        self.hate_disabled = str(self.hate_model_id).strip().lower() in {"none", "off", "disabled", "no-roberta"}

    @staticmethod
    def _fallback_hate_result() -> HateApiResult:
        return HateApiResult(
            label="NOT_HATE_DISABLED",
            label_norm="NOT_HATE",
            scores={"HATE": 0.0, "NOT_HATE": 1.0},
            p_hate=0.0,
        )

    def _run_parallel_signals(self, clean_text: str) -> tuple[Dict[str, float], HateApiResult]:
        if self.hate_disabled:
            dist0 = self.type_head.predict_type_distribution(clean_text)
            return dist0, self._fallback_hate_result()
        if self.parallel_signals:
            with ThreadPoolExecutor(max_workers=2) as ex:
                fut_dist = ex.submit(self.type_head.predict_type_distribution, clean_text)
                fut_hate = ex.submit(classify_hate, clean_text, model_id=self.hate_model_id, token=self.hf_token)
                dist0 = fut_dist.result()
                hr = fut_hate.result()
            return dist0, hr
        dist0 = self.type_head.predict_type_distribution(clean_text)
        hr = classify_hate(clean_text, model_id=self.hate_model_id, token=self.hf_token)
        return dist0, hr

    def finalize_from_signals(
        self,
        post_id: str,
        raw_text: str,
        clean_text: str,
        dist0: Dict[str, float],
        hr: HateApiResult,
        rules: RuleFusionSignals,
        dist_social0: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        p_hate = float(hr.p_hate)
        dist = fusion_distribution_snapshot(dist0)
        dist_social = (
            fusion_distribution_snapshot(dist_social0) if dist_social0 is not None else None
        )

        bias_detected, bias_type, confidence, fusion_trace = self.meta_fusion.predict(
            dist,
            rules,
            p_hate,
            clean_text=clean_text,
            use_legacy_fallback=True,
            dist_social=dist_social,
        )

        iw = important_words_for_fusion(
            bias_detected,
            clean_text,
            bias_type,
            self.type_head,
            rules,
            p_hate,
            top_k=5,
        )
        expl = build_fusion_explanation(
            bias_detected,
            rules,
            p_hate,
            bias_type,
            dist,
            fusion_trace=fusion_trace,
        )

        # Display text: preserve user-facing original if non-empty
        display_text = (raw_text or "").strip() if (raw_text or "").strip() else clean_text

        out: Dict[str, Any] = {
            "post_id": str(post_id),
            "text": display_text,
            "bias_detected": bias_detected,
            "bias_type": bias_type,
            "confidence": float(round(confidence, 3)),
            "important_words": iw,
            "explanation": expl,
        }
        if fusion_trace.get("posterior"):
            out["meta_fusion"] = {
                "mode": fusion_trace.get("mode"),
                "posterior": fusion_trace["posterior"],
                "posterior_calibrated": fusion_trace.get("posterior_calibrated"),
                "predicted_class": fusion_trace.get("predicted_class"),
                "features": fusion_trace.get("features"),
                "use_thresholds": fusion_trace.get("use_thresholds"),
            }
        if dist_social is not None:
            out["auxiliary_type_distribution"] = dist_social
        return out

    def finalize(
        self,
        post_id: str,
        text: str,
        dist0: Dict[str, float],
        hr: HateApiResult,
        *,
        clean_text: Optional[str] = None,
        rules: Optional[RuleFusionSignals] = None,
    ) -> Dict[str, Any]:
        raw_text = text or ""
        ct = clean_text if clean_text is not None else preprocess_social_post(raw_text)
        r = rules if rules is not None else extract_rule_fusion_signals(ct)
        return self.finalize_from_signals(post_id, raw_text, ct, dist0, hr, r)

    def predict_one(self, post_id: str, text: str) -> Dict[str, Any]:
        raw_text = text or ""
        clean_text = preprocess_social_post(raw_text)
        dist0, hr = self._run_parallel_signals(clean_text)
        rules = extract_rule_fusion_signals(clean_text)
        ds0 = None
        if self.aux_type_head is not None:
            ds0 = self.aux_type_head.predict_type_distribution(clean_text)
        return self.finalize_from_signals(
            str(post_id), raw_text, clean_text, dist0, hr, rules, dist_social0=ds0
        )

    def predict_batch(self, rows: List[Dict[str, str]], batch_size: int = 8) -> List[Dict[str, Any]]:
        if not rows:
            return []
        ids = [str(r.get("post_id", "")) for r in rows]
        raw_texts = [str(r.get("text", "")) for r in rows]
        cleans = [preprocess_social_post(t) for t in raw_texts]

        dists = self.type_head.predict_type_distribution_batch(cleans, batch_size=batch_size)
        dists_social: Optional[List[Dict[str, float]]] = None
        if self.aux_type_head is not None:
            dists_social = self.aux_type_head.predict_type_distribution_batch(
                cleans, batch_size=batch_size
            )
        if self.hate_disabled:
            hates = [self._fallback_hate_result() for _ in cleans]
        else:
            hates = classify_hate_batch(cleans, model_id=self.hate_model_id, token=self.hf_token)

        out: List[Dict[str, Any]] = []
        for i, (pid, rw, ct, d0, hr) in enumerate(zip(ids, raw_texts, cleans, dists, hates)):
            rules = extract_rule_fusion_signals(ct)
            ds0 = dists_social[i] if dists_social else None
            out.append(self.finalize_from_signals(pid, rw, ct, d0, hr, rules, dist_social0=ds0))
        return out


def analyze_single_post_body(body: Dict[str, Any], pipeline: HybridBiasPipeline, post_id: str = "") -> Dict[str, Any]:
    """Accept input shape ``{'text': '<social media post>'}`` (optional ``post_id``)."""
    text = str(body.get("text", "") or "")
    pid = str(body.get("post_id") or post_id or "")
    return pipeline.predict_one(pid, text)


def extract_debug_signals(post_id: str, text: str, pipeline: HybridBiasPipeline) -> Dict[str, Any]:
    """Structured multi-signal view for debugging (sections 2–4 of the spec)."""
    raw = text or ""
    clean_text = preprocess_social_post(raw)
    dist0, hr = pipeline._run_parallel_signals(clean_text)
    rules = extract_rule_fusion_signals(clean_text)
    ling = compute_linguistic_features(clean_text, rules)
    stance = extract_stance_features(clean_text)

    dist = fusion_distribution_snapshot(dist0)
    bs, cand = interpret_distilbert(dist)
    struct = structure_score_from_rules(rules)
    intent = intent_from_p_hate(float(hr.p_hate))
    meta = getattr(pipeline, "meta_fusion", None)
    if meta and meta.is_loaded:
        ds0 = None
        if getattr(pipeline, "aux_type_head", None) is not None:
            ds0 = pipeline.aux_type_head.predict_type_distribution(clean_text)
        dist_s = fusion_distribution_snapshot(ds0) if ds0 is not None else None
        _bd, _bt, _c, fusion_trace = meta.predict(
            dist,
            rules,
            float(hr.p_hate),
            clean_text=clean_text,
            use_legacy_fallback=True,
            dist_social=dist_s,
        )
        trace = fusion_trace
    else:
        _bd, _bt, _c, _bf, trace = fuse_scores(dist, rules, float(hr.p_hate), clean_text)

    return {
        "post_id": str(post_id),
        "clean_text": clean_text,
        "distilbert_distribution": dist,
        "rule_layer": rules.as_dict(),
        "linguistic_features": {
            "group_presence": float(ling.group_presence),
            "soft_preference_norm": float(ling.soft_preference_norm),
            "hedging_rate": float(ling.hedging_rate),
            "polarity_gap": float(ling.polarity_gap),
            "implicit_x_group": float(ling.implicit_x_group),
            "target_negative_sentiment": float(ling.target_negative_sentiment),
            "exclusion_intent": float(ling.exclusion_intent),
            "anti_stereotype_cue": float(ling.anti_stereotype_cue),
        },
        "stance_features": {
            "group_target_present": float(stance.group_target_present),
            "sentiment_toward_group": float(stance.sentiment_toward_group),
            "attribution_assertion": float(stance.attribution_assertion),
            "attribution_denial": float(stance.attribution_denial),
            "attribution_critique_of_stereotype": float(stance.attribution_critique_of_stereotype),
            "attribution_endorsement": float(stance.attribution_endorsement),
            "normative_language_score": float(stance.normative_language_score),
            "negation_scope_over_group": float(stance.negation_scope_over_group),
            "essentialist_claim_score": float(stance.essentialist_claim_score),
        },
        "roberta_intent": {"p_hate": float(hr.p_hate)},
        "interpretation": {
            "bias_strength": bs,
            "bias_type_candidate": cand,
            "structure_score": struct,
            "intent_label": intent,
        },
        "fusion_trace": trace,
    }
