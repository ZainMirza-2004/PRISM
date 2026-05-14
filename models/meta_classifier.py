"""Learned fusion layer: logistic regression over DistilBERT + RoBERTa + rule features."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import joblib
import numpy as np

from models.feature_vector import (
    FEATURE_SCHEMA_VERSION,
    META_FEATURE_NAMES,
    META_FEATURE_NAMES_CORE,
    build_meta_feature_row,
)
from models.fusion_engine import fuse_scores
from models.label_config import LABELS, NEUTRAL_LABEL
from models.meta_calibration import OvRCalibrator, apply_calibration_to_trace
from models.rule_signals import RuleFusionSignals
from models.threshold_optimizer import load_threshold_config, predicted_class_with_thresholds


def resolve_meta_classifier_path(explicit: Optional[str] = None) -> Optional[str]:
    """Prefer explicit path, then META_FUSION_MODEL env, then default joblib if present."""
    if explicit:
        p = Path(explicit)
        return str(p.resolve()) if p.is_file() else None
    env = (os.environ.get("META_FUSION_MODEL") or "").strip()
    if env:
        ep = Path(env)
        return str(ep.resolve()) if ep.is_file() else None
    root = Path(__file__).resolve().parents[1]
    v2_human = root / "models" / "meta_fusion" / "lr_meta_v2_human.joblib"
    stance = root / "models" / "meta_fusion" / "lr_meta_stance.joblib"
    legacy = root / "models" / "meta_fusion" / "lr_meta.joblib"
    if v2_human.is_file():
        return str(v2_human.resolve())
    if stance.is_file():
        return str(stance.resolve())
    if legacy.is_file():
        return str(legacy.resolve())
    return None


def resolve_hybrid_meta_path(
    explicit: Optional[str],
    primary_type_dir: str,
    auxiliary_type_dir: Optional[str],
) -> Optional[str]:
    """Meta joblib for hybrid runs: explicit > env > dual (40-d) > mono (36-d) > legacy v2_human."""
    if explicit:
        p = Path(explicit)
        return str(p.resolve()) if p.is_file() else None
    env = (os.environ.get("META_FUSION_MODEL") or "").strip()
    if env:
        ep = Path(env)
        return str(ep.resolve()) if ep.is_file() else None
    root = Path(__file__).resolve().parents[1]
    prim = Path(primary_type_dir).name
    if auxiliary_type_dir:
        aux_name = Path(auxiliary_type_dir).name
        # Prefer dissertation final dual meta (v3 100-train gold slice) when shipped alongside base dual.
        dual_v3 = root / "models" / "meta_fusion" / f"lr_meta_dual_{prim}_plus_{aux_name}_v3_100train.joblib"
        if dual_v3.is_file():
            return str(dual_v3.resolve())
        dual = root / "models" / "meta_fusion" / f"lr_meta_dual_{prim}_plus_{aux_name}.joblib"
        if dual.is_file():
            return str(dual.resolve())
    mono = root / "models" / "meta_fusion" / f"lr_meta_mono_{prim}.joblib"
    if mono.is_file():
        return str(mono.resolve())
    return resolve_meta_classifier_path(None)


_ROOT = Path(__file__).resolve().parents[1]


class MetaFusionClassifier:
    """Wrapper: sklearn Pipeline (StandardScaler + learned multiclass meta model), 4-way output.

    Optional env ``META_BLEND_PRIMARY_DIST`` in ``[0, 0.5]`` mixes meta posterior with the primary
    type-head distribution before calibration/thresholds (helps noisy profession↔neutral margins).
    Optional env ``META_BLEND_AUX_DIST`` in ``[0, 0.7]`` blends in the auxiliary social-bias type
    head when present (40-d dual-head meta), allowing explicit social-priority tuning.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        *,
        calibrated: bool = False,
        thresholds_path: Optional[str] = None,
        use_thresholds: bool = False,
    ):
        if model_path == "":
            self.model_path = None
            self._pipeline = None
            self._feature_names = None
            self._meta_input_dim: Optional[int] = None
            self._calibration: Optional[OvRCalibrator] = None
            self._thresholds: Optional[Dict[str, float]] = None
            self._ambiguity_margin = None
            self.use_thresholds = False
            self._label_order_classes = tuple(LABELS)
            self._feature_schema_version: Optional[int] = None
            return

        self.use_thresholds = bool(use_thresholds)
        self._thresholds = None
        self._ambiguity_margin: Optional[float] = None

        calibrated_path = _ROOT / "models" / "meta_fusion" / "lr_meta_calibrated.joblib"
        explicit_resolved = resolve_meta_classifier_path(model_path)
        stance_default = _ROOT / "models" / "meta_fusion" / "lr_meta_stance.joblib"

        if calibrated:
            self.model_path = str(calibrated_path) if calibrated_path.is_file() else (
                explicit_resolved or (str(stance_default) if stance_default.is_file() else "")
            )
        else:
            self.model_path = explicit_resolved or (str(stance_default) if stance_default.is_file() else "")
            if self.model_path and not Path(self.model_path).is_file():
                self.model_path = explicit_resolved or ""

        default_thr = _ROOT / "models" / "meta_fusion" / "optimal_thresholds.json"
        if self.use_thresholds:
            thr_p = thresholds_path or (str(default_thr) if default_thr.is_file() else None)
            if thr_p and Path(thr_p).is_file():
                self._thresholds, self._ambiguity_margin = load_threshold_config(thr_p)

        self._pipeline = None
        self._feature_names: Optional[list[str]] = None
        self._feature_schema_version: Optional[int] = None
        self._meta_input_dim: Optional[int] = None
        self._calibration: Optional[OvRCalibrator] = None
        self._label_order_classes: Tuple[str, ...] = tuple(LABELS)

        ml = self.model_path
        if ml and Path(ml).is_file():
            raw = joblib.load(ml)
            if isinstance(raw, dict):
                fn = raw.get("feature_names")
                self._feature_schema_version = raw.get("feature_schema_version")
                self._pipeline = raw.get("pipeline")
                scaler = None
                if self._pipeline is not None and hasattr(self._pipeline, "named_steps"):
                    scaler = self._pipeline.named_steps.get("scaler")
                if scaler is not None and hasattr(scaler, "n_features_in_"):
                    self._meta_input_dim = int(scaler.n_features_in_)
                elif fn is not None:
                    self._meta_input_dim = len(fn)
                else:
                    self._meta_input_dim = len(META_FEATURE_NAMES_CORE)
                # Trace labels: prefer artifact list when it matches scaler width (handles v5 vs v6).
                if fn is not None and len(fn) == self._meta_input_dim:
                    self._feature_names = list(map(str, fn))
                elif self._meta_input_dim == len(META_FEATURE_NAMES):
                    self._feature_names = list(META_FEATURE_NAMES)
                else:
                    self._feature_names = list(META_FEATURE_NAMES_CORE)
                lo = raw.get("label_order")
                if lo:
                    self._label_order_classes = tuple(map(str, lo))
                self._calibration = raw.get("calibration_ovr")
            else:
                self._pipeline = raw
                if self._pipeline is not None and hasattr(self._pipeline, "named_steps"):
                    scaler = self._pipeline.named_steps.get("scaler")
                    if scaler is not None and hasattr(scaler, "n_features_in_"):
                        self._meta_input_dim = int(scaler.n_features_in_)

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def predict(
        self,
        dist: Dict[str, float],
        rules: RuleFusionSignals,
        p_hate: float,
        *,
        clean_text: str,
        use_legacy_fallback: bool = True,
        dist_social: Optional[Dict[str, float]] = None,
    ) -> Tuple[Union[bool, str], str, float, Dict[str, Any]]:
        dim = self._meta_input_dim if self._meta_input_dim is not None else len(META_FEATURE_NAMES_CORE)
        if self.is_loaded and dim == 40 and dist_social is None:
            raise ValueError(
                "Loaded meta fusion expects 40 features (dual-head). "
                "Pass dist_social from the auxiliary type head, or use a 36-feature meta joblib."
            )
        ds = dist_social if dim == 40 else None
        X = build_meta_feature_row(dist, p_hate, rules, clean_text=clean_text or "", dist_social=ds)
        feat_names = self._feature_names or (
            list(META_FEATURE_NAMES) if dim == len(META_FEATURE_NAMES) else list(META_FEATURE_NAMES_CORE)
        )
        trace: Dict[str, Any] = {
            "mode": "meta_lr" if self.is_loaded else "legacy_rules",
            "features": dict(zip(feat_names, X.ravel().tolist())),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "artifact_schema_version": self._feature_schema_version,
            "meta_input_dim": dim,
            "use_thresholds": self.use_thresholds,
            "p_hate_used": float(p_hate),
        }

        if self.is_loaded and self._pipeline is not None:
            proba = self._pipeline.predict_proba(X)[0]
            classes = np.asarray(self._pipeline.classes_)
            ordered_classes = tuple(str(c) for c in classes.tolist())
            posterior_raw = {str(ca): float(p) for ca, p in zip(classes, proba)}
            # Optional: re-mix meta posterior with primary and auxiliary type-head distributions.
            # This is useful when we want to prioritize social-bias cues while retaining the
            # stronger benchmark head signal in dual (40-d) fusion.
            default_primary = "0.10" if dim == 40 and dist_social is not None else "0"
            default_aux = "0.35" if dim == 40 and dist_social is not None else "0"
            blend_primary = float(os.environ.get("META_BLEND_PRIMARY_DIST", default_primary).strip() or "0")
            blend_primary = max(0.0, min(0.5, blend_primary))
            blend_aux = float(os.environ.get("META_BLEND_AUX_DIST", default_aux).strip() or "0")
            blend_aux = max(0.0, min(0.7, blend_aux))
            total_blend = blend_primary + blend_aux
            if total_blend > 0:
                labs_ord = self._label_order_classes
                dv = np.asarray([float(dist.get(str(c), 0.0)) for c in labs_ord], dtype=np.float64)
                s_d = float(dv.sum())
                if s_d > 0:
                    dv = dv / s_d
                else:
                    dv = np.ones(len(labs_ord), dtype=np.float64) / max(len(labs_ord), 1)
                sv = None
                if dist_social is not None:
                    sv = np.asarray([float(dist_social.get(str(c), 0.0)) for c in labs_ord], dtype=np.float64)
                    s_s = float(sv.sum())
                    if s_s > 0:
                        sv = sv / s_s
                    else:
                        sv = np.ones(len(labs_ord), dtype=np.float64) / max(len(labs_ord), 1)
                pv = np.asarray([float(posterior_raw.get(str(c), 0.0)) for c in labs_ord], dtype=np.float64)
                s_p = float(pv.sum())
                pv = pv / max(s_p, 1e-12)
                keep = max(0.0, 1.0 - total_blend)
                mx = keep * pv + blend_primary * dv
                if sv is not None:
                    mx = mx + blend_aux * sv
                s_m = float(mx.sum())
                mx = mx / max(s_m, 1e-12)
                posterior_raw = {str(labs_ord[i]): float(mx[i]) for i in range(len(labs_ord))}
                trace["meta_blend_primary_dist"] = blend_primary
                trace["meta_blend_aux_dist"] = blend_aux

            posterior_cal = posterior_raw
            if self._calibration is not None:
                posterior_cal = apply_calibration_to_trace(posterior_raw, self._calibration, ordered_classes)

            trace["posterior"] = posterior_raw
            trace["posterior_calibrated"] = posterior_cal
            if self._ambiguity_margin is not None:
                trace["ambiguity_margin"] = float(self._ambiguity_margin)

            if self.use_thresholds and self._thresholds:
                pred_label = predicted_class_with_thresholds(
                    posterior_cal,
                    self._thresholds,
                    classes_ordered=self._label_order_classes,
                    ambiguity_margin=self._ambiguity_margin,
                )
                conf_raw = float(posterior_cal.get(pred_label, 0.0))
            else:
                if self.use_thresholds:
                    trace["threshold_fallback"] = "use_thresholds set but thresholds file missing; using argmax on calibrated probs"
                cal_list = np.asarray([posterior_cal.get(str(c), 0.0) for c in self._label_order_classes])
                j = int(np.argmax(cal_list))
                pred_label = self._label_order_classes[j]
                conf_raw = float(posterior_cal.get(pred_label, 0.0))

            bias_detected = pred_label != NEUTRAL_LABEL
            bias_type = pred_label if bias_detected else NEUTRAL_LABEL
            conf = max(0.5, min(0.95, float(conf_raw)))
            trace["predicted_class"] = pred_label
            return bias_detected, bias_type, conf, trace

        if use_legacy_fallback:
            bd, bt, conf, _, leg = fuse_scores(dist, rules, p_hate, clean_text)
            trace["mode"] = "legacy_rules"
            trace["fusion_trace"] = leg
            return bd, bt, conf, trace

        raise RuntimeError("Meta fusion model not loaded and legacy fallback disabled.")
