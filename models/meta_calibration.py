"""Platt (sigmoid) per-class probability calibration for the meta fusion classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, MutableMapping, Sequence

import joblib
import numpy as np
from scipy.special import logit as scipy_logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

_CALIBRATION_VERSION = "platt_ovr_v1"


def _clamp_prob(p: float, eps: float = 1e-6) -> float:
    return float(min(1.0 - eps, max(eps, p)))


@dataclass
class OvRCalibrator:
    """One-vs-rest Platt scaler or per-class isotonic regression on raw multiclass probs."""

    classes: tuple[str, ...]
    method: Literal["platt", "isotonic"]
    calibrators: Dict[str, LogisticRegression | IsotonicRegression]

    def predict_per_class_probs(self, per_class_probs: Mapping[str, float]) -> Dict[str, float]:
        logits = []
        vals = []
        ordered = []
        for c in self.classes:
            p_raw = float(per_class_probs.get(str(c), 0.0))
            logits.append(float(scipy_logit(_clamp_prob(p_raw))))
            ordered.append(str(c))

        calibrated: MutableMapping[str, float] = {}
        for c, lx in zip(ordered, logits):
            clf = self.calibrators[c]
            if isinstance(clf, LogisticRegression):
                q = clf.predict_proba(np.asarray([[lx]], dtype=np.float64))[0, 1]
            else:
                # isotonic predicts P(y | raw prob), use raw probability on trained domain
                p_in = np.clip(float(per_class_probs.get(str(c), 0.0)), 0.001, 0.999)
                q = float(clf.predict([p_in])[0])
            calibrated[c] = max(1e-9, float(q))

        s = sum(calibrated.values())
        out = {k: float(calibrated[k] / s) for k in calibrated}
        return out


def fit_ovr_calibration(
    y_true: Sequence[str],
    rows_per_class_probs: Sequence[MutableMapping[str, float]],
    classes_ordered: Sequence[str],
    *,
    method: Literal["platt", "isotonic"] = "platt",
    random_state: int = 42,
) -> OvRCalibrator:
    """Fit OvR calibration from validation multiclass probs (sums to ~1).

    Args:
      y_true: gold labels aligned with rows_per_class_probs
      rows_per_class_probs: each element dict[class] = raw softmax prob from LR
    """

    ys = np.asarray(list(y_true), dtype=object)
    classes = tuple(classes_ordered)

    calibrators: Dict[str, LogisticRegression | IsotonicRegression] = {}
    rng = np.random.RandomState(random_state)

    if method == "platt":
        for c in classes:
            logits = []
            y_bin = (ys == c).astype(np.int32)
            for row in rows_per_class_probs:
                pk = float(row.get(str(c), 0.0))
                logits.append([float(scipy_logit(_clamp_prob(pk)))])
            lr = LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
                class_weight=None,
                random_state=int(rng.randint(1_000_000)),
            )
            X = np.asarray(logits, dtype=np.float64)
            if np.std(X.ravel()) < 1e-9:
                X[:, 0] = X[:, 0] + rng.normal(scale=1e-4, size=X.shape[0])
            lr.fit(X, y_bin)
            calibrators[c] = lr

    elif method == "isotonic":
        for c in classes:
            raw_probs = []
            y_bin = (ys == c).astype(np.float64)
            for row in rows_per_class_probs:
                raw_probs.append(float(row.get(str(c), 0.0)))
            x = np.clip(np.asarray(raw_probs, dtype=np.float64), 0.0, 1.0)
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(x, y_bin)
            calibrators[c] = iso

    else:
        raise ValueError(f"Unsupported method={method}")

    return OvRCalibrator(classes=classes, method=method, calibrators=calibrators)


def build_calibrated_payload(
    base_payload: Mapping[str, Any],
    *,
    calibration: OvRCalibrator,
) -> Dict[str, Any]:
    out = dict(base_payload)
    out["pipeline"] = base_payload.get("pipeline") or base_payload.get("model")
    out["feature_names"] = base_payload.get("feature_names")
    out["label_order"] = list(base_payload.get("label_order") or calibration.classes)
    out["feature_schema_version"] = base_payload.get("feature_schema_version")
    out["meta_calibration_version"] = _CALIBRATION_VERSION
    out["calibration_ovr"] = calibration
    return out


def save_calibrated_artifact(
    base_joblib_path: str,
    out_path: str,
    calibration_rows: Sequence[tuple[str, MutableMapping[str, float]]],
    *,
    classes_ordered: Sequence[str],
    method: Literal["platt", "isotonic"] = "platt",
) -> str:
    """Load base meta joblib (dict with pipeline); fit OvR calibration; save merged artifact."""

    base = joblib.load(base_joblib_path)
    if isinstance(base, dict):
        bp = dict(base)
    else:
        raise ValueError("Expected meta classifier saved as dict with pipeline/key metadata.")

    y_true = [lab for lab, _ in calibration_rows]
    rows_probs = [dict(rw) for _, rw in calibration_rows]
    calibrator = fit_ovr_calibration(y_true, rows_probs, classes_ordered=classes_ordered, method=method)
    merged = build_calibrated_payload(bp, calibration=calibrator)
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(merged, str(outp))
    return str(outp)


def apply_calibration_to_trace(
    raw_posterior: Mapping[str, float],
    calibration: OvRCalibrator | None,
    classes_ordered: Sequence[str],
) -> Dict[str, float]:
    if calibration is None:
        return dict(raw_posterior)
    return calibration.predict_per_class_probs(raw_posterior)


# ---- script glue (minimal path import) ----


def paths_default(root: Path | None = None) -> tuple[Path, Path]:
    r = Path(__file__).resolve().parents[1] if root is None else Path(root)
    return r / "models" / "meta_fusion" / "lr_meta_stance.joblib", r / "models" / "meta_fusion" / "lr_meta_calibrated.joblib"
