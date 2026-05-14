"""Classification metrics and confusion matrix helpers (gold label vs system output)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from models.label_config import LABELS, NEUTRAL_LABEL


def gold_labels_to_list(rows: List[Dict[str, Any]], key: str = "label") -> List[str]:
    return [str(r.get(key, "")).strip() for r in rows]


def predictions_to_labels(results: List[Dict[str, Any]], use_detected: bool = True) -> List[str]:
    out: List[str] = []
    for r in results:
        b = r.get("bias_type", NEUTRAL_LABEL)
        if use_detected:
            bd = r.get("bias_detected")
            # Only explicit True counts as a bias detection (False or "uncertain" → neutral).
            if bd is not True:
                b = NEUTRAL_LABEL
        out.append(str(b).strip() if b else NEUTRAL_LABEL)
    return out


def binary_bias_any(y: List[str]) -> List[int]:
    return [0 if (x or "").lower() == NEUTRAL_LABEL else 1 for x in y]


def compute_multiclass_metrics(
    y_true: List[str], y_pred: List[str], *, labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    lab = labels or list(LABELS)
    y_t = [t if t in lab else "unknown" for t in y_true]
    y_p = [p if p in lab else "unknown" for p in y_pred]
    # Filter pairs where gold invalid
    pairs = [(t, p) for t, p in zip(y_t, y_p) if t in lab]
    if not pairs:
        return {
            "accuracy": 0.0,
            "n": 0,
            "per_class": {},
            "confusion_matrix": [],
            "labels_order": lab,
        }
    yt, yp = zip(*pairs)
    acc = float(accuracy_score(yt, yp))
    p_w, r_w, f1_w, sup_w = precision_recall_fscore_support(yt, yp, average="weighted", zero_division=0, labels=lab)
    p_m, r_m, f1_m, sup_m = precision_recall_fscore_support(yt, yp, average="macro", zero_division=0, labels=lab)
    cm = confusion_matrix(yt, yp, labels=lab)
    report = classification_report(yt, yp, labels=lab, output_dict=True, zero_division=0)

    return {
        "n": len(pairs),
        "accuracy": acc,
        "precision_weighted": float(p_w),
        "recall_weighted": float(r_w),
        "f1_weighted": float(f1_w),
        "precision_macro": float(p_m),
        "recall_macro": float(r_m),
        "f1_macro": float(f1_m),
        "per_class": report,
        "confusion_matrix": cm.tolist(),
        "labels_order": lab,
    }


def compute_binary_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    a = np.array(binary_bias_any(y_true))
    b = np.array(binary_bias_any(y_pred))
    if len(a) == 0:
        return {"n": 0}
    p, r, f1, _ = precision_recall_fscore_support(a, b, average="binary", pos_label=1, zero_division=0)
    acc = float(accuracy_score(a, b))
    cm2 = confusion_matrix(a, b, labels=[0, 1])
    return {
        "n": len(a),
        "accuracy": acc,
        "precision_bias_vs_neutral": float(p),
        "recall_bias_vs_neutral": float(r),
        "f1_bias_vs_neutral": float(f1),
        "confusion_matrix_binary": cm2.tolist(),
        "binary_cm_rows_cols": "rows: gold 0=neutral,1=bias; cols: pred 0,1 (see sklearn.confusion_matrix)",
    }


def save_report(out_dir: Path, multiclass: Dict[str, Any], binary: Dict[str, Any], errors: List[Dict[str, Any]]) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics_multiclass.json").write_text(json.dumps(multiclass, indent=2), encoding="utf-8")
    (out_dir / "metrics_binary.json").write_text(json.dumps(binary, indent=2), encoding="utf-8")
    (out_dir / "errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")

    # Human-readable CM
    lines = [f"Multiclass confusion matrix (rows=gold, cols=pred). Labels: {multiclass.get('labels_order', [])}\n"]
    for row in multiclass.get("confusion_matrix", []):
        lines.append(" ".join(str(int(x)) for x in row))
    lines.append("\nBinary (0=neutral, 1=any bias):\n")
    for row in binary.get("confusion_matrix_binary", []):
        lines.append(" ".join(str(int(x)) for x in row))
    (out_dir / "confusion_matrices.txt").write_text("\n".join(lines), encoding="utf-8")
