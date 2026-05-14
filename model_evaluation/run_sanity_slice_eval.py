from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, MutableMapping

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.loaders import load_labeled_goldset
from model_evaluation.metrics import compute_multiclass_metrics, gold_labels_to_list, predictions_to_labels
from models.hybrid_pipeline import HybridBiasPipeline


def _aggregate_sanity_metrics(
    rows: list[MutableMapping[str, Any]],
    y_true: list[str],
    y_pred: list[str],
) -> dict[str, Any]:
    mc = compute_multiclass_metrics(y_true, y_pred)
    fp = fp_denom = fn = fn_denom = 0
    per_slice = {
        "neutral_policy": {"n": 0, "fp": 0},
        "positive_group": {"n": 0, "fp": 0},
        "explicit_or_implicit_bias": {"n": 0, "fn": 0},
    }
    stance_correct = stance_n = 0
    for row, pred in zip(rows, y_pred):
        st = str(row.get("slice_type", "")).strip()
        ambiguous = str(row.get("ambiguous_stance", "0")).strip().lower() in {"1", "true", "yes"}
        if ambiguous:
            continue
        per_slice.setdefault(st, {"n": 0})
        per_slice[st]["n"] = int(per_slice[st].get("n", 0)) + 1
        if st in {"neutral_policy", "positive_group"}:
            fp_denom += 1
            if pred != "neutral":
                fp += 1
                per_slice[st]["fp"] = int(per_slice[st].get("fp", 0)) + 1
            else:
                stance_correct += 1
            stance_n += 1
        elif st == "explicit_or_implicit_bias":
            fn_denom += 1
            if pred == "neutral":
                fn += 1
                per_slice[st]["fn"] = int(per_slice[st].get("fn", 0)) + 1
            else:
                stance_correct += 1
            stance_n += 1

    return {
        "overall": mc,
        "fpr_neutral_policy_positive_group": float(fp / fp_denom) if fp_denom else 0.0,
        "fnr_explicit_or_implicit_bias": float(fn / fn_denom) if fn_denom else 0.0,
        "counts": {"fp": fp, "fp_denom": fp_denom, "fn": fn, "fn_denom": fn_denom},
        "stance_aware_accuracy": float(stance_correct / stance_n) if stance_n else 0.0,
        "macro_f1": float(mc.get("f1_macro", 0.0)),
        "per_slice_rates": {
            "neutral_policy_fpr": (
                float(per_slice["neutral_policy"].get("fp", 0) / max(1, per_slice["neutral_policy"].get("n", 1)))
                if per_slice["neutral_policy"].get("n", 0)
                else 0.0
            ),
            "positive_group_fpr": (
                float(per_slice["positive_group"].get("fp", 0) / max(1, per_slice["positive_group"].get("n", 1)))
                if per_slice["positive_group"].get("n", 0)
                else 0.0
            ),
            "explicit_or_implicit_bias_fnr": (
                float(per_slice["explicit_or_implicit_bias"].get("fn", 0) / max(1, per_slice["explicit_or_implicit_bias"].get("n", 1)))
                if per_slice["explicit_or_implicit_bias"].get("n", 0)
                else 0.0
            ),
        },
    }


def run_sanity_eval(
    sanity_csv: str,
    model: str = "models/distilbert_B_balanced",
    out: str = "data/output/sanity_eval",
    batch_size: int = 4,
    hf_token: str | None = None,
    hate_model: str = "cardiffnlp/twitter-roberta-base-hate-latest",
    meta_classifier_path: str | None = None,
    auxiliary_type_model_dir: str | None = None,
    legacy_fusion: bool = False,
    *,
    calibrated: bool = False,
    use_thresholds: bool = False,
    thresholds_path: str | None = None,
    compare_calibration: bool = False,
) -> dict[str, Any]:
    token = hf_token or os.environ.get("HF_API_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    rows = load_labeled_goldset(sanity_csv)
    y_true = gold_labels_to_list(rows, key="label")

    def _run_pipe(
        *,
        cal: bool,
        use_thr: bool,
        thr_path: str | None,
    ) -> tuple[list[dict], list[str]]:
        meta_path = "" if legacy_fusion else meta_classifier_path
        pipe = HybridBiasPipeline(
            model,
            hf_token=token,
            hate_model_id=hate_model,
            meta_classifier_path=meta_path,
            auxiliary_type_model_dir=auxiliary_type_model_dir,
            calibrated=cal,
            use_thresholds=use_thr,
            thresholds_path=thr_path,
        )
        res = pipe.predict_batch(rows, batch_size=batch_size)
        yp = predictions_to_labels(res, use_detected=True)
        return res, yp

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if compare_calibration and not legacy_fusion:
        res_pre, yp_pre = _run_pipe(cal=False, use_thr=False, thr_path=None)
        m_pre = _aggregate_sanity_metrics(rows, y_true, yp_pre)
        res_post, yp_post = _run_pipe(cal=calibrated or True, use_thr=use_thresholds or True, thr_path=thresholds_path)
        m_post = _aggregate_sanity_metrics(rows, y_true, yp_post)
        payload: dict[str, Any] = {
            "pre_calibration_argmax": m_pre,
            "post_calibration_thresholded": m_post,
        }
        (out_dir / "sanity_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (out_dir / "sanity_predictions_pre.json").write_text(json.dumps(res_pre, indent=2), encoding="utf-8")
        (out_dir / "sanity_predictions_post.json").write_text(json.dumps(res_post, indent=2), encoding="utf-8")
        return payload

    meta_path = "" if legacy_fusion else meta_classifier_path
    pipe = HybridBiasPipeline(
        model,
        hf_token=token,
        hate_model_id=hate_model,
        meta_classifier_path=meta_path,
        auxiliary_type_model_dir=auxiliary_type_model_dir,
        calibrated=calibrated,
        use_thresholds=use_thresholds,
        thresholds_path=thresholds_path,
    )
    results = pipe.predict_batch(rows, batch_size=batch_size)
    y_pred = predictions_to_labels(results, use_detected=True)
    payload = _aggregate_sanity_metrics(rows, y_true, y_pred)
    (out_dir / "sanity_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "sanity_predictions.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate hard sanity slice with FPR/FNR diagnostics")
    ap.add_argument("--sanity", required=True, help="CSV with text,label,slice_type")
    ap.add_argument("--model", default="models/distilbert_B_balanced")
    ap.add_argument("--out", default="data/output/sanity_eval")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--hf-token", default=os.environ.get("HF_API_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN"))
    ap.add_argument("--hate-model", default="cardiffnlp/twitter-roberta-base-hate-latest")
    ap.add_argument("--meta-model", default=None)
    ap.add_argument("--aux-type-model", default=None)
    ap.add_argument("--legacy-fusion", action="store_true")
    ap.add_argument("--calibrated", action="store_true")
    ap.add_argument("--use-thresholds", action="store_true")
    ap.add_argument("--thresholds", default=None, help="Path to optimal_thresholds.json")
    ap.add_argument("--compare-calibration", action="store_true", help="Emit pre vs post metrics in one report")
    args = ap.parse_args()
    rep = run_sanity_eval(
        sanity_csv=args.sanity,
        model=args.model,
        out=args.out,
        batch_size=args.batch_size,
        hf_token=args.hf_token,
        hate_model=args.hate_model,
        meta_classifier_path=args.meta_model,
        auxiliary_type_model_dir=args.aux_type_model,
        legacy_fusion=args.legacy_fusion,
        calibrated=args.calibrated,
        use_thresholds=args.use_thresholds,
        thresholds_path=args.thresholds,
        compare_calibration=args.compare_calibration,
    )
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
