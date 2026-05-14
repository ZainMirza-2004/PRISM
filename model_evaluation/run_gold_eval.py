"""Run hybrid pipeline on a labelled gold set; write metrics, confusion matrices, error analysis."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Run as script from PRISM root
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.loaders import load_labeled_goldset
from model_evaluation.metrics import (
    compute_binary_metrics,
    compute_multiclass_metrics,
    gold_labels_to_list,
    predictions_to_labels,
    save_report,
)
from models.hybrid_pipeline import HybridBiasPipeline


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    p = Path(__file__).resolve().parents[1] / ".env"
    if p.is_file():
        load_dotenv(p)


def build_errors(
    rows: list[dict], results: list[dict], y_gold: list[str], y_pred: list[str]
) -> list[dict]:
    err: list[dict] = []
    for i, (g, p) in enumerate(zip(y_gold, y_pred)):
        if g != p:
            err.append(
                {
                    "post_id": rows[i].get("post_id", str(i)),
                    "gold_label": g,
                    "predicted_bias_type": results[i].get("bias_type"),
                    "bias_detected": results[i].get("bias_detected"),
                    "confidence": results[i].get("confidence"),
                    "text": rows[i].get("text", ""),
                    "explanation": results[i].get("explanation"),
                }
            )
    return err


def run_evaluation(
    gold: str,
    model: str = "models/distilbert_B_balanced",
    out: str = "data/output/eval_report",
    batch_size: int = 4,
    hf_token: str | None = None,
    hate_model: str = "cardiffnlp/twitter-roberta-base-hate-latest",
    *,
    meta_classifier_path: str | None = None,
    auxiliary_type_model_dir: str | None = None,
    legacy_fusion: bool = False,
    calibrated: bool = False,
    use_thresholds: bool = False,
    thresholds_path: str | None = None,
) -> dict:
    _load_dotenv()
    token = hf_token or os.environ.get("HF_API_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    rows = load_labeled_goldset(gold)
    y_true = gold_labels_to_list(rows, key="label")

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
    # Do not pass gold labels into the model — only post_id + text (and optional non-gold fields).
    rows_for_pred: list[dict] = []
    for r in rows:
        d: dict = {
            "post_id": str(r.get("post_id", "")),
            "text": str(r.get("text", "")),
        }
        for k in ("slice_type", "ambiguous_stance"):
            if k in r and r[k] is not None:
                d[k] = r[k]
        rows_for_pred.append(d)
    results = pipe.predict_batch(rows_for_pred, batch_size=batch_size)
    y_pred = predictions_to_labels(results, use_detected=True)

    mc = compute_multiclass_metrics(y_true, y_pred)
    bi = compute_binary_metrics(y_true, y_pred)
    errors = build_errors(rows, results, y_true, y_pred)

    out_p = Path(out)
    save_report(out_p, mc, bi, errors)
    (out_p / "full_predictions.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    summary = {
        "multiclass": {k: mc[k] for k in ("n", "accuracy", "f1_macro", "f1_weighted") if k in mc},
        "binary": {k: bi[k] for k in ("n", "accuracy", "f1_bias_vs_neutral") if k in bi},
        "errors_written": len(errors),
        "out": str(out_p),
    }
    return {"metrics_multiclass": mc, "metrics_binary": bi, "summary": summary, "n_errors": len(errors)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate hybrid bias pipeline on a gold-labelled set")
    ap.add_argument("--gold", required=True, help="CSV or JSON with columns text, label (and optional post_id)")
    ap.add_argument("--model", default="models/distilbert_B_balanced", help="Fine-tuned type head directory")
    ap.add_argument("--out", default="data/output/eval_report", help="Directory for metrics + errors")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--hf-token", default=os.environ.get("HF_API_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN"))
    ap.add_argument("--hate-model", default="cardiffnlp/twitter-roberta-base-hate-latest")
    ap.add_argument("--meta-model", default=None, help="Meta-classifier joblib (see train-meta)")
    ap.add_argument(
        "--aux-type-model",
        default=None,
        help="Optional second DistilBERT head (e.g. models/distilbert_social_bias) for 40-d meta fusion",
    )
    ap.add_argument("--legacy-fusion", action="store_true", help="Disable learned meta-classifier")
    ap.add_argument("--calibrated", action="store_true", help="Use models/meta_fusion/lr_meta_calibrated.joblib when present")
    ap.add_argument("--use-thresholds", action="store_true")
    ap.add_argument("--thresholds", default=None, help="optimal_thresholds.json path")
    args = ap.parse_args()
    rep = run_evaluation(
        gold=args.gold,
        model=args.model,
        out=args.out,
        batch_size=args.batch_size,
        hf_token=args.hf_token,
        hate_model=args.hate_model,
        meta_classifier_path=getattr(args, "meta_model", None),
        auxiliary_type_model_dir=getattr(args, "aux_type_model", None),
        legacy_fusion=getattr(args, "legacy_fusion", False),
        calibrated=getattr(args, "calibrated", False),
        use_thresholds=getattr(args, "use_thresholds", False),
        thresholds_path=getattr(args, "thresholds", None),
    )
    print(json.dumps(rep["summary"], indent=2))


if __name__ == "__main__":
    main()
