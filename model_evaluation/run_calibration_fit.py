"""Fit OvR Platt from calibration_dataset.json; persist lr_meta_calibrated.joblib + optimal_thresholds.json."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.loaders import load_labeled_goldset
from models.hybrid_pipeline import HybridBiasPipeline
from models.label_config import LABELS
from models.meta_calibration import save_calibrated_artifact
from models.threshold_optimizer import optimize_thresholds, save_thresholds_json


def extract_posterior_cal(rr: dict) -> dict[str, float]:
    mf = rr.get("meta_fusion") or {}
    pc = mf.get("posterior_calibrated")
    raw = mf.get("posterior")
    use = pc or raw
    return dict(use) if isinstance(use, dict) else {}


def collect_and_fit_calibration(
    calibration_json_path: str,
    base_meta_path: str,
    out_calibrated_path: str,
    out_thresholds_path: str,
    sanity_csv_path: str,
    type_model_path: str,
    *,
    batch_size: int = 8,
    hate_model: str = "cardiffnlp/twitter-roberta-base-hate-latest",
    hf_token: str | None = None,
    auxiliary_type_model_dir: str | None = None,
) -> dict[str, str]:
    raw = Path(calibration_json_path).read_text(encoding="utf-8")
    data = json.loads(raw)
    rows_data = data.get("rows", data)

    rows_tuples: list[tuple[str, dict[str, float]]] = []
    for r in rows_data:
        pr = r.get("posterior_raw")
        tl = str(r.get("true_label", "")).strip()
        if not pr:
            continue
        rows_tuples.append((tl, dict(pr)))

    base_p = Path(base_meta_path) if Path(base_meta_path).is_absolute() else (_ROOT / base_meta_path)
    out_cal = Path(out_calibrated_path) if Path(out_calibrated_path).is_absolute() else (_ROOT / out_calibrated_path)
    outp_thr = Path(out_thresholds_path) if Path(out_thresholds_path).is_absolute() else (_ROOT / out_thresholds_path)

    save_calibrated_artifact(str(base_p), str(out_cal), rows_tuples, classes_ordered=LABELS, method="platt")

    type_pt = Path(type_model_path) if Path(type_model_path).is_absolute() else (_ROOT / type_model_path)
    sane = Path(sanity_csv_path) if Path(sanity_csv_path).is_absolute() else (_ROOT / sanity_csv_path)

    tk = hf_token or os.environ.get("HF_API_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")

    pipe = HybridBiasPipeline(
        str(type_pt),
        hf_token=tk,
        hate_model_id=hate_model,
        meta_classifier_path=str(out_cal),
        auxiliary_type_model_dir=auxiliary_type_model_dir,
        calibrated=True,
        use_thresholds=False,
    )

    val_rows: list[dict] = []
    for r in rows_data:
        if not r.get("posterior_raw"):
            continue
        val_rows.append(
            {
                "post_id": str(r.get("post_id", "")),
                "text": str(r.get("text", "")),
                "label": str(r["true_label"]),
            }
        )
    val_results = pipe.predict_batch(val_rows, batch_size=batch_size)
    val_post = [extract_posterior_cal(x) for x in val_results]
    y_val_aligned = [str(r.get("label", "")).strip() for r in val_rows]

    sanity_rows = load_labeled_goldset(str(sane))
    sanity_res = pipe.predict_batch(sanity_rows, batch_size=batch_size)
    sanity_post = [extract_posterior_cal(x) for x in sanity_res]

    th, meta = optimize_thresholds(
        y_val_true=y_val_aligned,
        calibrated_val_posteriors=val_post,
        sanity_rows=sanity_rows,
        calibrated_sanity_posteriors=sanity_post,
        classes_ordered=LABELS,
    )
    save_thresholds_json(str(outp_thr), th, meta=meta)
    return {
        "calibrated_artifact": str(out_cal),
        "thresholds": str(outp_thr),
        "meta": meta,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration-dataset", default="data/output/calibration/calibration_dataset.json")
    ap.add_argument("--base-meta", default="models/meta_fusion/lr_meta_stance.joblib")
    ap.add_argument("--out-calibrated", default="models/meta_fusion/lr_meta_calibrated.joblib")
    ap.add_argument("--out-thresholds", default="models/meta_fusion/optimal_thresholds.json")
    ap.add_argument("--sanity", default="data/meta_training/sanity_slice.csv")
    ap.add_argument("--model", default="models/distilbert_social_bias")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--hate-model", default="cardiffnlp/twitter-roberta-base-hate-latest")
    ap.add_argument("--hf-token", default=os.environ.get("HF_API_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN"))
    ap.add_argument(
        "--aux-type-model",
        default=None,
        help="Dual-head calibration: e.g. models/distilbert_social_bias",
    )
    args = ap.parse_args()
    aux = args.aux_type_model
    aux_p = str(_ROOT / aux) if aux and not Path(aux).is_absolute() else aux
    out = collect_and_fit_calibration(
        str(_ROOT / args.calibration_dataset)
        if not Path(args.calibration_dataset).is_absolute()
        else args.calibration_dataset,
        args.base_meta,
        args.out_calibrated,
        args.out_thresholds,
        args.sanity,
        args.model,
        batch_size=args.batch_size,
        hate_model=args.hate_model,
        hf_token=args.hf_token,
        auxiliary_type_model_dir=aux_p,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
