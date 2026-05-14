"""Build calibration_dataset.json by stratified split + meta pipeline raw posteriors."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from sklearn.model_selection import train_test_split

from data.loaders import load_labeled_goldset
from models.hybrid_pipeline import HybridBiasPipeline
from models.label_config import LABELS


def collect_calibration_records(
    gold_path: str,
    type_model: str,
    meta_classifier_path: str | None,
    *,
    hate_model: str,
    hf_token: str | None,
    batch_size: int,
    test_size: float,
    random_state: int,
    auxiliary_type_model_dir: str | None = None,
) -> dict:
    rows = load_labeled_goldset(gold_path)
    labels = [str(r.get("label", "")).strip() for r in rows]
    idx_all = np.arange(len(rows))
    try:
        idx_train, idx_val = train_test_split(
            idx_all,
            test_size=test_size,
            stratify=np.asarray(labels),
            random_state=random_state,
        )
    except ValueError:
        idx_train, idx_val = train_test_split(idx_all, test_size=test_size, random_state=random_state)
    val_rows = [rows[int(i)] for i in sorted(idx_val)]
    pipe = HybridBiasPipeline(
        type_model,
        hf_token=hf_token or os.environ.get("HF_API_TOKEN"),
        hate_model_id=hate_model,
        meta_classifier_path=meta_classifier_path or None,
        auxiliary_type_model_dir=auxiliary_type_model_dir,
        calibrated=False,
        use_thresholds=False,
    )
    results = pipe.predict_batch(val_rows, batch_size=batch_size)
    calibration_rows: list[dict] = []
    for vr, rr in zip(val_rows, results):
        tg = str(vr.get("label", "")).strip()
        mf = rr.get("meta_fusion") or {}
        post = mf.get("posterior")
        if post is None:
            continue
        calibration_rows.append(
            {
                "post_id": str(vr.get("post_id", "")),
                "text": str(vr.get("text", "")),
                "true_label": tg,
                "posterior_raw": dict(post),
            }
        )

    stats = {"n_validation": len(calibration_rows), "classes": list(LABELS)}
    payload = {"meta": stats, "rows": calibration_rows}
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Export calibration_dataset.json with raw meta posteriors on a held-out slice")
    ap.add_argument("--gold", required=True, help="Labeled CSV/JSON path")
    ap.add_argument("--model", default="models/distilbert_social_bias", help="Fine-tuned type head dir")
    ap.add_argument("--meta-model", default="models/meta_fusion/lr_meta_stance.joblib")
    ap.add_argument("--out", default="data/output/calibration/calibration_dataset.json")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--hate-model", default="cardiffnlp/twitter-roberta-base-hate-latest")
    ap.add_argument("--hf-token", default=os.environ.get("HF_API_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN"))
    ap.add_argument(
        "--aux-type-model",
        default=None,
        help="Must match dual meta (40-d): e.g. models/distilbert_social_bias",
    )
    args = ap.parse_args()

    aux = args.aux_type_model
    aux_p = (
        str(_ROOT / aux) if aux and not Path(aux).is_absolute() else aux
    )

    out_p = _ROOT / args.out
    out_p.parent.mkdir(parents=True, exist_ok=True)
    payload = collect_calibration_records(
        str(_ROOT / args.gold) if not Path(args.gold).is_absolute() else args.gold,
        str(_ROOT / args.model) if not Path(args.model).is_absolute() else args.model,
        str(_ROOT / args.meta_model) if not Path(args.meta_model).is_absolute() else args.meta_model,
        hate_model=args.hate_model,
        hf_token=args.hf_token,
        batch_size=args.batch_size,
        test_size=args.test_size,
        random_state=args.random_state,
        auxiliary_type_model_dir=aux_p,
    )
    out_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": str(out_p), "n_rows": len(payload["rows"])}, indent=2))


if __name__ == "__main__":
    main()
