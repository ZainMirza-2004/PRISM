from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from pipeline.analyze import analyze_posts
from synthetic_data_generator import generate_data
from train.train_model import train
from utils.logging_config import configure_logging


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.is_file():
        load_dotenv(env_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Social bias detection (hybrid API + type head + light cues)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser("generate-data", help="Generate synthetic training dataset")
    gen.add_argument("--samples-per-class", type=int, default=320)
    gen.add_argument("--output", default="data/training/generated_social_bias_data.csv")

    tr = subparsers.add_parser(
        "train",
        help="Train the local 4-class bias *type* head (DistilBERT). Use --fuse for peer-reviewed binary + synthetic fusion.",
    )
    tr.add_argument(
        "--data",
        default="data/training/generated_social_bias_data.csv",
        help="Training CSV path (text,label with gender_bias|nationality_bias|profession_bias|neutral)",
    )
    tr.add_argument("--output", default="models/distilbert_social_bias")
    tr.add_argument("--epochs", type=int, default=5, help="Training epochs (3–5+ recommended)")
    tr.add_argument("--batch-size", type=int, default=16)
    tr.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splits, shuffling, and Trainer (StereoSet/CrowS CSVs use group_id splits).",
    )
    tr.add_argument(
        "--fuse",
        action="store_true",
        help="Build fused CSV from peer-reviewed binary (train_data) + synthetic, then train on fused output.",
    )
    tr.add_argument("--real-data", default="train_data", help="Binary CSV: Content,text + Label (0/1)")
    tr.add_argument("--synthetic-data", default="data/training/generated_social_bias_data.csv")
    tr.add_argument(
        "--fusion-output",
        default="data/training/fused_real_synthetic_balanced.csv",
        help="Where to write fused balanced CSV when using --fuse",
    )
    tr.add_argument("--fusion-seed", type=int, default=42)
    tr.add_argument(
        "--max-real",
        type=int,
        default=None,
        help="Randomly subsample at most this many real rows before fusion (speed/debug; omit for full corpus)",
    )

    an = subparsers.add_parser("analyze", help="Run hybrid pipeline (Cardiff API + local type head)")
    an.add_argument(
        "--dataset",
        default="data/input/sample_posts.json",
        help="Input CSV/JSON path (default: bundled sample posts)",
    )
    an.add_argument(
        "--model",
        default="models/distilbert_B_balanced",
        help="Primary fine-tuned 4-class type head (final PRISM stack uses distilbert_B_balanced)",
    )
    an.add_argument("--output", default="data/output/results.json")
    an.add_argument("--confidence-threshold", type=float, default=0.6, help="Legacy; hybrid uses the decision layer by default")
    an.add_argument("--batch-size", type=int, default=8)
    an.add_argument(
        "--hf-token",
        default=os.environ.get("HF_API_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
        help="Hugging Face token (or set HF_API_TOKEN) for the Inference API",
    )
    an.add_argument(
        "--hate-model",
        default="cardiffnlp/twitter-roberta-base-hate-latest",
        help="Hugging Face model id for the cloud hate/NOT-HATE base model",
    )
    an.add_argument(
        "--meta-model",
        default=None,
        help="Joblib meta-classifier (LR fusion). Default: META_FUSION_MODEL env or models/meta_fusion/lr_meta.joblib if present.",
    )
    an.add_argument(
        "--aux-type-model",
        default="models/distilbert_social_bias",
        help="Auxiliary type head for dual meta (default: distilbert_social_bias)",
    )
    an.add_argument(
        "--legacy-fusion",
        action="store_true",
        help="Disable learned meta-classifier; use hand-tuned fusion_engine rules only.",
    )
    an.add_argument("--calibrated", action="store_true", help="Load lr_meta_calibrated.joblib for meta layer")
    an.add_argument("--use-thresholds", action="store_true", help="Apply per-class thresholds from JSON")
    an.add_argument("--thresholds", default=None, help="Path to optimal_thresholds.json")

    tm = subparsers.add_parser(
        "train-meta",
        help="Train logistic-regression fusion on labeled gold (DistilBERT+RoBERTa+rules); use --extra-gold to merge v3-500 or other CSVs",
    )
    tm.add_argument("--gold", required=True, help="CSV/JSON with text + label columns")
    tm.add_argument(
        "--type-model",
        default="models/distilbert_B_balanced",
        help="Fine-tuned DistilBERT type head dir",
    )
    tm.add_argument("--output", default="models/meta_fusion/lr_meta.joblib")
    tm.add_argument("--hate-model", default="cardiffnlp/twitter-roberta-base-hate-latest")
    tm.add_argument(
        "--hf-token",
        default=os.environ.get("HF_API_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
    )
    tm.add_argument("--batch-size", type=int, default=8)
    tm.add_argument("--test-size", type=float, default=0.2)
    tm.add_argument("--random-state", type=int, default=42)
    tm.add_argument("--max-iter", type=int, default=500)
    tm.add_argument("--sanity-slice", default="data/meta_training/sanity_slice.csv")
    tm.add_argument("--neutral-weight", type=float, default=None)
    tm.add_argument(
        "--extra-gold",
        action="append",
        default=[],
        help=(
            "Extra labeled CSV/JSON merged after --gold (dedupe by post_id). "
            "Example: --extra-gold data/evaluation/manual_eval_v3_500_posts.csv for benchmark-aligned training."
        ),
    )
    tm.add_argument(
        "--extra-gold-max-rows",
        type=int,
        default=None,
        help="Optional cap on merged extra-gold rows (e.g. 100 from the 500-post set).",
    )
    tm.add_argument(
        "--no-refit-full",
        action="store_true",
        help="Do not refit the selected LR on the full merged gold (keep train-split fit only).",
    )
    tm.add_argument(
        "--aux-type-model",
        default=None,
        help="Second type head for 40-d fusion training (e.g. models/distilbert_social_bias).",
    )

    eva = subparsers.add_parser("evaluate", help="Gold-set metrics: P/R/F1, confusion matrices, error export (uses HF API)")
    eva.add_argument(
        "--gold",
        default="data/evaluation/manual_eval_v3_400_posts.csv",
        help="CSV/JSON with text + label (default: 400-post manual benchmark)",
    )
    eva.add_argument(
        "--model",
        default="models/distilbert_B_balanced",
        help="Primary type head (final PRISM stack)",
    )
    eva.add_argument("--out", default="data/output/eval_report", help="Output directory for reports")
    eva.add_argument("--batch-size", type=int, default=4)
    eva.add_argument(
        "--hf-token",
        default=os.environ.get("HF_API_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
        help="Hugging Face token (or .env / environment)",
    )
    eva.add_argument("--hate-model", default="cardiffnlp/twitter-roberta-base-hate-latest")
    eva.add_argument("--meta-model", default=None, help="Joblib LR fusion model path (see train-meta)")
    eva.add_argument(
        "--aux-type-model",
        default="models/distilbert_social_bias",
        help="Auxiliary type head for dual meta (default: distilbert_social_bias)",
    )
    eva.add_argument("--legacy-fusion", action="store_true", help="Disable learned meta-classifier")
    eva.add_argument("--calibrated", action="store_true", help="Use calibrated meta bundle when available")
    eva.add_argument("--use-thresholds", action="store_true")
    eva.add_argument("--thresholds", default=None)

    sev = subparsers.add_parser("evaluate-sanity", help="Evaluate hard sanity slice with FPR/FNR")
    sev.add_argument(
        "--sanity",
        default="data/meta_training/sanity_slice.csv",
        help="CSV with text,label,slice_type (default: bundled sanity slice)",
    )
    sev.add_argument("--model", default="models/distilbert_B_balanced", help="Primary type head (PRISM stack)")
    sev.add_argument("--out", default="data/output/sanity_eval", help="Output directory")
    sev.add_argument("--batch-size", type=int, default=4)
    sev.add_argument("--hf-token", default=os.environ.get("HF_API_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN"))
    sev.add_argument("--hate-model", default="cardiffnlp/twitter-roberta-base-hate-latest")
    sev.add_argument("--meta-model", default=None, help="Joblib LR fusion model path")
    sev.add_argument(
        "--aux-type-model",
        default="models/distilbert_social_bias",
        help="Auxiliary type head for dual meta (default: distilbert_social_bias)",
    )
    sev.add_argument("--legacy-fusion", action="store_true", help="Disable learned meta-classifier")
    sev.add_argument("--calibrated", action="store_true")
    sev.add_argument("--use-thresholds", action="store_true")
    sev.add_argument("--thresholds", default=None)
    sev.add_argument("--compare-calibration", action="store_true", help="Pre-argmax vs post-calibrated+thresholds")

    bb = subparsers.add_parser(
        "build-benchmark-csv",
        help="StereoSet (HF) + CrowS-Pairs (CSV) → unified training CSV with group_id for leakage-safe splits.",
    )
    bb.add_argument(
        "--output",
        default="data/training/stereoset_crows_combined.csv",
        help="Columns: text,label,group_id,source",
    )
    bb.add_argument("--seed", type=int, default=42)
    bb.add_argument("--balance", action="store_true", help="Equal class counts (undersample to min class).")
    bb.add_argument("--no-neutral-cap", action="store_true")
    bb.add_argument("--neutral-cap-ratio", type=float, default=2.5)
    bb.add_argument(
        "--mix-csv",
        default=None,
        help="Optional text,label CSV to mix in (~mix-fraction × benchmark rows); e.g. generated_social_bias_data.csv",
    )
    bb.add_argument(
        "--mix-fraction",
        type=float,
        default=0.0,
        help="Scale of extra rows vs benchmark size before mix (e.g. 0.075 ≈ 7.5%% added).",
    )
    bb.add_argument("--mix-source-tag", default="mixed_noisy")

    ins = subparsers.add_parser(
        "inspect-benchmark-csv",
        help="Print random samples per label from a stereoset_crows_*.csv (sanity check before training).",
    )
    ins.add_argument("--data", required=True)
    ins.add_argument("--per-label", type=int, default=20)
    ins.add_argument("--seed", type=int, default=42)
    ins.add_argument("--output", default=None, help="Optional Markdown file path")

    ec = subparsers.add_parser(
        "evaluate-crows-stereo",
        help="Macro-F1 of the type head on CrowS-Pairs (stereo-only rows; external benchmark).",
    )
    ec.add_argument("--model", required=True, help="Fine-tuned DistilBERT type head directory")
    ec.add_argument("--batch-size", type=int, default=16)
    ec.add_argument("--json-out", default=None)

    fc = subparsers.add_parser("fit-calibration", help="Build calibration_dataset + lr_meta_calibrated + optimal thresholds")
    fc.add_argument("--gold", default="data/meta_training/gemini_300_posts_curated.csv")
    fc.add_argument("--model", default="models/distilbert_social_bias")
    fc.add_argument(
        "--meta-model",
        default="models/meta_fusion/lr_meta_v2_human.joblib",
        help="Base (uncalibrated) meta joblib; default matches human-v2-trained fusion when present.",
    )
    fc.add_argument("--sanity", default="data/meta_training/sanity_slice.csv")
    fc.add_argument("--out-dataset", default="data/output/calibration/calibration_dataset.json")
    fc.add_argument("--out-calibrated", default="models/meta_fusion/lr_meta_calibrated.joblib")
    fc.add_argument("--out-thresholds", default="models/meta_fusion/optimal_thresholds.json")
    fc.add_argument("--batch-size", type=int, default=8)
    fc.add_argument("--test-size", type=float, default=0.25)
    fc.add_argument("--random-state", type=int, default=42)
    fc.add_argument("--hate-model", default="cardiffnlp/twitter-roberta-base-hate-latest")
    fc.add_argument("--hf-token", default=os.environ.get("HF_API_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN"))
    fc.add_argument(
        "--aux-type-model",
        default=None,
        help="Second type head for 40-d meta during calibration/threshold fitting (production PRISM stack).",
    )

    return parser.parse_args()


def main():
    _load_dotenv()
    configure_logging()
    args = parse_args()

    if args.command == "generate-data":
        output, total = generate_data(samples_per_class=args.samples_per_class, output_file=args.output)
        print(f"Generated dataset at {output} ({total} rows)")

    elif args.command == "train":
        data_path = args.data
        if getattr(args, "fuse", False):
            from train.fusion_dataset import build_fusion_csv

            fusion_path = build_fusion_csv(
                args.real_data,
                args.synthetic_data,
                args.fusion_output,
                seed=getattr(args, "fusion_seed", 42),
                max_real_rows=getattr(args, "max_real", None),
            )
            data_path = str(fusion_path)
            print(f"Fusion dataset written to {data_path}")
        metrics = train(
            data_path=data_path,
            output_dir=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=getattr(args, "seed", 42),
        )
        print("Training complete. Test metrics:")
        print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report_test"}, indent=2, default=str))

    elif args.command == "train-meta":
        from train.train_meta_classifier import train_meta_classifier

        rep = train_meta_classifier(
            args.gold,
            args.type_model,
            args.output,
            hate_model_id=args.hate_model,
            hf_token=getattr(args, "hf_token", None),
            batch_size=args.batch_size,
            test_size=args.test_size,
            random_state=args.random_state,
            max_iter=args.max_iter,
            sanity_slice_path=args.sanity_slice,
            neutral_weight=args.neutral_weight,
            extra_gold_paths=list(getattr(args, "extra_gold", None) or []),
            extra_gold_max_rows=getattr(args, "extra_gold_max_rows", None),
            refit_full=not getattr(args, "no_refit_full", False),
            aux_type_model_dir=getattr(args, "aux_type_model", None),
        )
        print(json.dumps({k: rep[k] for k in rep if k != "classification_report"}, indent=2))

    elif args.command == "analyze":
        results = analyze_posts(
            dataset_path=args.dataset,
            model_dir=args.model,
            output_path=args.output,
            confidence_threshold=args.confidence_threshold,
            batch_size=args.batch_size,
            hf_token=getattr(args, "hf_token", None),
            hate_model_id=getattr(args, "hate_model", None),
            meta_classifier_path=getattr(args, "meta_model", None),
            auxiliary_type_model_dir=getattr(args, "aux_type_model", None),
            legacy_fusion=getattr(args, "legacy_fusion", False),
            calibrated=getattr(args, "calibrated", False),
            use_thresholds=getattr(args, "use_thresholds", False),
            thresholds_path=getattr(args, "thresholds", None),
        )
        out_path = args.output
        print(f"Analyzed {len(results)} posts. Output saved to {out_path}")

    elif args.command == "evaluate":
        from model_evaluation.run_gold_eval import run_evaluation

        rep = run_evaluation(
            gold=args.gold,
            model=args.model,
            out=args.out,
            batch_size=args.batch_size,
            hf_token=getattr(args, "hf_token", None),
            hate_model=args.hate_model,
            meta_classifier_path=getattr(args, "meta_model", None),
            auxiliary_type_model_dir=getattr(args, "aux_type_model", None),
            legacy_fusion=getattr(args, "legacy_fusion", False),
            calibrated=getattr(args, "calibrated", False),
            use_thresholds=getattr(args, "use_thresholds", False),
            thresholds_path=getattr(args, "thresholds", None),
        )
        print(json.dumps(rep["summary"], indent=2))
    elif args.command == "evaluate-sanity":
        from model_evaluation.run_sanity_slice_eval import run_sanity_eval

        rep = run_sanity_eval(
            sanity_csv=args.sanity,
            model=args.model,
            out=args.out,
            batch_size=args.batch_size,
            hf_token=getattr(args, "hf_token", None),
            hate_model=args.hate_model,
            meta_classifier_path=getattr(args, "meta_model", None),
            auxiliary_type_model_dir=getattr(args, "aux_type_model", None),
            legacy_fusion=getattr(args, "legacy_fusion", False),
            calibrated=getattr(args, "calibrated", False),
            use_thresholds=getattr(args, "use_thresholds", False),
            thresholds_path=getattr(args, "thresholds", None),
            compare_calibration=getattr(args, "compare_calibration", False),
        )
        print(json.dumps(rep, indent=2))

    elif args.command == "build-benchmark-csv":
        from train.build_stereoset_crows_csv import build_combined_csv

        cap = None if getattr(args, "no_neutral_cap", False) else getattr(args, "neutral_cap_ratio", 2.5)
        out = build_combined_csv(
            args.output,
            balance=getattr(args, "balance", False),
            neutral_cap_ratio=cap,
            seed=getattr(args, "seed", 42),
            mix_csv=getattr(args, "mix_csv", None),
            mix_fraction=float(getattr(args, "mix_fraction", 0.0)),
            mix_source_tag=getattr(args, "mix_source_tag", "mixed_noisy"),
        )
        print(f"Wrote benchmark training CSV: {out}")

    elif args.command == "inspect-benchmark-csv":
        from models.label_config import LABELS
        from train.inspect_benchmark_csv import format_samples_md, sample_benchmark_csv

        samples = sample_benchmark_csv(args.data, per_label=args.per_label, seed=args.seed)
        for lab in LABELS:
            for i, (txt, src, gid) in enumerate(samples[lab], 1):
                meta = f" [{src}]" if src else ""
                print(f"\n--- {lab} #{i}{meta} ---\n{txt}\n")
        if args.output:
            md = format_samples_md(samples, title=f"Benchmark CSV samples: {args.data}")
            Path(args.output).write_text(md, encoding="utf-8")
            print(f"Wrote {args.output}")

    elif args.command == "evaluate-crows-stereo":
        from train.eval_type_head_on_crows_stereo import evaluate_type_head_crows

        rep = evaluate_type_head_crows(args.model, batch_size=args.batch_size)
        print(rep.get("classification_report_text", ""))
        print(json.dumps({"macro_f1": rep["macro_f1"], "n_examples": rep["n_examples"]}, indent=2))
        if getattr(args, "json_out", None):
            p = Path(args.json_out)
            p.parent.mkdir(parents=True, exist_ok=True)
            serializable = {k: v for k, v in rep.items() if k != "classification_report_text"}
            p.write_text(json.dumps(serializable, indent=2, default=str), encoding="utf-8")

    elif args.command == "fit-calibration":
        from model_evaluation.run_calibration_fit import collect_and_fit_calibration
        from model_evaluation.run_calibration_split_eval import collect_calibration_records

        root = Path(__file__).resolve().parent
        gold_path = Path(args.gold)
        gold_path = gold_path if gold_path.is_absolute() else root / gold_path

        ds_out = Path(args.out_dataset)
        ds_out = ds_out if ds_out.is_absolute() else root / ds_out
        ds_out.parent.mkdir(parents=True, exist_ok=True)
        aux_tm = getattr(args, "aux_type_model", None) or None
        aux_resolved = None
        if aux_tm:
            aux_resolved = str(root / aux_tm) if not Path(aux_tm).is_absolute() else aux_tm

        ds_payload = collect_calibration_records(
            str(gold_path),
            str(root / args.model) if not Path(args.model).is_absolute() else args.model,
            str(root / args.meta_model) if not Path(args.meta_model).is_absolute() else args.meta_model,
            hate_model=args.hate_model,
            hf_token=args.hf_token,
            batch_size=args.batch_size,
            test_size=args.test_size,
            random_state=args.random_state,
            auxiliary_type_model_dir=aux_resolved,
        )
        ds_out.write_text(json.dumps(ds_payload, indent=2), encoding="utf-8")
        print(f"Wrote calibration dataset: {ds_out}")
        fit_paths = collect_and_fit_calibration(
            str(ds_out),
            str(root / args.meta_model) if not Path(args.meta_model).is_absolute() else args.meta_model,
            str(root / args.out_calibrated) if not Path(args.out_calibrated).is_absolute() else args.out_calibrated,
            str(root / args.out_thresholds) if not Path(args.out_thresholds).is_absolute() else args.out_thresholds,
            str(root / args.sanity) if not Path(args.sanity).is_absolute() else args.sanity,
            str(root / args.model) if not Path(args.model).is_absolute() else args.model,
            batch_size=args.batch_size,
            hate_model=args.hate_model,
            hf_token=args.hf_token,
            auxiliary_type_model_dir=aux_resolved,
        )
        print(json.dumps(fit_paths, indent=2, default=str))


if __name__ == "__main__":
    main()
