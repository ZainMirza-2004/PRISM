"""Train logistic regression meta-classifier on frozen DistilBERT + RoBERTa + rule features.

Use ``--extra-gold`` to merge additional labeled CSVs (dedupe by ``post_id``). For example, add
``data/evaluation/manual_eval_v3_500_posts.csv`` so the fusion layer trains on the benchmark-style
distribution (disclose in write-ups if those labels overlap the held-out eval protocol).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.loaders import load_labeled_goldset
from models.bias_type_head import BiasTypeHead
from models.feature_vector import (
    FEATURE_SCHEMA_VERSION,
    META_FEATURE_NAMES,
    META_FEATURE_NAMES_CORE,
    build_meta_feature_row,
)
from models.fusion_engine import fusion_distribution_snapshot
from models.hf_hate_client import classify_hate_batch
from models.label_config import LABELS, NEUTRAL_LABEL
from models.preprocess import preprocess_social_post
from models.rule_signals import extract_rule_fusion_signals


def _merge_gold_rows(gold_path: str, extra_gold_paths: list[str] | None) -> list[dict]:
    """Primary gold CSV plus optional extras (dedupe by post_id)."""
    rows = load_labeled_goldset(gold_path)
    seen = {str(r.get("post_id", "")) for r in rows}
    for ep in extra_gold_paths or []:
        if not ep:
            continue
        p = Path(ep)
        if not p.is_file():
            continue
        for r in load_labeled_goldset(str(p)):
            pid = str(r.get("post_id", ""))
            if pid in seen:
                continue
            seen.add(pid)
            rows.append(r)
    return rows


def _sample_rows(rows: list[dict], max_rows: int | None, random_state: int) -> list[dict]:
    """Deterministically subsample rows while preserving full set when uncapped."""
    if max_rows is None or max_rows <= 0 or len(rows) <= max_rows:
        return rows
    rng = np.random.default_rng(int(random_state))
    idx = rng.choice(len(rows), size=int(max_rows), replace=False)
    idx_sorted = np.sort(idx)
    return [rows[int(i)] for i in idx_sorted]


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    p = _ROOT / ".env"
    if p.is_file():
        load_dotenv(p)


def build_training_matrix(
    rows: list[dict],
    type_model_dir: str,
    *,
    hate_model_id: str,
    hf_token: str | None,
    batch_size: int,
    aux_type_model_dir: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    texts = [str(r.get("text", "")) for r in rows]
    labels = [str(r.get("label", "")).strip() for r in rows]
    cleans = [preprocess_social_post(t) for t in texts]

    head = BiasTypeHead(type_model_dir)
    dists = head.predict_type_distribution_batch(cleans, batch_size=batch_size)
    dists_social: list[dict[str, float]] | None = None
    if aux_type_model_dir:
        aux = BiasTypeHead(aux_type_model_dir)
        dists_social = aux.predict_type_distribution_batch(cleans, batch_size=batch_size)
    hates = classify_hate_batch(cleans, model_id=hate_model_id, token=hf_token)

    X_list: list[np.ndarray] = []
    y_list: list[str] = []
    for i, (lab, clean, dist0, hr) in enumerate(zip(labels, cleans, dists, hates)):
        if lab not in LABELS:
            continue
        dist = fusion_distribution_snapshot(dist0)
        rules = extract_rule_fusion_signals(clean)
        ds0 = dists_social[i] if dists_social else None
        dist_social_snap = fusion_distribution_snapshot(ds0) if ds0 is not None else None
        row = build_meta_feature_row(
            dist, float(hr.p_hate), rules, clean_text=clean, dist_social=dist_social_snap
        )
        X_list.append(row)
        y_list.append(lab)

    if not X_list:
        raise ValueError("No valid labeled rows (labels must be one of: %s)." % LABELS)

    X = np.vstack(X_list)
    y = np.asarray(y_list)
    return X, y


def train_meta_classifier(
    gold_path: str,
    type_model_dir: str,
    output_joblib: str,
    *,
    hate_model_id: str,
    hf_token: str | None,
    batch_size: int,
    test_size: float,
    random_state: int,
    max_iter: int,
    sanity_slice_path: str | None = None,
    neutral_weight: float | None = None,
    extra_gold_paths: list[str] | None = None,
    refit_full: bool = True,
    aux_type_model_dir: str | None = None,
    extra_gold_max_rows: int | None = None,
) -> dict:
    rows = _merge_gold_rows(gold_path, extra_gold_paths)
    if extra_gold_paths:
        base_n = len(load_labeled_goldset(gold_path))
        extra_rows = rows[base_n:]
        sampled_extra = _sample_rows(extra_rows, extra_gold_max_rows, random_state)
        rows = rows[:base_n] + sampled_extra
    X, y = build_training_matrix(
        rows,
        type_model_dir,
        hate_model_id=hate_model_id,
        hf_token=hf_token,
        batch_size=batch_size,
        aux_type_model_dir=aux_type_model_dir,
    )

    # Stratified split when enough samples per class
    uniq, counts = np.unique(y, return_counts=True)
    min_per_class = int(counts.min()) if len(counts) else 0
    use_split = len(y) >= 8 and min_per_class >= 2 and test_size > 0

    if use_split:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        X_train, y_train = X, y
        X_test = y_test = np.array([])

    penalties = ["l2"]
    cs = [0.25, 0.5, 1.0, 2.0, 4.0]
    class_weights = ["balanced"]
    if neutral_weight is not None:
        class_weights.append({NEUTRAL_LABEL: float(neutral_weight)})

    def _build_pipeline(c_val: float, penalty: str, class_weight: object) -> Pipeline:
        lr = LogisticRegression(
            max_iter=max_iter,
            class_weight=class_weight,
            solver="lbfgs",
            random_state=random_state,
            C=float(c_val),
            penalty=penalty,
        )
        return Pipeline([("scaler", StandardScaler()), ("lr", lr)])

    sanity_rows: list[dict] = []
    if sanity_slice_path:
        p = Path(sanity_slice_path)
        if p.is_file():
            sanity_rows = load_labeled_goldset(str(p))
    X_sanity = y_sanity = None
    sanity_types: list[str] = []
    if sanity_rows:
        Xs, ys = build_training_matrix(
            sanity_rows,
            type_model_dir,
            hate_model_id=hate_model_id,
            hf_token=hf_token,
            batch_size=batch_size,
            aux_type_model_dir=aux_type_model_dir,
        )
        X_sanity, y_sanity = Xs, ys
        sanity_types = [str(r.get("slice_type", "")).strip() for r in sanity_rows if str(r.get("label", "")).strip() in LABELS]

    def _sanity_fpr(model: Pipeline) -> float:
        if X_sanity is None or y_sanity is None or not sanity_types:
            return 1.0
        preds = model.predict(X_sanity)
        fp = 0
        denom = 0
        for yt, yp, st in zip(y_sanity, preds, sanity_types):
            if st in {"neutral_policy", "positive_group"}:
                denom += 1
                if yp != NEUTRAL_LABEL:
                    fp += 1
        return float(fp / denom) if denom else 1.0

    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state)
    candidates: list[dict] = []
    for c_val in cs:
        for penalty in penalties:
            for class_weight in class_weights:
                fold_scores: list[float] = []
                for tr_idx, va_idx in cv.split(X_train, y_train):
                    model = _build_pipeline(c_val, penalty, class_weight)
                    model.fit(X_train[tr_idx], y_train[tr_idx])
                    pred = model.predict(X_train[va_idx])
                    fold_scores.append(float(f1_score(y_train[va_idx], pred, labels=list(LABELS), average="macro", zero_division=0)))
                mean_f1 = float(np.mean(fold_scores))
                full_model = _build_pipeline(c_val, penalty, class_weight)
                full_model.fit(X_train, y_train)
                candidates.append(
                    {
                        "C": c_val,
                        "penalty": penalty,
                        "class_weight": class_weight,
                        "cv_f1_macro": mean_f1,
                        "sanity_fpr": _sanity_fpr(full_model),
                        "model": full_model,
                    }
                )

    # Prefer higher CV macro-F1 but penalize high FPR on neutral_policy / positive_group rows.
    sanity_penalty = float(os.environ.get("META_SANITY_FPR_PENALTY", "0.22"))
    candidates.sort(
        key=lambda x: (
            -(float(x["cv_f1_macro"]) - sanity_penalty * float(x["sanity_fpr"])),
            float(x["sanity_fpr"]),
        )
    )
    best = candidates[0]
    train_only_model: Pipeline = best["model"]
    if refit_full:
        pipeline = _build_pipeline(best["C"], best["penalty"], best["class_weight"])
        pipeline.fit(X, y)
    else:
        pipeline = train_only_model

    lr_classes = pipeline.named_steps["lr"].classes_
    report: dict = {
        "n_train": int(len(y_train)),
        "n_gold_rows": int(len(rows)),
        "refit_full_dataset": bool(refit_full),
        "extra_gold_sources": list(extra_gold_paths or []),
        "extra_gold_max_rows": (int(extra_gold_max_rows) if extra_gold_max_rows else None),
        "feature_names": (META_FEATURE_NAMES if X.shape[1] >= 40 else META_FEATURE_NAMES_CORE),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "classes": list(map(str, lr_classes.tolist())),
        "model_selection": [
            {
                "C": c["C"],
                "penalty": c["penalty"],
                "class_weight": c["class_weight"],
                "cv_f1_macro": c["cv_f1_macro"],
                "sanity_fpr": c["sanity_fpr"],
            }
            for c in candidates
        ],
        "selected_hyperparams": {
            "C": best["C"],
            "penalty": best["penalty"],
            "class_weight": best["class_weight"],
            "cv_f1_macro": best["cv_f1_macro"],
            "sanity_fpr": best["sanity_fpr"],
        },
    }

    coefs = pipeline.named_steps["lr"].coef_
    mean_abs = np.mean(np.abs(coefs), axis=0)
    n_feat = int(mean_abs.shape[0])
    groups = {
        "distilbert_primary": [0, 1, 2, 3],
        "hate": [4],
        "rules": [5, 6, 7, 8, 9, 10, 11, 12],
        "linguistic": [13, 14, 15, 16, 17, 18, 19, 20],
        "stance": [21, 22, 23, 24, 25, 26, 27, 28, 29],
        "methodology": [30, 31, 32, 33, 34, 35],
    }
    if n_feat >= 40:
        groups["distilbert_social_aux"] = [36, 37, 38, 39]
    report["coefficient_magnitude_by_group"] = {
        g: float(np.mean([mean_abs[i] for i in idxs if i < n_feat]))
        for g, idxs in groups.items()
        if any(i < n_feat for i in idxs)
    }

    if len(y_test):
        # When refit_full, evaluate hold-out with train-split model to avoid optimistic bias
        eval_model = train_only_model if refit_full else pipeline
        y_pred = eval_model.predict(X_test)
        report["n_test"] = int(len(y_test))
        report["classification_report"] = classification_report(y_test, y_pred, labels=list(LABELS), zero_division=0, output_dict=True)
        report["holdout_evaluator"] = "train_split_model" if refit_full else "final_model"

    out_path = Path(output_joblib)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feat_names_out = META_FEATURE_NAMES if X.shape[1] >= 40 else META_FEATURE_NAMES_CORE
    meta_payload = {
        "pipeline": pipeline,
        "feature_names": feat_names_out,
        "label_order": list(LABELS),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
    }
    joblib.dump(meta_payload, out_path)

    sidecar = out_path.with_suffix(".json")
    sidecar.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["artifact"] = str(out_path)
    report["sidecar"] = str(sidecar)
    return report


def main() -> None:
    _load_dotenv()
    ap = argparse.ArgumentParser(
        description="Train LR meta-classifier fusion model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example — merged gold + dual 40-d head (StereoSet B primary):\n"
            "  python main.py train-meta --gold data/evaluation/manual_eval_v2_200_posts.csv \\\n"
            "    --extra-gold data/evaluation/manual_eval_v3_500_posts.csv \\\n"
            "    --type-model models/distilbert_B_balanced \\\n"
            "    --aux-type-model models/distilbert_social_bias \\\n"
            "    --output models/meta_fusion/lr_meta_dual_distilbert_B_balanced_plus_distilbert_social_bias.joblib\n"
        ),
    )
    ap.add_argument("--gold", required=True, help="CSV/JSON with text + label columns")
    ap.add_argument("--type-model", default="models/distilbert_B_balanced", help="Primary fine-tuned DistilBERT type head dir")
    ap.add_argument(
        "--aux-type-model",
        default=None,
        help="Optional second type head (e.g. models/distilbert_social_bias); extends meta features to 40-d and expects matching --output naming (lr_meta_dual_*_plus_*).",
    )
    ap.add_argument("--output", default="models/meta_fusion/lr_meta.joblib", help="Output joblib path")
    ap.add_argument("--hate-model", default="cardiffnlp/twitter-roberta-base-hate-latest")
    ap.add_argument("--hf-token", default=os.environ.get("HF_API_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN"))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument("--sanity-slice", default="data/meta_training/sanity_slice.csv")
    ap.add_argument("--neutral-weight", type=float, default=None)
    ap.add_argument(
        "--extra-gold",
        action="append",
        default=[],
        help=(
            "Additional labeled CSV/JSON merged after --gold (dedupe by post_id). "
            "Repeat for multiple files (e.g. Gemini-curated sets). "
            "Using data/evaluation/manual_eval_v3_500_posts.csv mixes benchmark-style posts into training."
        ),
    )
    ap.add_argument(
        "--extra-gold-max-rows",
        type=int,
        default=None,
        help="Optional cap on merged --extra-gold rows (e.g. 100 from v3-500 to preserve 400 for held-out eval).",
    )
    ap.add_argument("--no-refit-full", action="store_true", help="Skip final refit on full merged dataset")
    args = ap.parse_args()

    rep = train_meta_classifier(
        args.gold,
        args.type_model,
        args.output,
        hate_model_id=args.hate_model,
        hf_token=args.hf_token,
        batch_size=args.batch_size,
        test_size=args.test_size,
        random_state=args.random_state,
        max_iter=args.max_iter,
        sanity_slice_path=args.sanity_slice,
        neutral_weight=args.neutral_weight,
        extra_gold_paths=list(args.extra_gold or []),
        extra_gold_max_rows=args.extra_gold_max_rows,
        refit_full=not args.no_refit_full,
        aux_type_model_dir=args.aux_type_model,
    )
    print(json.dumps({k: rep[k] for k in rep if k != "classification_report"}, indent=2))
    if "classification_report" in rep:
        print("Test classification_report written to sidecar .json")


if __name__ == "__main__":
    main()
