"""Train DistilBERT model for social bias classification (4-way: no binary collapse)."""

from __future__ import annotations

import inspect
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from models.label_config import LABELS, LABEL_TO_ID, ID_TO_LABEL

LOGGER = logging.getLogger(__name__)


def _ensure_hf_token_env() -> None:
    """
    huggingface_hub may read HF_TOKEN or HUGGINGFACE_HUB_TOKEN depending on version.
    Sync env so a token set under either name authenticates Hub downloads.
    """
    tok = (
        os.environ.get("HUGGINGFACE_HUB_TOKEN", "").strip()
        or os.environ.get("HF_TOKEN", "").strip()
        or os.environ.get("HUGGINGFACEHUB_API_TOKEN", "").strip()
    )
    if tok:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = tok
        os.environ["HF_TOKEN"] = tok
        LOGGER.info("Hugging Face Hub token detected in environment (authenticated downloads).")
    else:
        LOGGER.warning(
            "No HF_TOKEN / HUGGINGFACE_HUB_TOKEN in environment — Hub may throttle unauthenticated requests. "
            "On Colab: add HF_TOKEN under Secrets and run the notebook cell that sets os.environ."
        )


def _make_training_arguments(**kwargs):
    """Transformers 5.x renamed evaluation_strategy → eval_strategy; support both."""
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in kwargs and "eval_strategy" in ta_params and "evaluation_strategy" not in ta_params:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    elif "eval_strategy" in kwargs and "evaluation_strategy" in ta_params and "eval_strategy" not in ta_params:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    # Drop keys unknown to this transformers version (defensive)
    allowed = set(ta_params) - {"self"}
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    unknown = set(kwargs) - set(filtered)
    if unknown:
        LOGGER.warning("Ignoring TrainingArguments keys not in this transformers build: %s", sorted(unknown))
    return TrainingArguments(**filtered)


def _trainer_tokenizer_kwargs(tokenizer) -> dict:
    """
    Transformers 5.x: Trainer no longer accepts `tokenizer=...`; use `processing_class=...` when available.
    Older versions expect `tokenizer=...`.
    """
    sig = inspect.signature(Trainer.__init__).parameters
    if "tokenizer" in sig:
        return {"tokenizer": tokenizer}
    if "processing_class" in sig:
        return {"processing_class": tokenizer}
    return {}


def _metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    ids = np.arange(len(LABELS))
    p, r, f1, sup = precision_recall_fscore_support(
        labels, preds, labels=ids, average=None, zero_division=0
    )
    out: dict[str, float] = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision_macro": float(np.mean(p)),
        "recall_macro": float(np.mean(r)),
        "f1_macro": float(np.mean(f1)),
        "precision_weighted": float(
            precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)[0]
        ),
        "recall_weighted": float(
            precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)[1]
        ),
        "f1_weighted": float(
            precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)[2]
        ),
    }
    for i, lab in enumerate(LABELS):
        if i < len(p):
            out[f"precision_{lab}"] = float(p[i])
            out[f"recall_{lab}"] = float(r[i])
            out[f"f1_{lab}"] = float(f1[i])
            out[f"support_{lab}"] = float(sup[i])
    return out


def _load_training_data(path: str, *, seed: int = 42) -> tuple[DatasetDict, dict[str, int]]:
    """Load CSV with string labels in LABELS; stratified train/val/test.

    If a ``group_id`` column is present (e.g. StereoSet/CrowS-Pairs build), splits are **group-aware**
    so related sentences from the same benchmark item do not leak across train/val/test.
    """
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Training CSV must have columns: text, label (see fusion_dataset / gold format).")
    df = df.dropna(subset=["text", "label"]).copy()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[df["label"].isin(LABELS)].reset_index(drop=True)
    if df["label"].nunique() < len(LABELS):
        raise ValueError("Training data must include all four labels: %s" % LABELS)
    class_counts = df["label"].value_counts()
    if class_counts.min() != class_counts.max():
        LOGGER.warning(
            "Class counts are not perfectly balanced (min=%s max=%s). Fusion script should balance; continuing.\n%s",
            int(class_counts.min()),
            int(class_counts.max()),
            class_counts,
        )

    use_groups = "group_id" in df.columns and df["group_id"].notna().any()
    if use_groups:
        n_gid = df["group_id"].nunique()
        LOGGER.info("Using group-aware splits (%s unique group_id).", n_gid)

    split_counts = df["label"].value_counts().to_dict()
    LOGGER.info("Loaded training CSV %s rows; label distribution:\n%s", len(df), split_counts)

    df["label_id"] = df["label"].map(LABEL_TO_ID)

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if use_groups:
        groups = df["group_id"].astype(str).values
        gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
        idx_all = np.arange(len(df))
        tr_val_idx, te_idx = next(gss_test.split(idx_all, groups=groups))
        df_tr_val = df.iloc[tr_val_idx].reset_index(drop=True)
        df_test = df.iloc[te_idx].reset_index(drop=True)
        g_val = df_tr_val["group_id"].astype(str).values
        # val ≈ 10% of full set: 0.10 / 0.85 of the train+val pool
        gss_val = GroupShuffleSplit(n_splits=1, test_size=0.10 / 0.85, random_state=seed + 1)
        inner_idx = np.arange(len(df_tr_val))
        tr_rel, va_rel = next(gss_val.split(inner_idx, groups=g_val))
        train_df = df_tr_val.iloc[tr_rel].reset_index(drop=True)
        val_df = df_tr_val.iloc[va_rel].reset_index(drop=True)
        test_df = df_test.reset_index(drop=True)
    else:
        train_df, test_df = train_test_split(
            df, test_size=0.15, random_state=seed, stratify=df["label_id"]
        )
        train_df, val_df = train_test_split(
            train_df, test_size=0.1, random_state=seed, stratify=train_df["label_id"]
        )

    train_df = train_df.drop(columns=["label"], errors="ignore").rename(columns={"label_id": "label"})
    val_df = val_df.drop(columns=["label"], errors="ignore").rename(columns={"label_id": "label"})
    test_df = test_df.drop(columns=["label"], errors="ignore").rename(columns={"label_id": "label"})
    drop_extra = [c for c in ("group_id", "source") if c in train_df.columns]
    if drop_extra:
        train_df = train_df.drop(columns=drop_extra)
        val_df = val_df.drop(columns=drop_extra)
        test_df = test_df.drop(columns=drop_extra)

    LOGGER.info(
        "Splits — train=%s val=%s test=%s",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    LOGGER.info("Train label counts:\n%s", train_df["label"].map(lambda i: ID_TO_LABEL[int(i)]).value_counts())

    ds = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
            "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
            "test": Dataset.from_pandas(test_df.reset_index(drop=True)),
        }
    )
    return ds, split_counts


def train(
    data_path: str = "data/training/generated_social_bias_data.csv",
    model_name: str = "distilbert-base-uncased",
    output_dir: str = "models/distilbert_social_bias",
    epochs: int = 5,
    batch_size: int = 16,
    seed: int = 42,
):
    _ensure_hf_token_env()
    dataset, raw_counts = _load_training_data(data_path, seed=seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=256)

    tokenized = dataset.map(tokenize, batched=True)
    drop_cols = [c for c in tokenized["train"].column_names if c not in {"input_ids", "attention_mask", "label"}]
    tokenized = tokenized.remove_columns(drop_cols)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    args = _make_training_arguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=2e-5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        metric_for_best_model="f1_macro",
        logging_steps=max(25, len(tokenized["train"]) // (batch_size * 10) or 25),
        save_total_limit=2,
        report_to="none",
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=_metrics,
        **_trainer_tokenizer_kwargs(tokenizer),
    )

    LOGGER.info("Starting training: epochs=%s batch_size=%s", epochs, batch_size)
    trainer.train()

    LOGGER.info("Evaluating on validation set (final epoch metrics already logged).")
    test_metrics = trainer.evaluate(tokenized["test"])

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Per-class report on test
    preds = trainer.predict(tokenized["test"]).predictions
    y_pred = np.argmax(preds, axis=1)
    y_true = np.array(tokenized["test"]["label"])
    rep = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        zero_division=0,
        output_dict=True,
    )
    LOGGER.info("Test classification_report:\n%s", classification_report(y_true, y_pred, labels=list(range(len(LABELS))), target_names=LABELS, zero_division=0))

    out = {**test_metrics, "classification_report_test": rep, "raw_label_counts_loaded": raw_counts}
    return out
