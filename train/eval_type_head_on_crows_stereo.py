"""External eval: CrowS-Pairs stereo rows only (sent_more / sent_less gold), type-head macro-F1."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, f1_score

from models.bias_type_head import BiasTypeHead
from models.label_config import LABEL_TO_ID, LABELS
from train.build_stereoset_crows_csv import rows_from_crows_pairs_stereo_only

LOGGER = logging.getLogger(__name__)


def build_crows_stereo_eval_pairs() -> tuple[list[str], list[str]]:
    """Gold: stereo-only pairs; sent_more → bias label, sent_less → neutral."""
    rows = rows_from_crows_pairs_stereo_only()
    texts: list[str] = []
    labels: list[str] = []
    by_g: dict[str, list[dict]] = {}
    for r in rows:
        by_g.setdefault(r["group_id"], []).append(r)
    for gid, items in by_g.items():
        if len(items) != 2:
            LOGGER.warning("Unexpected group %s size %s", gid, len(items))
            continue
        items_sorted = sorted(items, key=lambda x: 0 if x["label"] != "neutral" else 1)
        for it in items_sorted:
            texts.append(it["text"])
            labels.append(it["label"])
    return texts, labels


def _logit_argmax_label(head: BiasTypeHead, logits: np.ndarray) -> str:
    pid = int(np.argmax(logits))
    id2label = head.model.config.id2label
    lab = id2label.get(pid)
    if lab is None:
        lab = id2label.get(str(pid))
    if lab is None:
        raise KeyError(f"No id2label for pred id {pid}: {id2label}")
    return str(lab)


def evaluate_type_head_crows(
    model_dir: str | Path,
    *,
    batch_size: int = 16,
) -> dict:
    texts, gold = build_crows_stereo_eval_pairs()
    head = BiasTypeHead(model_dir)
    preds: list[str] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        for t in chunk:
            logits = head.forward_logits(t)
            preds.append(_logit_argmax_label(head, logits))

    y_true = np.array([LABEL_TO_ID[g] for g in gold])
    y_pred = np.array([LABEL_TO_ID[p] for p in preds])

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    rep = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        zero_division=0,
        output_dict=True,
    )
    LOGGER.info("CrowS-Pairs (stereo rows only) — n=%s macro-F1=%.4f", len(texts), macro_f1)
    text_report = classification_report(
        gold,
        preds,
        labels=LABELS,
        zero_division=0,
    )
    return {
        "n_examples": len(texts),
        "macro_f1": macro_f1,
        "classification_report": rep,
        "classification_report_text": text_report,
    }


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Macro-F1 on CrowS-Pairs (stereo-only gold).")
    p.add_argument("--model", required=True, help="Fine-tuned type head directory")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--json-out", default=None, help="Optional path to save metrics JSON.")
    args = p.parse_args()
    out = evaluate_type_head_crows(args.model, batch_size=args.batch_size)
    print(json.dumps({k: v for k, v in out.items() if k not in ("classification_report", "classification_report_text")}, indent=2))
    print(out["classification_report_text"])
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        serializable = {k: v for k, v in out.items() if k != "classification_report_text"}
        Path(args.json_out).write_text(json.dumps(serializable, indent=2, default=str), encoding="utf-8")
        LOGGER.info("Wrote %s", args.json_out)


if __name__ == "__main__":
    main()
