"""Local fine-tuned head: 4-way bias type (DistilBERT-sized; runs on CPU)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from models.label_config import LABELS, NEUTRAL_LABEL


def _temper_probs(label_to_p: Dict[str, float], temperature: float = 1.35) -> Dict[str, float]:
    T = max(0.5, float(temperature))
    keys = list(LABELS)
    raw = []
    for k in keys:
        p = max(1e-9, float(label_to_p.get(k, 0.0)))
        raw.append(p ** (1.0 / T))
    s = float(sum(raw))
    if s <= 0:
        return {k: 1.0 / len(keys) for k in keys}
    return {k: raw[i] / s for i, k in enumerate(keys)}


class BiasTypeHead:
    def __init__(self, model_dir: str | Path, *, calm_temperature: float = 1.35):
        self.model_dir = str(model_dir)
        self.calm_temperature = calm_temperature
        self.device = -1
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.eval()
        if getattr(self.model.config, "model_type", "") == "distilbert":
            # DistilBERT forward() does not accept token_type_ids.
            self.tokenizer.model_input_names = [
                n for n in self.tokenizer.model_input_names if n != "token_type_ids"
            ]

        self.classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            return_all_scores=True,
            truncation=True,
            max_length=256,
        )

    def _scores_to_map(self, scores_list: List[Dict]) -> Dict[str, float]:
        m: Dict[str, float] = {k: 0.0 for k in LABELS}
        for item in scores_list:
            lab = str(item.get("label", ""))
            sc = float(item.get("score", 0.0))
            if lab.startswith("LABEL_"):
                idx = int(lab.split("_")[-1])
                lab2 = self.model.config.id2label.get(idx, lab)
                lab = str(lab2)
            if lab in m:
                m[lab] = max(m[lab], sc)
        return m

    def predict_type_distribution(self, text: str) -> Dict[str, float]:
        batch = self.classifier(text)
        # single string -> list of list in some versions
        if batch and isinstance(batch[0], dict):
            scores_list = batch
        else:
            scores_list = batch[0] if batch else []
        m = self._scores_to_map(scores_list)
        return _temper_probs(m, self.calm_temperature)

    def predict_type_distribution_batch(self, texts: List[str], batch_size: int = 8) -> List[Dict[str, float]]:
        raw = self.classifier(texts, batch_size=batch_size)
        out: List[Dict[str, float]] = []
        for row in raw:
            scores_list = row if row and isinstance(row[0], dict) else []
            m = self._scores_to_map(scores_list)
            out.append(_temper_probs(m, self.calm_temperature))
        return out

    def argmax_label(self, dist: Dict[str, float]) -> str:
        return max(dist.items(), key=lambda x: x[1])[0]

    def forward_logits(self, text: str) -> np.ndarray:
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        enc.pop("token_type_ids", None)
        with torch.no_grad():
            o = self.model(**enc)
            return o.logits.detach().cpu().numpy().reshape(-1)
