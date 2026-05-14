"""Legacy local-only DistilBERT wrapper (sentiment + masked attributions).

Prefer :class:`HybridBiasPipeline` for production: it uses the Cardiff RoBERTa hate model
via the Hugging Face Inference API plus this repo's fine-tuned type head and light phrase cues.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

from explainability.attributions import explain_text
from models.label_config import ID_TO_LABEL, NEUTRAL_LABEL


@dataclass
class PredictionResult:
    prediction: str
    confidence: float
    important_words: List[str]


class SocialBiasDetector:
    def __init__(self, model_dir: str | Path):
        model_dir = str(model_dir)
        self.device = -1
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()

        self.classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            return_all_scores=False,
            truncation=True,
            max_length=256,
        )
        self.sentiment = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=self.device,
            truncation=True,
            max_length=256,
        )

    def predict_batch(self, texts: List[str], batch_size: int = 16) -> List[Dict]:
        return self.classifier(texts, batch_size=batch_size)

    def normalize_prediction(self, raw_label: str, score: float, confidence_threshold: float = 0.0) -> str:
        label = raw_label
        if raw_label.startswith("LABEL_"):
            label_idx = int(raw_label.split("_")[-1])
            label = ID_TO_LABEL.get(label_idx, NEUTRAL_LABEL)
        if score < confidence_threshold:
            return NEUTRAL_LABEL
        return label

    def predict_one(self, text: str, confidence_threshold: float = 0.0) -> PredictionResult:
        result = self.classifier(text)[0]
        label = result["label"]
        score = float(result["score"])
        label = self.normalize_prediction(label, score, confidence_threshold)

        important_words = self.explain(text, label)
        return PredictionResult(prediction=label, confidence=score, important_words=important_words)

    def explain(self, text: str, predicted_label: str, top_k: int = 5) -> List[str]:
        return explain_text(self.model, self.tokenizer, text, predicted_label, top_k=top_k)

    def predict_sentiment(self, text: str) -> str:
        sent = self.sentiment(text)[0]["label"].lower()
        if sent == "positive":
            return "positive"
        if sent == "negative":
            return "negative"
        return "neutral"

    def predict_sentiment_batch(self, texts: List[str], batch_size: int = 16) -> List[str]:
        results = self.sentiment(texts, batch_size=batch_size)
        normalized = []
        for result in results:
            label = result["label"].lower()
            if label == "positive":
                normalized.append("positive")
            elif label == "negative":
                normalized.append("negative")
            else:
                normalized.append("neutral")
        return normalized
