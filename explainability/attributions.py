"""Word-level attribution utilities using transformers-interpret."""

from __future__ import annotations

import re
from typing import List, Tuple

from transformers_interpret import SequenceClassificationExplainer

NOISE_TOKENS = {"[CLS]", "[SEP]", "[PAD]", "[UNK]"}
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has",
    "have", "in", "is", "it", "its", "of", "on", "or", "that", "the", "their",
    "them", "they", "this", "to", "was", "we", "with", "you", "your", "our",
}
PUNCT_PATTERN = re.compile(r"^[^\w]+$")


def _merge_subwords(raw_attributions: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    merged: List[Tuple[str, float]] = []
    current_token = ""
    current_score = 0.0

    for token, score in raw_attributions:
        token = token.strip()
        if not token:
            continue
        if token.startswith("##"):
            sub = token[2:]
            if current_token:
                current_token += sub
                current_score += score
            else:
                current_token = sub
                current_score = score
            continue

        if current_token:
            merged.append((current_token, current_score))
        current_token = token
        current_score = score

    if current_token:
        merged.append((current_token, current_score))
    return merged


def clean_word_attributions(raw_attributions: List[Tuple[str, float]], top_k: int = 5) -> List[str]:
    merged = _merge_subwords(raw_attributions)
    cleaned = []
    for token, score in merged:
        token = token.strip()
        if not token or token in NOISE_TOKENS:
            continue
        token_lower = token.lower()
        if token_lower in STOPWORDS:
            continue
        if PUNCT_PATTERN.match(token):
            continue
        if score <= 0:
            continue
        cleaned.append((token, score))

    deduped = []
    seen = set()
    for token, score in sorted(cleaned, key=lambda x: x[1], reverse=True):
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(token)
        if len(deduped) >= top_k:
            break
    return deduped


def explain_text(model, tokenizer, text: str, predicted_label: str, top_k: int = 5) -> List[str]:
    if predicted_label == "neutral":
        return []
    explainer = SequenceClassificationExplainer(model, tokenizer)
    raw = explainer(text, class_name=predicted_label)
    return clean_word_attributions(raw, top_k=top_k)
