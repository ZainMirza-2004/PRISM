"""Salient content words and hybrid important-word selection (non-empty, interpretable)."""

from __future__ import annotations

import re
from typing import List, Set

from models.label_config import NEUTRAL_LABEL
from models.interpretation_signals import HeuristicSignals

# Extended stop list for "salient neutral" display (keeps some domain social words)
_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has", "have", "in", "is", "it",
    "its", "of", "on", "or", "so", "that", "the", "their", "them", "they", "this", "to", "was", "we", "with",
    "you", "your", "our", "just", "also", "like", "got", "get", "did", "does", "do", "really", "very",
    "ngl", "im", "i",
}

PUNCT_PATTERN = re.compile(r"^[^\w]+$")
_WORD = re.compile(r"\b[0-9a-zA-Z][0-9a-zA-Z\-'’]+\b", re.I)


def _content_words(text: str) -> List[str]:
    out: List[str] = []
    for m in _WORD.finditer(text or ""):
        w = m.group(0)
        lo = w.lower()
        if lo in _STOP or len(lo) < 2:
            continue
        if PUNCT_PATTERN.match(w):
            continue
        if len(w) < 3 and lo not in ("hr", "ai", "uk", "us"):
            continue
        out.append(w)
    return out


def _rank_salient_words(text: str, top_k: int = 5) -> List[str]:
    cands = _content_words(text)
    if not cands and text:
        return [t for t in re.split(r"\s+", (text or "").strip()) if len(t) > 1][:top_k] or [text[:32].strip()]
    # Prefer longer, rarer by length; demote all-lowercase 3-4 char unless domain
    scored = []
    for w in cands:
        s = (len(w) * 0.3) + (2.0 if w[:1].isupper() else 0) + (1.0 if re.search(r"[A-Z]", w) else 0)
        if len(w) >= 6:
            s += 1.0
        scored.append((s, w))
    scored.sort(key=lambda x: -x[0])
    seen: Set[str] = set()
    out: List[str] = []
    for _, w in scored:
        k = w.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(w)
        if len(out) >= top_k:
            break
    if not out and text:
        t = re.sub(r"\s+", " ", (text or "").strip())
        if t:
            out = [t[:48] + "…" if len(t) > 48 else t]
    return out


def build_important_words(
    text: str,
    bias_type: str,
    heur: HeuristicSignals,
    p_hate: float,
    type_head,  # BiasTypeHead
    use_integrated_attribution: bool = True,
    top_k: int = 5,
) -> List[str]:
    words: List[str] = []
    if bias_type != NEUTRAL_LABEL and use_integrated_attribution:
        from explainability.attributions import explain_text

        try:
            raw = explain_text(type_head.model, type_head.tokenizer, text, bias_type, top_k=top_k)
        except Exception:
            raw = []
        words = list(raw) if raw else []
    for t in heur.match_terms:
        if t and t not in words:
            words.append(t)
    if p_hate >= 0.7 and (not words or p_hate >= 0.85):
        w = "toxic" if p_hate >= 0.78 else "hostile"
        if w not in [x.lower() for x in words]:
            words.append(w)
    for w in _rank_salient_words(text, top_k=top_k):
        if w.lower() not in [x.lower() for x in words]:
            words.append(w)
    final: List[str] = []
    seen2: set[str] = set()
    for w in words:
        k = w.lower()
        if k in seen2:
            continue
        seen2.add(k)
        final.append(w)
    if not final and text:
        w = re.sub(r"\s+", " ", text).strip()[:32]
        final = [w + "…"] if len(text) > 32 else [w]
    return final[: max(2, min(top_k, 6))]


def type_hint_keywords(bias_type: str) -> str:
    m = {
        "gender_bias": "men or women as groups",
        "nationality_bias": "immigrants, locals, or country-of-origin framing",
        "profession_bias": "roles or job categories",
    }
    return m.get(bias_type, "a protected group in this task")
