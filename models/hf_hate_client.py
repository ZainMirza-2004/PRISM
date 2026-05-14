"""RoBERTa hate signal: local-stable default plus optional HF Inference API."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
import warnings
from dataclasses import dataclass
from typing import List, Optional

# Binary hate classifier trained on large Twitter/English data (HATE / NOT HATE)
MODEL_ID = "cardiffnlp/twitter-roberta-base-hate-latest"

# When unset/false: no external API (stable offline calibration/train/eval).
# Set to 1/true to call Hugging Face for Cardiff hate RoBERTa.
_ENV_REMOTE_HATE = "HF_REMOTE_HATE"


@dataclass
class HateApiResult:
    label: str
    label_norm: str  # HATE or NOT_HATE
    scores: dict[str, float]  # normalized keys
    p_hate: float


def use_remote_hate_api() -> bool:
    """Return True iff Hub inference for hate should be attempted."""
    return os.environ.get(_ENV_REMOTE_HATE, "").strip().lower() in {"1", "true", "yes"}


def predict_hate_local(_text: str) -> HateApiResult:
    """Offline hate signal: no remote dependency.

    Uses p_hate=0 / NOT_HATE. Training evals showed RoBERTa margin rarely flips fused
    decisions vs this baseline; set ``HF_REMOTE_HATE=1`` to restore Hub calls.
    """
    return HateApiResult(
        label="NOT_HATE_LOCAL_FALLBACK",
        label_norm="NOT_HATE",
        scores={"HATE": 0.0, "NOT_HATE": 1.0},
        p_hate=0.0,
    )


def _norm_label(s: str) -> str:
    t = s.strip()
    lo = t.lower()
    if "hate" in lo and "not" in lo:
        return "NOT_HATE"
    if lo in ("neither", "not hate", "not_hate", "no_hate", "nohate"):
        return "NOT_HATE"
    if "hate" in lo:
        return "HATE"
    u = t.upper().replace(" ", "_").replace("-", "_")
    if u in ("LABEL_0", "0"):
        return "HATE"
    if u in ("LABEL_1", "1"):
        return "NOT_HATE"
    return t


def _flatten_raw_payload(raw) -> list:
    if raw is None:
        return []
    if isinstance(raw, list):
        if len(raw) == 0:
            return []
        # batch-of-one → nested list
        if isinstance(raw[0], list) and all(isinstance(x, dict) for x in raw[0]):
            return list(raw[0])
        if isinstance(raw[0], dict) and "label" in raw[0]:
            return list(raw)
    if isinstance(raw, dict) and "error" in raw:
        raise RuntimeError(raw.get("error", "Hugging Face API error"))
    if isinstance(raw, dict) and "label" in raw:
        return [raw]
    return []


def parse_hate_response(raw) -> HateApiResult:
    items = _flatten_raw_payload(raw)
    if not items:
        raise ValueError("Empty hate-classification response from API")

    by_norm: dict[str, float] = {}
    ph, pn = 0.0, 0.0
    for it in items:
        lab = str(it.get("label", ""))
        s = float(it.get("score", 0.0))
        l = lab.lower()
        if "hate" in l:
            if re.search(r"\bnot(\s+|-)hate\b", l) or re.search(r"\bno(\s+|-)hate\b", l) or l in ("nohate", "neither"):
                pn = max(pn, s)
            else:
                ph = max(ph, s)
        else:
            kn = _norm_label(lab)
            if kn == "HATE":
                ph = max(ph, s)
            elif kn == "NOT_HATE":
                pn = max(pn, s)

    if ph + pn > 1e-9:
        p_hate = float(ph / (ph + pn))
    else:
        p_hate = float(max(0.0, ph))

    by_norm = {"HATE": ph, "NOT_HATE": pn} if (ph or pn) else {}
    first = str(items[0].get("label", ""))
    lnorm = _norm_label(first)
    if lnorm in ("HATE", "NOT_HATE"):
        top = lnorm
    else:
        top = "HATE" if p_hate >= 0.5 else "NOT_HATE"
    return HateApiResult(
        label=first,
        label_norm=top,
        scores=by_norm,
        p_hate=p_hate,
    )


def _http_classify(text: str, model_id: str, token: Optional[str]) -> HateApiResult:
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    body = json.dumps({"inputs": text, "options": {"wait_for_model": True}}).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = json.loads(r.read().decode("utf-8"))
    return parse_hate_response(raw)


def _hf_hub_classify(text: str, model_id: str, token: Optional[str]) -> HateApiResult:
    from huggingface_hub import InferenceClient

    client = InferenceClient(token=token)
    if not hasattr(client, "text_classification"):
        raise AttributeError("InferenceClient has no text_classification")
    raw = client.text_classification(text, model=model_id)
    return parse_hate_response(raw)


def _resolve_token(explicit: Optional[str]) -> Optional[str]:
    v = (explicit or os.environ.get("HF_API_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or "").strip()
    return v or None


def classify_hate(
    text: str,
    *,
    model_id: str = MODEL_ID,
    token: Optional[str] = None,
) -> HateApiResult:
    if not use_remote_hate_api():
        return predict_hate_local(text)
    t = _resolve_token(token)
    err_hub = None
    try:
        return _hf_hub_classify(text, model_id, t)
    except Exception as e:  # noqa: BLE001
        err_hub = e
    try:
        return _http_classify(text, model_id, t)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if e.fp else str(e)
        warnings.warn(
            f"Hugging Face hate API HTTP {e.code}: {body[:400]!s}; Hub err: {err_hub!s}. Using predict_hate_local.",
            RuntimeWarning,
            stacklevel=2,
        )
        return predict_hate_local(text)
    except Exception as e:
        warnings.warn(
            f"Hugging Face hate API failed (Hub err: {err_hub!s}; HTTP err: {e!s}). Using predict_hate_local.",
            RuntimeWarning,
            stacklevel=2,
        )
        return predict_hate_local(text)


def classify_hate_batch(
    texts: List[str],
    *,
    model_id: str = MODEL_ID,
    token: Optional[str] = None,
) -> List[HateApiResult]:
    return [classify_hate(t, model_id=model_id, token=token) for t in texts]
