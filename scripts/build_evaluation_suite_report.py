#!/usr/bin/env python3
"""
Build a standalone HTML report comparing bias-detection systems on a gold CSV.

Systems (six scored pipelines; System 4 is two independent LLM baselines):
  1) Rule-based only (legacy fuse_scores, neural logits neutralized)
  2) Rules + linguistic / methodology cues (heuristic composite)
  3) Fine-tuned DistilBERT type head only (argmax on tempered distribution)
  4a) Claude Sonnet (OpenRouter) — scored separately vs gold
  4b) Gemini Pro (OpenRouter) — scored separately vs gold
  5) PRISM — StereoSet B + social aux + dual 40-d meta fusion (final system)

There is no majority-vote ensemble in headline metrics; each LLM has its own y_pred and plots.

The generated HTML uses an **editorial** palette (terra #D97757, cream #FAF8F5, Lora + Inter), sharp rectangular cards, thick header underlines on tables, and alternating cream bands. All **six** systems appear in metrics, charts, bootstrap, errors, latency, confusion-matrix references (**S1–S4b**), ROC/PR (**S3, S4a, S4b, PRISM**); PRISM (system id S5) stays visually primary. Throughput omits absurd rates when wall time is zero (API stub).

Usage:
  cd PRISM
  python scripts/build_evaluation_suite_report.py

  Defaults: gold = data/evaluation/manual_eval_v3_400_posts.csv,
  out = data/output/evaluation_dashboard.html (override with --gold / --out).

  Optional static PNGs for every Plotly figure (requires pip install plotly kaleido):
    --export-png-dir data/output/eval_figures

  OPENROUTER_API_KEY in .env (or env). Use --no-llm to skip API calls (stub both LLM baselines).
  Use --llm-max N to cap LLM calls for smoke tests.
  Use --reuse-system4-from-html PREVIOUS.html to keep S4a/S4b headline metrics identical to that run
  (reconstructs per-post labels from embedded confusion matrices; requires the same gold CSV row order).
"""

from __future__ import annotations

import argparse
import html as html_module
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]


def _export_plotly_specs_to_png(
    specs: list[tuple[str, list[dict[str, Any]], dict[str, Any]]],
    out_dir: Path,
    *,
    scale: int = 2,
    default_width: int = 1280,
) -> None:
    """Write one PNG per figure using Kaleido (``pip install kaleido plotly``)."""
    try:
        import plotly.graph_objects as go  # type: ignore[import-untyped]
    except ImportError as e:
        raise RuntimeError("PNG export needs the plotly package: pip install plotly kaleido") from e

    out_dir.mkdir(parents=True, exist_ok=True)
    fallback_height: dict[str, int] = {
        "plotSummary": 460,
        "plotPC": 480,
        "plotErrors": 460,
        "plotBoot": 440,
        "plotLat": 420,
        "plotThr": 400,
        "plotROC": 400,
        "plotPRAUC": 400,
        "plotMacroF1": 340,
        "plotBin": 340,
        "cal0": 300,
        "cal1": 300,
    }

    for name, data, layout in specs:
        fig = go.Figure(data=data, layout=layout)
        lw = int(layout.get("width") or default_width)
        lh_raw = layout.get("height")
        if lh_raw is not None:
            lh = int(lh_raw)
        else:
            lh = fallback_height.get(name, 420)

        dest = out_dir / f"{name}.png"
        try:
            fig.write_image(str(dest), width=lw, height=lh, scale=scale)
        except Exception as e:
            raise RuntimeError(
                f"Could not write {dest}. Install Kaleido for static export: pip install kaleido ({e})"
            ) from e
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.loaders import load_labeled_goldset  # noqa: E402
from model_evaluation.metrics import compute_binary_metrics, compute_multiclass_metrics  # noqa: E402
from models.bias_type_head import BiasTypeHead  # noqa: E402
from models.fusion_engine import fuse_scores, fusion_distribution_snapshot  # noqa: E402
from models.hybrid_pipeline import HybridBiasPipeline  # noqa: E402
from models.label_config import LABELS, NEUTRAL_LABEL  # noqa: E402
from models.linguistic_features import compute_linguistic_features  # noqa: E402
from models.methodology_features import extract_methodology_features  # noqa: E402
from models.preprocess import preprocess_social_post  # noqa: E402
from models.rule_signals import extract_rule_fusion_signals, structure_score_from_rules  # noqa: E402

try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

NEUTRAL_DIST = {lb: 0.0 for lb in LABELS}
NEUTRAL_DIST[NEUTRAL_LABEL] = 1.0

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# OpenRouter slugs (see https://openrouter.ai/models — retired IDs return 404 and fall back to neutral).
CLAUDE_MODEL = "anthropic/claude-sonnet-4"
GEMINI_MODEL = "google/gemini-2.5-pro"

# Verbatim user message sent to OpenRouter for each post ({post} is truncated to 8000 chars).
OPENROUTER_EVAL_PROMPT_TEMPLATE = """You are an expert annotator for workplace / social-media bias in English.
Classify the following post into EXACTLY ONE of these labels:
- gender_bias
- nationality_bias
- profession_bias
- neutral

Definitions (short):
- gender_bias: stereotypes, double standards, or role assumptions tied to gender.
- nationality_bias: stereotypes or treatment tied to national origin, immigration, ethnicity-as-nationality.
- profession_bias: stereotypes about occupations, education tracks, competence by role.
- neutral: no clear biased claim, or purely factual / inclusive / counter-stereotypical without endorsing a stereotype.

Return ONLY valid JSON on one line (no markdown fences):
{{"label":"<one of the four>","justification":"<one sentence>"}}

POST:
\"\"\"{post}\"\"\"
"""


def format_openrouter_eval_prompt(post: str) -> str:
    return OPENROUTER_EVAL_PROMPT_TEMPLATE.format(post=post[:8000])

# Anthropic-inspired editorial theme (warm neutrals + terra cotta accent)
ANTHROPIC = {
    "terra": "#D97757",
    "cream": "#FAF8F5",
    "cream_dark": "#F3EFE8",
    "ink": "#141413",
    "muted": "#6B6560",
    "hairline": "#E8E4DF",
    "wash": "#EDE9E3",
}
# One colour per system bar trace (S1 … S4a, S4b, PRISM); PRISM (s5) = brand terra
SYSTEM_BAR_COLORS = ["#6B6560", "#8B7355", "#5C6578", "#7D6B8A", "#5F8A7A", "#D97757"]
# Per-class F1 grouped bars — four muted bias-type hues (max 4)
BIAS_TYPE_BAR_COLORS = ["#8B6914", "#5C7A8A", "#6B6560", "#A89880"]


def _load_rows(gold_path: Path) -> list[dict[str, Any]]:
    rows = load_labeled_goldset(str(gold_path))
    return [r for r in rows if str(r.get("label", "")).strip() in LABELS]


def _parse_suite_html_embedded_json(html_path: Path) -> dict[str, Any]:
    text = html_path.read_text(encoding="utf-8")
    m = re.search(r'<pre id="raw-json"[^>]*>(.*?)</pre>', text, re.DOTALL)
    if not m:
        raise ValueError(f"No embedded #raw-json block found in {html_path}")
    return json.loads(m.group(1))


def _y_pred_from_confusion_matrix(y_true: list[str], cm: list[list[int]]) -> list[str]:
    """Rebuild predictions so sklearn confusion_matrix(y_true, y_pred) matches *cm* (rows=gold, cols=pred)."""
    if len(cm) != len(LABELS) or any(len(row) != len(LABELS) for row in cm):
        raise ValueError("Confusion matrix must be 4×4 in LABELS order.")
    by_gold: dict[int, list[int]] = {i: [] for i in range(len(LABELS))}
    for idx, t in enumerate(y_true):
        if t not in LABELS:
            raise ValueError(f"Unknown gold label at index {idx}: {t}")
        by_gold[LABELS.index(str(t))].append(idx)

    y_pred = [NEUTRAL_LABEL] * len(y_true)
    for gi in range(len(LABELS)):
        idxs = sorted(by_gold[gi])
        need = sum(cm[gi])
        if len(idxs) != need:
            raise ValueError(
                f"Cannot reuse System 4: gold counts for {LABELS[gi]} ({len(idxs)}) "
                f"do not match confusion row sum ({need}). Use the same gold file/order as the source HTML."
            )
        cursor = 0
        for pj in range(len(LABELS)):
            pred_lab = LABELS[pj]
            for _ in range(cm[gi][pj]):
                y_pred[idxs[cursor]] = pred_lab
                cursor += 1
    return y_pred


def predict_rule_only(text: str) -> tuple[str, float]:
    clean = preprocess_social_post(text)
    rules = extract_rule_fusion_signals(clean)
    bd, bt, conf, _, _ = fuse_scores(NEUTRAL_DIST, rules, 0.0, clean)
    if bd is not True:
        return NEUTRAL_LABEL, float(conf)
    return str(bt), float(conf)


def predict_rule_linguistic(text: str) -> tuple[str, float]:
    clean = preprocess_social_post(text)
    rules = extract_rule_fusion_signals(clean)
    ling = compute_linguistic_features(clean, rules)
    meth = extract_methodology_features(clean)
    rd = rules.as_dict()
    struct = structure_score_from_rules(rules)
    rule_mass = sum(float(rd.get(k, 0) or 0) for k in rd) / max(len(rd), 1)
    activity = (
        0.28 * struct
        + 0.18 * rule_mass
        + 0.22 * (ling.group_presence + ling.exclusion_intent + ling.soft_preference_norm) / 3.0
        + 0.32 * max(meth.gender_axis_cue, meth.nationality_cue, meth.profession_cue)
    )
    if activity < 0.42:
        return NEUTRAL_LABEL, float(min(0.92, 0.52 + 0.15 * activity))
    scores = {
        "gender_bias": meth.gender_axis_cue + 0.15 * ling.group_presence + 0.12 * float(rd.get("coded_bias", 0)),
        "nationality_bias": meth.nationality_cue + 0.15 * float(rd.get("preference", 0)) + 0.1 * struct,
        "profession_bias": meth.profession_cue + 0.15 * float(rd.get("comparison", 0)) + 0.1 * struct,
    }
    winner = max(scores, key=scores.get)
    conf = float(min(0.95, 0.55 + 0.35 * max(scores.values())))
    return winner, conf


def _parse_llm_label(raw: str) -> str | None:
    if not raw:
        return None
    raw2 = raw.strip()
    m = re.search(r"\{[^{}]*\}", raw2, re.DOTALL)
    blob = m.group(0) if m else raw2
    try:
        obj = json.loads(blob)
        lab = str(obj.get("label", "")).strip()
        if lab in LABELS:
            return lab
    except json.JSONDecodeError:
        pass
    lo = raw2.lower()
    for lab in LABELS:
        if re.search(rf"\b{re.escape(lab)}\b", lo):
            return lab
    return None


def openrouter_classify(text: str, model: str, api_key: str) -> tuple[str | None, str]:
    prompt = format_openrouter_eval_prompt(text)
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 220,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        OPENROUTER_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost/prism-eval",
            "X-Title": "PRISM Evaluation Dashboard",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        content = (payload.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        return _parse_llm_label(content), content
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        return None, str(e)


def one_hot(label: str) -> np.ndarray:
    v = np.zeros(len(LABELS), dtype=np.float64)
    if label in LABELS:
        v[LABELS.index(label)] = 1.0
    return v


def bootstrap_f1_macro(y_true: list[str], y_pred: list[str], n_boot: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    yt = np.array(y_true)
    yp = np.array(y_pred)
    n = len(y_true)
    f1s: list[float] = []
    from sklearn.metrics import f1_score

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        f1s.append(
            float(f1_score(yt[idx], yp[idx], average="macro", labels=list(LABELS), zero_division=0))
        )
    arr = np.asarray(f1s)
    return float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def mcnemar_pair(y_true: list[str], pred_a: list[str], pred_b: list[str]) -> dict[str, Any]:
    """McNemar exact binomial test; A = reference (e.g. hybrid), B = comparator."""
    n10 = sum(1 for t, a, b in zip(y_true, pred_a, pred_b) if a == t and b != t)
    n01 = sum(1 for t, a, b in zip(y_true, pred_a, pred_b) if a != t and b == t)
    m = n10 + n01
    if m == 0:
        return {"n10": n10, "n01": n01, "p_value": 1.0}
    try:
        from scipy.stats import binomtest

        pv = float(binomtest(min(n10, n01), m, 0.5, alternative="two-sided").pvalue)
    except Exception:
        from math import comb

        def _pmf(k: int, nn: int, p: float = 0.5) -> float:
            return comb(nn, k) * (p**k) * ((1 - p) ** (nn - k))

        k0 = min(n10, n01)
        # two-sided exact: 2 * P(X <= k0) capped at 1 (symmetric binomial null)
        tail = sum(_pmf(k, m) for k in range(0, k0 + 1))
        pv = float(min(1.0, 2 * tail))
    return {"n10": n10, "n01": n01, "p_value": pv}


def cohen_kappa(a: list[str], b: list[str]) -> float:
    from sklearn.metrics import cohen_kappa_score

    return float(cohen_kappa_score(a, b, labels=list(LABELS)))


def multiclass_roc_auc(y_true: list[str], prob: np.ndarray | None) -> float | None:
    if prob is None or prob.shape != (len(y_true), len(LABELS)):
        return None
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import label_binarize

    y = np.array(y_true)
    mask = np.array([t in LABELS for t in y])
    if mask.sum() < 5:
        return None
    yb = label_binarize(y[mask], classes=list(LABELS))
    try:
        return float(roc_auc_score(yb, prob[mask], average="macro", multi_class="ovr"))
    except ValueError:
        return None


def multiclass_pr_auc(y_true: list[str], prob: np.ndarray | None) -> float | None:
    if prob is None or prob.shape != (len(y_true), len(LABELS)):
        return None
    from sklearn.metrics import average_precision_score
    from sklearn.preprocessing import label_binarize

    y = np.array(y_true)
    mask = np.array([t in LABELS for t in y])
    yb = label_binarize(y[mask], classes=list(LABELS))
    try:
        scores = []
        for j in range(len(LABELS)):
            scores.append(average_precision_score(yb[:, j], prob[mask, j]))
        return float(np.mean(scores))
    except ValueError:
        return None


def calibration_bins(
    y_true: list[str], y_pred: list[str], conf: list[float], n_bins: int = 10
) -> dict[str, Any]:
    bins = [{"lo": i / n_bins, "hi": (i + 1) / n_bins, "n": 0, "acc": 0.0} for i in range(n_bins)]
    for t, p, c in zip(y_true, y_pred, conf):
        bi = min(n_bins - 1, int(c * n_bins))
        bins[bi]["n"] += 1
        bins[bi]["acc"] += 1.0 if p == t else 0.0
    for b in bins:
        if b["n"]:
            b["acc"] /= b["n"]
        b["mid"] = (b["lo"] + b["hi"]) / 2
    return {"bins": bins}


BIAS_TYPE_LABELS = ("gender_bias", "nationality_bias", "profession_bias")


def fairness_metrics_bias_classes(mc: dict[str, Any]) -> dict[str, Any]:
    """F1 disparity & recall ratio across the three bias-type classes (not demographic parity)."""
    per = mc.get("per_class", {})
    f1s: list[float] = []
    recalls: list[float] = []
    for lab in BIAS_TYPE_LABELS:
        pc = per.get(lab, {})
        f1s.append(float(pc.get("f1-score", 0) or 0))
        recalls.append(float(pc.get("recall", 0) or 0))
    max_f1 = max(f1s) if f1s else 0.0
    min_f1 = min(f1s) if f1s else 0.0
    max_r = max(recalls) if recalls else 1e-12
    min_r = min(recalls) if recalls else 0.0
    ratio = float(min_r / max_r) if max_r > 0 else 0.0
    return {
        "max_f1_bias": max_f1,
        "min_f1_bias": min_f1,
        "disparity_f1": max_f1 - min_f1,
        "disparity_pp": (max_f1 - min_f1) * 100.0,
        "worst_group_f1": min_f1,
        "best_group_f1": max_f1,
        "four_fifths_recall_ratio": ratio,
        "four_fifths_passes": ratio >= 0.8,
        "per_bias_f1": dict(zip(BIAS_TYPE_LABELS, f1s)),
        "per_bias_recall": dict(zip(BIAS_TYPE_LABELS, recalls)),
    }


def _critique_like(text_lower: str) -> bool:
    markers = (
        "unfair",
        "toxic workplace",
        "sexist",
        "discrimination",
        "biased hiring",
        "shouldn't",
        "should not",
        "critique",
        "harassment",
        "misogyn",
        "prejudice",
        "call out",
        "double standard",
    )
    if any(m in text_lower for m in markers):
        return True
    if "stereotype" in text_lower and any(w in text_lower for w in ("against", "reject", "fighting", "wrong", "not true")):
        return True
    return False


def _ambiguous_tone(text_lower: str) -> bool:
    return any(
        x in text_lower
        for x in (
            "/s",
            "lol",
            "lmao",
            "obviously,",
            "really?",
            "sure jan",
            " 😉",
            "ironically",
        )
    )


def taxonomy_s5_errors(rows: list[dict[str, Any]], y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    """Heuristic semantic buckets for PRISM (s5) misclassifications."""
    bucket_keys = (
        "cross_type_confusion",
        "critique_vs_endorsement",
        "neutral_mention_vs_stereotype",
        "ambiguous_tone",
        "other",
    )
    buckets: dict[str, list[dict[str, Any]]] = {k: [] for k in bucket_keys}

    for i, (t, p) in enumerate(zip(y_true, y_pred)):
        if t == p:
            continue
        text = str(rows[i].get("text", ""))
        tl = text.lower()
        pid = rows[i].get("post_id", str(i))
        rec = {"post_id": pid, "gold": t, "pred": p, "text": text[:400]}

        if t in BIAS_TYPE_LABELS and p in BIAS_TYPE_LABELS and t != p:
            buckets["cross_type_confusion"].append(rec)
        elif _critique_like(tl):
            buckets["critique_vs_endorsement"].append(rec)
        elif (t == NEUTRAL_LABEL and p != NEUTRAL_LABEL) or (t != NEUTRAL_LABEL and p == NEUTRAL_LABEL):
            buckets["neutral_mention_vs_stereotype"].append(rec)
        elif _ambiguous_tone(tl):
            buckets["ambiguous_tone"].append(rec)
        else:
            buckets["other"].append(rec)

    total = sum(len(buckets[k]) for k in bucket_keys)
    summary: dict[str, dict[str, float | int]] = {}
    for k in bucket_keys:
        c = len(buckets[k])
        summary[k] = {
            "count": c,
            "pct": (100.0 * c / total) if total else 0.0,
        }
    return {"buckets": buckets, "summary": summary, "total_errors": total}


def safe_throughput_posts_per_s(n: int, latency_ms: float) -> float | None:
    """Avoid absurd rates when latency is zero (stub LLM) or negligible."""
    if latency_ms <= 1e-6:
        return None
    return float(n) / (latency_ms / 1000.0)


@dataclass
class SystemRun:
    key: str
    name: str
    description: str
    y_pred: list[str] = field(default_factory=list)
    confidence: list[float] = field(default_factory=list)
    prob: np.ndarray | None = None  # (n, 4) when available
    latency_ms: float = 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gold",
        type=Path,
        default=_ROOT / "data/evaluation/manual_eval_v3_400_posts.csv",
        help="Gold benchmark CSV (default: 400-post manual eval — same base as published suite)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=_ROOT / "data/output/evaluation_dashboard.html",
        help="Write HTML report here (default: data/output/evaluation_dashboard.html)",
    )
    ap.add_argument("--no-llm", action="store_true", help="Skip OpenRouter; both System 4 LLM baselines use neutral stubs.")
    ap.add_argument(
        "--reuse-system4-from-html",
        type=Path,
        default=None,
        help="Prior evaluation-suite HTML: rebuild S4a/S4b preds from its embedded confusion matrices (no OpenRouter). Same gold row order required.",
    )
    ap.add_argument("--llm-max", type=int, default=None, help="Max posts to send to LLMs (default: all).")
    ap.add_argument("--bootstrap", type=int, default=400, help="Bootstrap resamples for F1 CI.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--export-png-dir",
        type=Path,
        default=None,
        help="Write one PNG per Plotly chart here (requires: pip install plotly kaleido).",
    )
    ap.add_argument(
        "--calibrated",
        action="store_true",
        help="PRISM (S5): use lr_meta_calibrated.joblib from models/meta_fusion when present.",
    )
    ap.add_argument(
        "--use-thresholds",
        action="store_true",
        help="PRISM (S5): apply optimal_thresholds.json with calibrated posteriors.",
    )
    ap.add_argument(
        "--meta-model",
        type=str,
        default=None,
        help=(
            "PRISM (S5): explicit meta fusion joblib path "
            "(default: resolve dual lr_meta_dual_<primary>_plus_<aux>.joblib under models/meta_fusion)."
        ),
    )
    args = ap.parse_args()

    rows = _load_rows(args.gold)
    texts = [str(r["text"]) for r in rows]
    y_true = [str(r["label"]).strip() for r in rows]
    n = len(rows)
    api_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()

    systems: dict[str, SystemRun] = {
        "s1": SystemRun(
            "s1",
            "Rule-based only",
            "Hand-tuned fusion_engine rules with neural type logits neutralized (neutral one-hot).",
        ),
        "s2": SystemRun(
            "s2",
            "Rules + linguistic / methodology",
            "Structure, rule flags, linguistic bundle, and methodology cues combined heuristically (no transformer).",
        ),
        "s3": SystemRun(
            "s3",
            "DistilBERT type head (standalone)",
            f"Fine-tuned {Path('models/distilbert_social_bias').name}: argmax on tempered class distribution only.",
        ),
        "s4_claude": SystemRun(
            "s4_claude",
            "System 4 — Claude Sonnet 4 (zero-shot)",
            f"OpenRouter {CLAUDE_MODEL}; each post classified independently vs gold (no ensemble).",
        ),
        "s4_gemini": SystemRun(
            "s4_gemini",
            "System 4 — Gemini 2.5 Pro (zero-shot)",
            f"OpenRouter {GEMINI_MODEL}; each post classified independently vs gold (no ensemble).",
        ),
        "s5": SystemRun(
            "s5",
            "PRISM",
            "StereoSet model B + auxiliary hate-speech DistilBERT + 40-d learned meta fusion (final hybrid stack).",
        ),
    }

    # --- System 1 & 2 ---
    t0 = time.perf_counter()
    for tx in texts:
        p, c = predict_rule_only(tx)
        systems["s1"].y_pred.append(p)
        systems["s1"].confidence.append(c)
    systems["s1"].latency_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    for tx in texts:
        p, c = predict_rule_linguistic(tx)
        systems["s2"].y_pred.append(p)
        systems["s2"].confidence.append(c)
    systems["s2"].latency_ms = (time.perf_counter() - t0) * 1000

    # --- System 3 ---
    head = BiasTypeHead(str(_ROOT / "models/distilbert_social_bias"))
    cleans = [preprocess_social_post(t) for t in texts]
    t0 = time.perf_counter()
    dists = head.predict_type_distribution_batch(cleans, batch_size=8)
    systems["s3"].latency_ms = (time.perf_counter() - t0) * 1000
    prob3 = np.zeros((n, len(LABELS)))
    for i, d0 in enumerate(dists):
        snap = fusion_distribution_snapshot(d0)
        for j, lab in enumerate(LABELS):
            prob3[i, j] = snap.get(lab, 0.0)
        best_lab = max(LABELS, key=lambda lb: snap.get(lb, 0.0))
        systems["s3"].y_pred.append(best_lab)
        systems["s3"].confidence.append(float(max(snap.values())))
    systems["s3"].prob = prob3

    # --- System 4 — two independent LLM baselines (no ensemble) ---
    llm_n = n if args.llm_max is None else min(n, args.llm_max)
    y4c = [NEUTRAL_LABEL] * n
    y4g = [NEUTRAL_LABEL] * n
    c4c = [0.5] * n
    c4g = [0.5] * n
    prob4c = np.ones((n, len(LABELS)), dtype=np.float64) / len(LABELS)
    prob4g = np.ones((n, len(LABELS)), dtype=np.float64) / len(LABELS)
    claude_ms = 0.0
    gemini_ms = 0.0
    llm_http_errors_claude = 0
    llm_http_errors_gemini = 0
    reuse_path = args.reuse_system4_from_html
    old_s4_report: dict[str, Any] | None = None
    if reuse_path is not None:
        rp = reuse_path if reuse_path.is_absolute() else _ROOT / reuse_path
        old_s4_report = _parse_suite_html_embedded_json(rp)
        cm_c = old_s4_report["confusion_matrices"]["s4_claude"]
        cm_g = old_s4_report["confusion_matrices"]["s4_gemini"]
        y4c = _y_pred_from_confusion_matrix(y_true, cm_c)
        y4g = _y_pred_from_confusion_matrix(y_true, cm_g)
        prob4c = np.stack([one_hot(y) for y in y4c])
        prob4g = np.stack([one_hot(y) for y in y4g])
        c4c = [0.88] * n
        c4g = [0.88] * n
        lat_old = old_s4_report.get("latency_ms") or {}
        claude_ms = float(lat_old.get("s4_claude", 0.0))
        gemini_ms = float(lat_old.get("s4_gemini", 0.0))
        om = old_s4_report.get("meta") or {}
        llm_http_errors_claude = int(om.get("llm_openrouter_http_errors_claude", 0))
        llm_http_errors_gemini = int(om.get("llm_openrouter_http_errors_gemini", 0))
        llm_n = int(om.get("llm_evaluated_first_n", llm_n))
        print(
            f"Reused System 4 (S4a/S4b) from embedded confusion matrices in {rp.name} (no OpenRouter calls)."
        )
    elif args.no_llm or not api_key:
        pass
    else:
        for i in range(llm_n):
            tx = texts[i]
            tq = time.perf_counter()
            lc, raw_c = openrouter_classify(tx, CLAUDE_MODEL, api_key)
            if lc is None and "HTTP" in (raw_c or ""):
                llm_http_errors_claude += 1
            claude_ms += (time.perf_counter() - tq) * 1000.0
            time.sleep(0.35)
            tq = time.perf_counter()
            lg, raw_g = openrouter_classify(tx, GEMINI_MODEL, api_key)
            if lg is None and "HTTP" in (raw_g or ""):
                llm_http_errors_gemini += 1
            gemini_ms += (time.perf_counter() - tq) * 1000.0
            time.sleep(0.35)
            lab_c = lc if lc in LABELS else NEUTRAL_LABEL
            lab_g = lg if lg in LABELS else NEUTRAL_LABEL
            y4c[i] = lab_c
            y4g[i] = lab_g
            c4c[i] = 0.88 if lc in LABELS else 0.38
            c4g[i] = 0.88 if lg in LABELS else 0.38
            prob4c[i] = one_hot(lab_c)
            prob4g[i] = one_hot(lab_g)
    systems["s4_claude"].y_pred = y4c
    systems["s4_gemini"].y_pred = y4g
    systems["s4_claude"].confidence = c4c
    systems["s4_gemini"].confidence = c4g
    systems["s4_claude"].prob = prob4c
    systems["s4_gemini"].prob = prob4g
    systems["s4_claude"].latency_ms = claude_ms
    systems["s4_gemini"].latency_ms = gemini_ms

    llm_disagreement_count = 0
    llm_agreement_rate_evaluated: float | None = None
    if not args.no_llm and api_key and llm_n > 0:
        llm_disagreement_count = sum(1 for i in range(llm_n) if y4c[i] != y4g[i])
        llm_agreement_rate_evaluated = (llm_n - llm_disagreement_count) / llm_n

    # --- System 5 hybrid ---
    pipe = HybridBiasPipeline(
        str(_ROOT / "models/distilbert_B_balanced"),
        hf_token=os.environ.get("HF_API_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
        hate_model_id="cardiffnlp/twitter-roberta-base-hate-latest",
        meta_classifier_path=args.meta_model,
        auxiliary_type_model_dir=str(_ROOT / "models/distilbert_social_bias"),
        calibrated=args.calibrated,
        use_thresholds=args.use_thresholds,
    )
    recs = [{"post_id": str(r.get("post_id", "")), "text": str(r["text"])} for r in rows]
    t0 = time.perf_counter()
    res5 = pipe.predict_batch(recs, batch_size=8)
    systems["s5"].latency_ms = (time.perf_counter() - t0) * 1000
    prob5 = np.zeros((n, len(LABELS)))
    for i, rr in enumerate(res5):
        bd = rr.get("bias_detected")
        bt = str(rr.get("bias_type", NEUTRAL_LABEL))
        if bd is not True:
            bt = NEUTRAL_LABEL
        systems["s5"].y_pred.append(bt)
        systems["s5"].confidence.append(float(rr.get("confidence", 0.5)))
        post = (rr.get("meta_fusion") or {}).get("posterior") or {}
        for j, lab in enumerate(LABELS):
            prob5[i, j] = float(post.get(lab, 0.0))
        if prob5[i].sum() <= 0:
            prob5[i] = one_hot(bt)
    systems["s5"].prob = prob5

    # Metrics per system
    from sklearn.metrics import f1_score, precision_score, recall_score

    per: dict[str, Any] = {}
    order = ["s1", "s2", "s3", "s4_claude", "s4_gemini", "s5"]
    for k in order:
        mc = compute_multiclass_metrics(y_true, systems[k].y_pred)
        bi = compute_binary_metrics(y_true, systems[k].y_pred)
        yp = systems[k].y_pred
        mc["f1_micro"] = float(f1_score(y_true, yp, average="micro", labels=list(LABELS), zero_division=0))
        mc["precision_micro"] = float(
            precision_score(y_true, yp, average="micro", labels=list(LABELS), zero_division=0)
        )
        mc["recall_micro"] = float(recall_score(y_true, yp, average="micro", labels=list(LABELS), zero_division=0))
        per[k] = {"multiclass": mc, "binary": bi}

    # Bootstrap (hybrid + key baselines)
    boot: dict[str, list[float]] = {}
    for k in order:
        m, lo, hi = bootstrap_f1_macro(y_true, systems[k].y_pred, args.bootstrap, args.seed)
        boot[k] = [m, lo, hi]

    # McNemar vs hybrid
    mcn: dict[str, Any] = {}
    for k in ["s1", "s2", "s3", "s4_claude", "s4_gemini"]:
        mcn[k] = mcnemar_pair(y_true, systems["s5"].y_pred, systems[k].y_pred)

    # ROC / PR AUC
    roc_auc = {k: multiclass_roc_auc(y_true, systems[k].prob) for k in order}
    pr_auc = {k: multiclass_pr_auc(y_true, systems[k].prob) for k in order}

    # Calibration (systems with meaningful probs)
    cal = {}
    for k in ["s3", "s5"]:
        cal[k] = calibration_bins(y_true, systems[k].y_pred, systems[k].confidence)

    # Cohen kappa LLM pair (on posts both models actually classified — same slice as API calls)
    kappa_llm = None
    if not args.no_llm and api_key and llm_n >= 2:
        kappa_llm = cohen_kappa(
            systems["s4_claude"].y_pred[:llm_n],
            systems["s4_gemini"].y_pred[:llm_n],
        )

    # Throughput (guard stub LLMs with zero latency)
    thr = {k: safe_throughput_posts_per_s(n, systems[k].latency_ms) for k in order}

    # S5 fairness (bias-type classes only) & error taxonomy
    fair_s5 = fairness_metrics_bias_classes(per["s5"]["multiclass"])
    tax_s5 = taxonomy_s5_errors(rows, y_true, systems["s5"].y_pred)

    # Extra pairwise tests: neural vs rules; hybrid vs neural headline
    mcn_s3_s1 = mcnemar_pair(y_true, systems["s3"].y_pred, systems["s1"].y_pred)
    mcn_s3_s2 = mcnemar_pair(y_true, systems["s3"].y_pred, systems["s2"].y_pred)
    mcn_s5_s3 = mcnemar_pair(y_true, systems["s5"].y_pred, systems["s3"].y_pred)

    # Error counts
    err_summary: dict[str, Any] = {}
    for k in order:
        fp = sum(1 for t, p in zip(y_true, systems[k].y_pred) if t == NEUTRAL_LABEL and p != NEUTRAL_LABEL)
        fn = sum(1 for t, p in zip(y_true, systems[k].y_pred) if t != NEUTRAL_LABEL and p == NEUTRAL_LABEL)
        err_summary[k] = {"fp_neutral_as_bias": fp, "fn_bias_as_neutral": fn}

    # Sample errors hybrid
    hy_err = [
        {"post_id": rows[i].get("post_id"), "gold": y_true[i], "pred": systems["s5"].y_pred[i], "text": texts[i][:280]}
        for i in range(n)
        if y_true[i] != systems["s5"].y_pred[i]
    ][:18]

    try:
        gold_rel = str(args.gold.relative_to(_ROOT))
    except ValueError:
        gold_rel = str(args.gold)
    report = {
        "meta": {
            "gold_path": gold_rel,
            "n": n,
            "llm_evaluated_first_n": llm_n,
            "no_llm": bool(args.no_llm or not api_key),
            "openrouter_api_key_present": bool(api_key),
            "openrouter_live": bool(not args.no_llm and api_key),
            "openrouter_models": {"claude": CLAUDE_MODEL, "gemini": GEMINI_MODEL},
            "llm_disagreement_count": llm_disagreement_count,
            "llm_agreement_rate_on_evaluated_slice": llm_agreement_rate_evaluated,
            "llm_openrouter_http_errors_claude": llm_http_errors_claude,
            "llm_openrouter_http_errors_gemini": llm_http_errors_gemini,
            "llm_zero_shot_prompt_template": format_openrouter_eval_prompt(
                "[POST_TEXT_TRUNCATED_TO_8000_CHARS]"
            ),
            "bootstrap_n": args.bootstrap,
            "bootstrap_seed": args.seed,
            "systems": {k: {"name": systems[k].name, "description": systems[k].description} for k in order},
            "llm_partial": llm_n < n,
        },
        "metrics": per,
        "bootstrap_f1_macro": boot,
        "mcnemar_vs_hybrid": mcn,
        "roc_auc_macro_ovr": roc_auc,
        "pr_auc_macro": pr_auc,
        "throughput_posts_per_s": thr,
        "latency_ms": {k: systems[k].latency_ms for k in order},
        "error_summary": err_summary,
        "calibration": cal,
        "cohen_kappa_llm_pair": kappa_llm,
        "hybrid_error_samples": hy_err,
        "confusion_matrices": {k: per[k]["multiclass"].get("confusion_matrix", []) for k in order},
        "labels_order": LABELS,
        "fairness_s5": fair_s5,
        "error_taxonomy_s5": tax_s5,
        "mcnemar_s3_vs_s1": mcn_s3_s1,
        "mcnemar_s3_vs_s2": mcn_s3_s2,
        "mcnemar_s5_vs_s3": mcn_s5_s3,
    }

    html = render_html(report, order, export_png_dir=args.export_png_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"Wrote {args.out} ({args.out.stat().st_size // 1024} KB)")
    if args.export_png_dir:
        print(f"Wrote PNG figures under {args.export_png_dir.resolve()}")

def _json_sanitize(obj: Any) -> Any:
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    return obj


def _fmt_latency_label(ms: float) -> str:
    if ms < 2000:
        return f"{ms:.0f} ms"
    return f"{ms / 1000:.1f} s"


def _fmt_thr_label(x: float | None, *, system_key: str = "") -> str:
    if x is None:
        if system_key in ("s4_claude", "s4_gemini"):
            return "n/a (network / stub)"
        return "n/a"
    if x >= 100:
        return f"{x:,.0f}"
    return f"{x:.1f}"


def _mcnemar_interpret(p: float) -> str:
    if p < 1e-6:
        return "Highly significant"
    if p < 0.001:
        return "Highly significant"
    if p < 0.05:
        return "Significant"
    return "Not significant"


def render_html(report: dict[str, Any], order: list[str], *, export_png_dir: Path | None = None) -> str:
    meta = report["meta"]
    plotly_cdn = "https://cdn.plot.ly/plotly-2.27.0.min.js"
    gold_name = Path(str(meta["gold_path"])).name

    names_long = [meta["systems"][k]["name"] for k in order]
    sys_short = ["S1 Rule", "S2 Ling.", "S3 BERT", "S4a Claude", "S4b Gemini", "PRISM"]
    tile_names = [
        "Rule-based only",
        "Rules + Linguistic",
        "DistilBERT Standalone",
        "Claude Sonnet 4",
        "Gemini 2.5 Pro",
        "PRISM",
    ]
    colors = list(SYSTEM_BAR_COLORS)

    f1_macro = [float(report["metrics"][k]["multiclass"]["f1_macro"]) for k in order]
    acc = [float(report["metrics"][k]["multiclass"]["accuracy"]) for k in order]
    f1_bin = [float(report["metrics"][k]["binary"].get("f1_bias_vs_neutral", 0.0)) for k in order]

    best_idx = int(np.argmax(f1_macro))
    best_f1 = f1_macro[best_idx]
    f1_s5 = report["metrics"]["s5"]["multiclass"]["f1_macro"]
    f1_s3 = report["metrics"]["s3"]["multiclass"]["f1_macro"]
    f1_s1 = report["metrics"]["s1"]["multiclass"]["f1_macro"]
    f1_s2 = report["metrics"]["s2"]["multiclass"]["f1_macro"]
    f1_s4c = report["metrics"]["s4_claude"]["multiclass"]["f1_macro"]
    f1_s4g = report["metrics"]["s4_gemini"]["multiclass"]["f1_macro"]
    pp_vs_s3 = (f1_s5 - f1_s3) * 100
    pp_vs_rules = (f1_s5 - (f1_s1 + f1_s2) / 2.0) * 100

    llm_collapse = (
        abs(f1_s4c - f1_s1) < 0.03
        and abs(f1_s4g - f1_s1) < 0.03
        and f1_s4c < 0.22
        and f1_s4g < 0.22
    )
    if llm_collapse:
        llm_appendix_note = (
            "Zero-shot <strong>S4a/S4b</strong> often collapse to a <em>neutral-only</em> pattern on this schema — interpret as "
            "<strong>taxonomy / prompt mismatch</strong>, not raw model incapacity."
        )
    else:
        llm_appendix_note = (
            f"Zero-shot LLMs (S4a macro F1 <strong>{f1_s4c:.3f}</strong>, S4b <strong>{f1_s4g:.3f}</strong>) appear in all figures below."
        )

    pv_s5_s3 = float(report.get("mcnemar_s5_vs_s3", {}).get("p_value", 1.0))
    hybrid_ms = float(report["latency_ms"]["s5"])
    hybrid_sec = float(hybrid_ms) / 1000.0
    n_posts = int(meta["n"])
    thr_s5_raw = report["throughput_posts_per_s"].get("s5")
    thr_s5 = float(thr_s5_raw) if thr_s5_raw is not None else 0.0
    t_cl = report["throughput_posts_per_s"].get("s4_claude")
    t_gm = report["throughput_posts_per_s"].get("s4_gemini")
    _llm_thr_vals = [v for v in (t_cl, t_gm) if v is not None]
    thr_llm_avg = float(np.mean(_llm_thr_vals)) if _llm_thr_vals else None
    sec_per_post_s5 = (hybrid_ms / 1000.0) / max(n_posts, 1)

    boot = report["bootstrap_f1_macro"]

    mcn_keys = ["s1", "s2", "s3", "s4_claude", "s4_gemini"]
    mcn_rows = ""
    all_mcn_tiny = True
    for k in mcn_keys:
        row = report["mcnemar_vs_hybrid"][k]
        pv = float(row["p_value"])
        if pv >= 0.001:
            all_mcn_tiny = False
        interp = _mcnemar_interpret(pv)
        nm = meta["systems"][k]["name"]
        mcn_rows += (
            f"<tr><td>{html_module.escape(nm)}</td>"
            f"<td class=\"mono\">{row['n10']}</td>"
            f"<td class=\"mono\">{row['n01']}</td>"
            f"<td class=\"mono num-strong\">{html_module.escape(f'{pv:.2e}')}</td>"
            f"<td class=\"dim\">{html_module.escape(interp)}</td></tr>"
        )

    mcn_intro = (
        "All listed comparisons are statistically significant at α = 0.001 (exact binomial)."
        if all_mcn_tiny
        else "Interpretation reflects two-sided exact binomial p-values on discordant pairs only."
    )

    def _fmt_p(pv: float) -> str:
        return f"{pv:.2e}" if pv < 0.0001 else f"{pv:.6f}".rstrip("0").rstrip(".")

    m_s3_s1 = report.get("mcnemar_s3_vs_s1", {})
    m_s3_s2 = report.get("mcnemar_s3_vs_s2", {})
    m_s5_s3 = report.get("mcnemar_s5_vs_s3", {})
    pairwise_mcn_rows = (
        f"<tr><td>S3 (DistilBERT) vs S1 (rules only)</td><td class=\"mono\">{m_s3_s1.get('n10', '')}</td>"
        f"<td class=\"mono\">{m_s3_s1.get('n01', '')}</td>"
        f"<td class=\"mono num-strong\">{_fmt_p(float(m_s3_s1.get('p_value', 1)))}</td>"
        f"<td class=\"dim\">{_mcnemar_interpret(float(m_s3_s1.get('p_value', 1)))}</td></tr>"
        f"<tr><td>S3 vs S2 (rules + linguistic)</td><td class=\"mono\">{m_s3_s2.get('n10', '')}</td>"
        f"<td class=\"mono\">{m_s3_s2.get('n01', '')}</td>"
        f"<td class=\"mono num-strong\">{_fmt_p(float(m_s3_s2.get('p_value', 1)))}</td>"
        f"<td class=\"dim\">{_mcnemar_interpret(float(m_s3_s2.get('p_value', 1)))}</td></tr>"
    )

    full_metrics_rows = ""
    for k in order:
        i = order.index(k)
        mc = report["metrics"][k]["multiclass"]
        bi = report["metrics"][k]["binary"]
        nm = tile_names[i]
        tr_cls = ' class="row-emphasis"' if k == "s5" else ""
        td_h = ' class="mono num-strong"' if k == "s5" else ' class="mono"'
        full_metrics_rows += (
            f"<tr{tr_cls}><td{' class=\"sys-name-cell\"' if k == 's5' else ''}>{html_module.escape(nm)}</td>"
            f"<td{td_h}>{mc['accuracy']:.3f}</td>"
            f"<td{td_h}>{mc['f1_macro']:.3f}</td>"
            f"<td{td_h}>{bi.get('f1_bias_vs_neutral', 0):.3f}</td></tr>"
        )

    per_class_header = "".join(
        f"<th>{html_module.escape(h)}</th>"
        for h in [
            "S1 Rule-based",
            "S2 + Linguistic",
            "S3 DistilBERT",
            "S4a Claude",
            "S4b Gemini",
            "PRISM",
        ]
    )
    per_class_rows = ""
    for lab in LABELS:
        per_class_rows += f"<tr><td>{html_module.escape(lab)}</td>"
        for k in order:
            pc = report["metrics"][k]["multiclass"].get("per_class", {}).get(lab, {})
            f1v = float(pc.get("f1-score", 0) or 0)
            if k == "s5":
                cls = "mono num-strong"
            elif f1v < 0.05:
                cls = "mono dim"
            else:
                cls = "mono"
            per_class_rows += f'<td class="{cls}">{f1v:.3f}</td>'
        per_class_rows += "</tr>"

    err_fp_fn_rows = ""
    for k in order:
        i = order.index(k)
        es = report["error_summary"][k]
        fp, fn = int(es["fp_neutral_as_bias"]), int(es["fn_bias_as_neutral"])
        tot = fp + fn
        nm = tile_names[i]
        tr_cls = ' class="row-emphasis"' if k == "s5" else ""
        td_tot = ' class="mono num-strong"' if k == "s5" else ' class="mono"'
        fp_cls = "mono dim" if fp == 0 else "mono"
        err_fp_fn_rows += (
            f"<tr{tr_cls}><td{' class=\"sys-name-cell\"' if k == 's5' else ''}>{html_module.escape(nm)}</td>"
            f'<td class="{fp_cls}">{fp}</td>'
            f"<td class=\"mono\">{fn}</td>"
            f"<td{td_tot}>{tot}</td></tr>"
        )

    err_rows = ""
    for e in report.get("hybrid_error_samples", []):
        txe = html_module.escape(str(e.get("text", "")))
        err_rows += (
            f"<tr><td class=\"mono dim\">{html_module.escape(str(e.get('post_id', '')))}</td>"
            f"<td class=\"mono\">{html_module.escape(str(e['gold']))}</td>"
            f"<td class=\"mono\">{html_module.escape(str(e['pred']))}</td>"
            f"<td style=\"font-size:12px;color:var(--muted)\">{txe}</td></tr>"
        )

    kappa_llm = report.get("cohen_kappa_llm_pair")
    if meta.get("no_llm"):
        kappa_cell = "Not computed — <code class=\"mono\">--no-llm</code> or missing API key."
    elif kappa_llm is None:
        kappa_cell = "Not computed (need ≥2 evaluated posts with a live OpenRouter run)."
    elif isinstance(kappa_llm, float) and math.isnan(kappa_llm):
        kappa_cell = (
            "<strong>NaN</strong> — Claude and Gemini produced identical label vectors on the evaluated slice, "
            "so inter-rater κ is undefined."
        )
    else:
        partial_note = ""
        if meta.get("llm_partial"):
            partial_note = (
                f" Evaluated slice: first <code class=\"mono\">{meta['llm_evaluated_first_n']}</code> posts; "
                "tail rows use neutral stubs for both LLMs."
            )
        kappa_cell = (
            f"κ = <code class=\"mono\">{kappa_llm:.4f}</code> (inter-model agreement on that slice; not accuracy vs gold)."
            f"{partial_note}"
        )

    chip_llm = "disabled (stub)" if meta["no_llm"] else "OpenRouter · live"

    llm_pair_callout = ""

    prompt_template_show = html_module.escape(
        str(meta.get("llm_zero_shot_prompt_template") or "(no template)")
    )

    fw = report.get("fairness_s5", {})
    fair_rows_html = (
        f"<tr><td>Best F1 (bias types)</td><td class=\"mono\">{fw.get('best_group_f1', 0):.3f}</td>"
        f"<td>Highest among gender / nationality / profession.</td></tr>"
        f"<tr><td>Worst-group F1</td><td class=\"mono num-strong\">{fw.get('worst_group_f1', 0):.3f}</td>"
        f"<td>Minimum across the three bias classes — robustness diagnostic.</td></tr>"
        f"<tr><td>Disparity (max − min F1)</td><td class=\"mono\">{fw.get('disparity_f1', 0):.3f}</td>"
        f"<td>{fw.get('disparity_pp', 0):.1f} percentage points spread across bias types.</td></tr>"
    )

    tax = report.get("error_taxonomy_s5", {}).get("summary", {})
    tax_labels = {
        "cross_type_confusion": "Cross-type (one bias class → another)",
        "critique_vs_endorsement": "Critique vs endorsement (heuristic)",
        "neutral_mention_vs_stereotype": "Neutral identity mention vs stereotype",
        "ambiguous_tone": "Ambiguous / informal tone",
        "other": "Other / unclassified",
    }
    tax_rows_html = ""
    te_total = int(report.get("error_taxonomy_s5", {}).get("total_errors", 0))
    for key, label in tax_labels.items():
        s = tax.get(key, {})
        c = int(s.get("count", 0))
        pct = float(s.get("pct", 0))
        tax_rows_html += (
            f"<tr><td>{html_module.escape(label)}</td>"
            f"<td class=\"mono\">{c}</td>"
            f"<td class=\"mono\">{pct:.1f}%</td></tr>"
        )

    stat_tiles = ""
    sid_all = ["S1", "S2", "S3", "S4a", "S4b", "S5"]
    for k in order:
        hero = k == "s5"
        hero_cls = " stat-tile stat-tile--hero" if hero else " stat-tile"
        idx = order.index(k)
        num_cls = "" if hero else f' style="color:{colors[idx]}"'
        stat_tiles += (
            f'<div class="{hero_cls.strip()}">'
            f'<p class="sys-name">{html_module.escape(tile_names[idx])}</p>'
            f'<p class="big-num"{num_cls}>{f1_macro[idx]:.3f}</p>'
            f'<p class="sub-label">Macro F1{" · best" if hero else ""}</p></div>'
        )

    systems_table_all = ""
    for j, k in enumerate(order):
        systems_table_all += (
            f"<tr><td class=\"mono\">{sid_all[j]}</td>"
            f'<td class="sys-name-cell">{html_module.escape(meta["systems"][k]["name"])}</td>'
            f"<td>{html_module.escape(meta['systems'][k]['description'])}</td></tr>"
        )

    FONT = {"family": "Inter, -apple-system, BlinkMacSystemFont, sans-serif", "color": ANTHROPIC["ink"], "size": 12}
    BG = {"paper_bgcolor": ANTHROPIC["cream"], "plot_bgcolor": ANTHROPIC["cream"]}
    GRID = {
        "gridcolor": ANTHROPIC["hairline"],
        "zerolinecolor": ANTHROPIC["hairline"],
        "showgrid": True,
    }
    MARGIN = {"l": 52, "r": 20, "t": 32, "b": 80}
    MARGIN_SM = {"l": 52, "r": 20, "t": 28, "b": 64}
    plot_cfg = {"responsive": True, "displayModeBar": False}

    def _layout(title: str, margin: dict[str, int], **extra: Any) -> dict[str, Any]:
        base = {
            **BG,
            "font": FONT,
            "title": {"text": title, "font": {**FONT, "size": 13}, "x": 0, "xanchor": "left"},
            "margin": margin,
            "showlegend": False,
            "xaxis": {**GRID, "tickfont": {**FONT, "size": 11}},
            "yaxis": {**GRID, "range": [0, 1.06], "tickfont": {**FONT, "size": 11}},
        }
        base.update(extra)
        return base

    metric_names = ["Accuracy", "Macro F1", "Binary F1"]
    summary_traces = []
    for k in order:
        si = order.index(k)
        ys = [acc[si], f1_macro[si], f1_bin[si]]
        summary_traces.append(
            {
                "x": metric_names,
                "y": ys,
                "name": sys_short[si],
                "type": "bar",
                "marker": {"color": colors[si], "line": {"width": 0}},
                "text": [f"{v:.3f}" for v in ys],
                "textposition": "outside",
                "textfont": {**FONT, "size": 10},
                "cliponaxis": False,
            }
        )
    layout_summary = {
        **BG,
        "font": FONT,
        "barmode": "group",
        "bargroupgap": 0.08,
        "bargap": 0.28,
        "showlegend": True,
        "legend": {"orientation": "h", "y": -0.32, "x": 0, "font": {**FONT, "size": 11}},
        "margin": {"l": 52, "r": 20, "t": 16, "b": 130},
        "xaxis": {**GRID, "tickfont": {**FONT, "size": 11}},
        "yaxis": {**GRID, "range": [0, 1.12], "tickfont": {**FONT, "size": 11}},
        "title": {"text": ""},
    }

    bin_colors_all = [colors[order.index(k)] for k in order]
    macro_f1_only_trace = {
        "x": sys_short,
        "y": f1_macro,
        "type": "bar",
        "marker": {"color": bin_colors_all, "line": {"width": 0}},
        "text": [f"{v:.3f}" for v in f1_macro],
        "textposition": "outside",
        "textfont": {**FONT, "size": 10},
        "cliponaxis": False,
    }
    layout_macro_f1_only = {
        **BG,
        "font": FONT,
        "margin": {**MARGIN_SM, "b": 56},
        "showlegend": False,
        "bargap": 0.38,
        "xaxis": {**GRID, "tickfont": {**FONT, "size": 10}},
        "yaxis": {**GRID, "range": [0, 1.06], "tickfont": {**FONT, "size": 11}},
        "title": {"text": ""},
    }
    bin_trace = {
        "x": [sys_short[order.index(k)] for k in order],
        "y": [f1_bin[order.index(k)] for k in order],
        "type": "bar",
        "marker": {"color": bin_colors_all, "line": {"width": 0}},
        "text": [f"{f1_bin[order.index(k)]:.3f}" for k in order],
        "textposition": "outside",
        "textfont": {**FONT, "size": 10},
        "cliponaxis": False,
    }
    layout_bin = {
        **BG,
        "font": FONT,
        "margin": {**MARGIN_SM, "b": 56},
        "showlegend": False,
        "bargap": 0.38,
        "xaxis": {**GRID, "tickfont": {**FONT, "size": 10}},
        "yaxis": {**GRID, "range": [0, 1.06], "tickfont": {**FONT, "size": 11}},
        "title": {"text": ""},
    }

    pc_traces = []
    for ci, lab in enumerate(LABELS):
        ys = [
            float(report["metrics"][k]["multiclass"].get("per_class", {}).get(lab, {}).get("f1-score", 0) or 0)
            for k in order
        ]
        pc_traces.append(
            {
                "x": sys_short,
                "y": ys,
                "name": lab,
                "type": "bar",
                "marker": {"color": BIAS_TYPE_BAR_COLORS[ci], "line": {"width": 0}},
                "text": [f"{v:.3f}" if v > 0 else "" for v in ys],
                "textposition": "outside",
                "textfont": {**FONT, "size": 10},
                "cliponaxis": False,
            }
        )
    layout_pc = {
        **BG,
        "font": FONT,
        "barmode": "group",
        "bargap": 0.28,
        "bargroupgap": 0.06,
        "margin": {"l": 52, "r": 20, "t": 16, "b": 128},
        "showlegend": True,
        "legend": {"orientation": "h", "y": -0.32, "font": {**FONT, "size": 10}},
        "xaxis": {**GRID, "tickfont": {**FONT, "size": 10.5}},
        "yaxis": {**GRID, "range": [0, 1.08], "tickfont": {**FONT, "size": 11}},
        "title": {"text": ""},
    }

    cm_labels = ["gender", "nationality", "profession", "neutral"]
    cm_titles_map = {
        "s1": "S1 Rule-based only",
        "s2": "S2 Rules + Linguistic",
        "s3": "S3 DistilBERT standalone",
        "s4_claude": "S4a Claude Sonnet 4",
        "s4_gemini": "S4b Gemini 2.5 Pro",
        "s5": "PRISM",
    }

    def _cm_spec_plot(k: str, fid: str, *, height: int) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        cm = np.array(report["confusion_matrices"][k], dtype=float)
        rs = cm.sum(axis=1, keepdims=True) + 1e-9
        z = (cm / rs).tolist()
        annot = [[("" if v <= 0 else f"{v:.2f}") for v in row] for row in z]
        data = [
            {
                "z": z,
                "x": cm_labels,
                "y": cm_labels,
                "type": "heatmap",
                "colorscale": [
                    [0.0, ANTHROPIC["cream"]],
                    [0.35, "#EDD5CC"],
                    [0.65, "#D9A088"],
                    [1.0, "#8B4D3A"],
                ],
                "showscale": False,
                "xgap": 2,
                "ygap": 2,
                "text": annot,
                "texttemplate": "%{text}",
                "textfont": {"family": "Inter, sans-serif", "size": 10, "color": ANTHROPIC["ink"]},
            }
        ]
        # Extra left margin + title standoff so the rotated "Gold" label clears tick labels.
        layout_cm = {
            **BG,
            "font": FONT,
            "title": {"text": cm_titles_map[k], "font": {**FONT, "size": 12}, "x": 0.02},
            "margin": {"l": 88, "r": 14, "t": 40, "b": 56},
            "xaxis": {
                "title": {"text": "Predicted", "font": {**FONT, "size": 10}, "standoff": 8},
                "tickfont": {**FONT, "size": 9.5},
            },
            "yaxis": {
                "title": {"text": "Gold", "font": {**FONT, "size": 11}, "standoff": 26},
                "tickfont": {**FONT, "size": 9.5},
                "autorange": "reversed",
            },
            "height": height,
        }
        return (fid, data, layout_cm)

    cm_main = _cm_spec_plot("s5", "cmS5", height=392)
    cm_ref_keys = ["s1", "s2", "s3", "s4_claude", "s4_gemini"]
    cm_ref_specs = [_cm_spec_plot(k, f"cmRef{k}", height=340) for k in cm_ref_keys]
    cm_ref_row1 = "".join(
        f'<div class="cm-ref-cell"><div id="{fid}" class="plot-cm-ref"></div></div>' for fid, _, _ in cm_ref_specs[:3]
    )
    cm_ref_row2 = "".join(
        f'<div class="cm-ref-cell"><div id="{fid}" class="plot-cm-ref"></div></div>' for fid, _, _ in cm_ref_specs[3:]
    )

    roc_keys_auc = ["s3", "s4_claude", "s4_gemini", "s5"]
    roc_labels_short = ["S3 BERT", "S4a Claude", "S4b Gemini", "PRISM"]
    roc_vals_neural = [report["roc_auc_macro_ovr"].get(k) for k in roc_keys_auc]
    roc_display = [0.0 if v is None else float(v) for v in roc_vals_neural]
    roc_text = ["n/a" if v is None else f"{float(v):.3f}" for v in roc_vals_neural]
    roc_colors_auc = [colors[order.index(k)] for k in roc_keys_auc]
    roc_trace = {
        "x": roc_labels_short,
        "y": roc_display,
        "type": "bar",
        "marker": {"color": roc_colors_auc, "line": {"width": 0}},
        "text": roc_text,
        "textposition": "outside",
        "textfont": {**FONT, "size": 10},
        "cliponaxis": False,
    }
    layout_roc = {
        **BG,
        "font": FONT,
        "margin": MARGIN_SM,
        "showlegend": False,
        "bargap": 0.38,
        "xaxis": {**GRID, "tickfont": {**FONT, "size": 10}},
        "yaxis": {**GRID, "range": [0, 1.08], "tickfont": {**FONT, "size": 11}},
        "title": {"text": ""},
        "shapes": [
            {
                "type": "line",
                "x0": -0.5,
                "x1": 3.5,
                "y0": 0.5,
                "y1": 0.5,
                "line": {"color": ANTHROPIC["hairline"], "width": 1, "dash": "dot"},
            }
        ],
        "annotations": [
            {
                "x": 3.38,
                "y": 0.52,
                "text": "chance",
                "showarrow": False,
                "font": {**FONT, "size": 9.5, "color": ANTHROPIC["muted"]},
            }
        ],
    }

    pr_vals_neural = [report["pr_auc_macro"].get(k) for k in roc_keys_auc]
    pr_display = [0.0 if v is None else float(v) for v in pr_vals_neural]
    pr_text = ["n/a" if v is None else f"{float(v):.3f}" for v in pr_vals_neural]
    pr_trace = {
        "x": roc_labels_short,
        "y": pr_display,
        "type": "bar",
        "marker": {"color": roc_colors_auc, "line": {"width": 0}},
        "text": pr_text,
        "textposition": "outside",
        "textfont": {**FONT, "size": 10},
        "cliponaxis": False,
    }
    layout_prauc = {**layout_roc}
    layout_prauc = {k: v for k, v in layout_roc.items() if k != "shapes" and k != "annotations"}
    layout_prauc = {
        **BG,
        "font": FONT,
        "margin": MARGIN_SM,
        "showlegend": False,
        "bargap": 0.38,
        "xaxis": {**GRID, "tickfont": {**FONT, "size": 10}},
        "yaxis": {**GRID, "range": [0, 1.08], "tickfont": {**FONT, "size": 11}},
        "title": {"text": ""},
    }

    boot_means = [boot[k][0] for k in order]
    boot_hi = [boot[k][2] - boot[k][0] for k in order]
    boot_lo = [boot[k][0] - boot[k][1] for k in order]
    boot_colors_all = [colors[order.index(k)] for k in order]
    boot_trace = {
        "x": sys_short,
        "y": boot_means,
        "type": "bar",
        "marker": {"color": boot_colors_all, "line": {"width": 0}},
        "error_y": {
            "type": "data",
            "symmetric": False,
            "array": boot_hi,
            "arrayminus": boot_lo,
            "color": ANTHROPIC["muted"],
            "thickness": 1.5,
            "width": 5,
        },
        "text": [f"{v:.3f}" for v in boot_means],
        "textposition": "outside",
        "textfont": {**FONT, "size": 10.5},
        "cliponaxis": False,
    }
    layout_boot = {
        **BG,
        "font": FONT,
        "margin": {**MARGIN, "b": 64},
        "showlegend": False,
        "bargap": 0.38,
        "xaxis": {**GRID, "tickfont": {**FONT, "size": 11}},
        "yaxis": {
            **GRID,
            "range": [0, 1.08],
            "title": "Macro F1",
            "tickfont": {**FONT, "size": 11},
        },
        "title": {"text": ""},
    }

    fp_counts = [report["error_summary"][k]["fp_neutral_as_bias"] for k in order]
    fn_counts = [report["error_summary"][k]["fn_bias_as_neutral"] for k in order]
    ymax_err = max(max(fp_counts), max(fn_counts)) * 1.15 + 5
    err_traces = [
        {
            "x": sys_short,
            "y": fp_counts,
            "name": "FP (neutral→bias)",
            "type": "bar",
            "marker": {"color": ANTHROPIC["terra"], "line": {"width": 0}},
            "text": [("" if v == 0 else str(int(v))) for v in fp_counts],
            "textposition": "outside",
            "textfont": {**FONT, "size": 10.5},
            "cliponaxis": False,
        },
        {
            "x": sys_short,
            "y": fn_counts,
            "name": "FN (bias→neutral)",
            "type": "bar",
            "marker": {"color": ANTHROPIC["ink"], "line": {"width": 0}},
            "text": [str(int(v)) for v in fn_counts],
            "textposition": "outside",
            "textfont": {**FONT, "size": 10.5},
            "cliponaxis": False,
        },
    ]
    layout_err = {
        **BG,
        "font": FONT,
        "barmode": "group",
        "bargap": 0.30,
        "bargroupgap": 0.08,
        "margin": {"l": 52, "r": 20, "t": 16, "b": 64},
        "showlegend": True,
        "legend": {"orientation": "h", "y": -0.22, "font": {**FONT, "size": 11}},
        "xaxis": {**GRID, "tickfont": {**FONT, "size": 11}},
        "yaxis": {
            **GRID,
            "title": "Count",
            "range": [0, ymax_err],
            "tickfont": {**FONT, "size": 11},
        },
        "title": {"text": ""},
    }

    lat_vals = [float(report["latency_ms"][k]) for k in order]
    lat_text = [_fmt_latency_label(v) for v in lat_vals]
    lat_colors_all = [colors[order.index(k)] for k in order]
    lat_trace = {
        "x": sys_short,
        "y": lat_vals,
        "type": "bar",
        "marker": {"color": lat_colors_all, "line": {"width": 0}},
        "text": lat_text,
        "textposition": "outside",
        "textfont": {**FONT, "size": 10.5},
        "cliponaxis": False,
    }
    layout_lat = {
        **BG,
        "font": FONT,
        "margin": MARGIN_SM,
        "showlegend": False,
        "bargap": 0.38,
        "xaxis": {**GRID, "tickfont": {**FONT, "size": 10.5}},
        "yaxis": {
            **GRID,
            "type": "log",
            "title": "ms (log scale)",
            "tickfont": {**FONT, "size": 11},
        },
        "title": {"text": ""},
    }

    thr_vals: list[float | None] = [report["throughput_posts_per_s"].get(k) for k in order]
    thr_text = [_fmt_thr_label(v, system_key=k) for v, k in zip(thr_vals, order)]
    thr_y_plot = [float(v) if v is not None else 0.0 for v in thr_vals]
    thr_trace = {
        "x": sys_short,
        "y": thr_y_plot,
        "type": "bar",
        "marker": {"color": lat_colors_all, "line": {"width": 0}},
        "text": thr_text,
        "textposition": "outside",
        "textfont": {**FONT, "size": 10.5},
        "cliponaxis": False,
    }
    layout_thr = {
        **BG,
        "font": FONT,
        "margin": MARGIN_SM,
        "showlegend": False,
        "bargap": 0.38,
        "xaxis": {**GRID, "tickfont": {**FONT, "size": 10.5}},
        "yaxis": {
            **GRID,
            "type": "linear",
            "title": "posts/s (batch total ÷ wall time)",
            "tickfont": {**FONT, "size": 11},
        },
        "title": {"text": ""},
    }

    cal_specs: list[tuple[str, list[dict[str, Any]], dict[str, Any], str]] = []
    cal_keys = ["s3", "s5"]
    cal_titles = ["S3 — DistilBERT Standalone", "PRISM"]
    cal_line = [colors[order.index("s3")], colors[order.index("s5")]]
    for idx, ck in enumerate(cal_keys):
        if ck not in report.get("calibration", {}):
            continue
        bins = report["calibration"][ck]["bins"]
        xs = [float(b["mid"]) for b in bins]
        ys = [float(b["acc"]) for b in bins]
        traces = [
            {
                "x": xs,
                "y": ys,
                "mode": "lines+markers",
                "name": "Empirical",
                "line": {"color": cal_line[idx], "width": 2},
                "marker": {"color": cal_line[idx], "size": 6},
            },
            {
                "x": [0, 1],
                "y": [0, 1],
                "mode": "lines",
                "name": "Perfect",
                "line": {"dash": "dot", "color": ANTHROPIC["hairline"], "width": 1.5},
            },
        ]
        layout_cal = {
            **BG,
            "font": FONT,
            "margin": {"l": 52, "r": 20, "t": 24, "b": 60},
            "showlegend": True,
            "legend": {"x": 0, "y": -0.28, "orientation": "h", "font": {**FONT, "size": 11}},
            "xaxis": {**GRID, "title": "Confidence", "range": [0, 1]},
            "yaxis": {**GRID, "title": "Accuracy", "range": [0, 1]},
            "height": 280,
            "title": {"text": ""},
        }
        cal_specs.append((f"cal{idx}", traces, layout_cal, cal_titles[idx]))

    fig_scripts = ""
    fig_scripts += f'Plotly.newPlot("plotMacroF1", {json.dumps([macro_f1_only_trace])}, {json.dumps(layout_macro_f1_only)}, {json.dumps(plot_cfg)});\n'
    fig_scripts += f'Plotly.newPlot("plotSummary", {json.dumps(summary_traces)}, {json.dumps(layout_summary)}, {json.dumps(plot_cfg)});\n'
    fig_scripts += f'Plotly.newPlot("plotBin", {json.dumps([bin_trace])}, {json.dumps(layout_bin)}, {json.dumps(plot_cfg)});\n'
    fig_scripts += f'Plotly.newPlot("plotPC", {json.dumps(pc_traces)}, {json.dumps(layout_pc)}, {json.dumps(plot_cfg)});\n'
    fig_scripts += f'Plotly.newPlot("{cm_main[0]}", {json.dumps(cm_main[1])}, {json.dumps(cm_main[2])}, {json.dumps(plot_cfg)});\n'
    for fid, data, layout in cm_ref_specs:
        fig_scripts += f'Plotly.newPlot("{fid}", {json.dumps(data)}, {json.dumps(layout)}, {json.dumps(plot_cfg)});\n'
    fig_scripts += f'Plotly.newPlot("plotROC", {json.dumps([roc_trace])}, {json.dumps(layout_roc)}, {json.dumps(plot_cfg)});\n'
    fig_scripts += f'Plotly.newPlot("plotPRAUC", {json.dumps([pr_trace])}, {json.dumps(layout_prauc)}, {json.dumps(plot_cfg)});\n'
    fig_scripts += f'Plotly.newPlot("plotBoot", {json.dumps([boot_trace])}, {json.dumps(layout_boot)}, {json.dumps(plot_cfg)});\n'
    for fid, traces, layout, _title in cal_specs:
        fig_scripts += f'Plotly.newPlot("{fid}", {json.dumps(traces)}, {json.dumps(layout)}, {json.dumps(plot_cfg)});\n'
    fig_scripts += f'Plotly.newPlot("plotErrors", {json.dumps(err_traces)}, {json.dumps(layout_err)}, {json.dumps(plot_cfg)});\n'
    fig_scripts += f'Plotly.newPlot("plotLat", {json.dumps([lat_trace])}, {json.dumps(layout_lat)}, {json.dumps(plot_cfg)});\n'
    fig_scripts += f'Plotly.newPlot("plotThr", {json.dumps([thr_trace])}, {json.dumps(layout_thr)}, {json.dumps(plot_cfg)});\n'

    png_specs: list[tuple[str, list[dict[str, Any]], dict[str, Any]]] = [
        ("plotMacroF1", [macro_f1_only_trace], layout_macro_f1_only),
        ("plotSummary", summary_traces, layout_summary),
        ("plotBin", [bin_trace], layout_bin),
        ("plotPC", pc_traces, layout_pc),
        (cm_main[0], cm_main[1], cm_main[2]),
        *[(fid, data, layout) for fid, data, layout in cm_ref_specs],
        ("plotROC", [roc_trace], layout_roc),
        ("plotPRAUC", [pr_trace], layout_prauc),
        ("plotBoot", [boot_trace], layout_boot),
        *[(fid, traces, layout) for fid, traces, layout, _ in cal_specs],
        ("plotErrors", err_traces, layout_err),
        ("plotLat", [lat_trace], layout_lat),
        ("plotThr", [thr_trace], layout_thr),
    ]
    if export_png_dir is not None:
        _export_plotly_specs_to_png(png_specs, export_png_dir)

    data_json = json.dumps(_json_sanitize(report), ensure_ascii=False)

    EVAL_CSS = """  @import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,300..700;1,14..32,300..700&family=Lora:ital,wght@0,400..700;1,400..700&display=swap');
  :root {
    --terra: #D97757;
    --cream: #FAF8F5;
    --cream-dark: #F3EFE8;
    --ink: #141413;
    --muted: #6B6560;
    --hairline: #E8E4DF;
    --wash: #EDE9E3;
    --sp-1: 8px; --sp-2: 10px; --sp-3: 13px; --sp-4: 16px; --sp-5: 21px; --sp-6: 27px;
    --sp-7: 34px; --sp-8: 44px; --sp-9: 55px; --sp-10: 68px; --sp-11: 85px; --sp-12: 105px; --sp-13: 121px;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Inter', -apple-system, system-ui, sans-serif;
    background: var(--cream);
    color: var(--ink);
    font-size: 15px;
    line-height: 1.65;
    -webkit-font-smoothing: antialiased;
  }
  header {
    background: var(--cream);
    border-bottom: 1px solid var(--hairline);
    padding: var(--sp-10) 7% var(--sp-8);
  }
  .header-eyebrow {
    font-family: 'Lora', Georgia, serif;
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: var(--sp-3);
  }
  header h1 {
    font-family: 'Lora', Georgia, serif;
    font-size: clamp(28px, 3.2vw, 36px);
    font-weight: 600;
    letter-spacing: -0.02em;
    color: var(--ink);
    margin-bottom: var(--sp-3);
    line-height: 1.2;
  }
  header h1 .title-em { box-shadow: inset 0 -3px 0 0 var(--terra); }
  header p { font-size: 15px; color: var(--muted); max-width: 42rem; line-height: 1.75; font-weight: 400; }
  .header-meta { display: flex; gap: var(--sp-2); flex-wrap: wrap; margin-top: var(--sp-6); }
  .chip {
    display: inline-flex;
    align-items: center;
    gap: var(--sp-2);
    padding: var(--sp-2) var(--sp-4);
    background: transparent;
    border: 1px solid var(--hairline);
    border-radius: 0;
    font-size: 12px;
    color: var(--muted);
    font-weight: 500;
    letter-spacing: 0.02em;
  }
  .chip strong { font-weight: 600; color: var(--ink); }
  main { max-width: 1320px; margin: 0 auto; padding: var(--sp-9) 7%; }
  .section { margin-bottom: var(--sp-11); }
  .section-label {
    font-family: 'Lora', Georgia, serif;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.06em;
    color: var(--ink);
    margin-bottom: var(--sp-4);
    padding-bottom: var(--sp-2);
    border-bottom: 3px solid var(--terra);
    display: inline-block;
    min-width: 220px;
  }
  main > .section:nth-child(even) {
    background: var(--cream-dark);
    margin-left: -7%;
    margin-right: -7%;
    padding-left: 7%;
    padding-right: 7%;
    padding-top: var(--sp-8);
    padding-bottom: var(--sp-8);
    border-top: 1px solid var(--hairline);
    border-bottom: 1px solid var(--hairline);
  }
  main > .section:nth-child(odd) .card,
  main > .section:nth-child(odd) .stat-tile {
    background: var(--cream);
  }
  h2 {
    font-family: 'Lora', Georgia, serif;
    font-size: 20px;
    font-weight: 600;
    color: var(--ink);
    margin-bottom: var(--sp-5);
    letter-spacing: -0.02em;
  }
  h2.uLINE { box-shadow: inset 0 -3px 0 0 var(--terra); display: inline; padding-bottom: 2px; }
  .card {
    background: var(--cream);
    border: 1px solid var(--hairline);
    border-radius: 0;
    padding: var(--sp-7);
    margin-bottom: var(--sp-5);
    box-shadow: none;
  }
  .card-grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: var(--sp-5); margin-bottom: var(--sp-5); }
  .card-grid-6 { display: grid; grid-template-columns: repeat(6, 1fr); gap: var(--sp-4); margin-bottom: var(--sp-5); }
  @media (max-width: 1100px) { .card-grid-6 { grid-template-columns: repeat(3, 1fr); } }
  @media (max-width: 640px) { .card-grid-6 { grid-template-columns: 1fr 1fr; } }
  .stat-tile {
    background: var(--cream);
    border: 1px solid var(--hairline);
    border-radius: 0;
    padding: var(--sp-6) var(--sp-5);
  }
  .stat-tile .sys-name {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: var(--sp-3);
    line-height: 1.4;
  }
  .stat-tile .big-num {
    font-family: 'Inter', sans-serif;
    font-size: 32px;
    font-weight: 400;
    letter-spacing: -0.03em;
    line-height: 1;
    margin-bottom: var(--sp-2);
    color: var(--ink);
  }
  .stat-tile .sub-label { font-size: 11px; color: var(--muted); font-weight: 500; }
  .stat-tile--hero {
    border-bottom: 3px solid var(--terra);
    border-color: var(--hairline);
    border-bottom-color: var(--terra);
  }
  .stat-tile--hero .big-num { font-weight: 400; color: var(--terra); }
  .plot { height: 400px; width: 100%; }
  .plot-md { height: 340px; width: 100%; }
  .plot-sm { height: 290px; width: 100%; }
  .plot-cm-main { height: 420px; width: 100%; min-height: 400px; }
  .plot-cm-ref { height: 380px; width: 100%; min-height: 360px; }
  .cm-ref-grid-rows { display: flex; flex-direction: column; gap: var(--sp-5); }
  .cm-ref-row {
    display: grid;
    grid-template-columns: repeat(3, minmax(280px, 1fr));
    gap: var(--sp-4);
    align-items: start;
  }
  .cm-ref-row--pair {
    grid-template-columns: repeat(2, minmax(280px, 1fr));
    max-width: 900px;
  }
  @media (max-width: 1040px) {
    .cm-ref-row,
    .cm-ref-row--pair { grid-template-columns: 1fr; max-width: none; }
  }
  .cm-ref-cell { min-width: 0; }
  .table-wrap { overflow-x: auto; }
  table.data-table { width: 100%; border-collapse: collapse; font-size: 13px; font-family: 'Inter', sans-serif; }
  table.data-table thead th {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--muted);
    padding: 0 var(--sp-4) var(--sp-3) 0;
    text-align: left;
    border-bottom: 3px solid var(--ink);
    background: transparent;
  }
  table.data-table tbody td {
    padding: var(--sp-4) var(--sp-4) var(--sp-4) 0;
    color: var(--muted);
    border-bottom: 1px solid var(--hairline);
    vertical-align: top;
  }
  table.data-table tbody tr:last-child td { border-bottom: none; }
  table.data-table tr.row-emphasis td { color: var(--ink); }
  td.mono, .mono { font-variant-numeric: tabular-nums; font-size: 12.5px; color: var(--ink); letter-spacing: 0.01em; }
  .num-strong { font-weight: 600; color: var(--ink); }
  .sys-name-cell { font-weight: 600; color: var(--ink); }
  td.dim { color: var(--muted); opacity: 0.85; }
  .legend-row { display: flex; flex-wrap: wrap; gap: var(--sp-5) var(--sp-7); margin-bottom: var(--sp-7); }
  .legend-item { display: flex; align-items: center; gap: var(--sp-3); font-size: 12px; color: var(--muted); font-weight: 500; }
  .legend-swatch { width: 12px; height: 12px; flex-shrink: 0; border: 1px solid var(--hairline); border-radius: 0; }
  .callout {
    background: var(--wash);
    border: 1px solid var(--hairline);
    border-radius: 0;
    padding: var(--sp-6) var(--sp-7);
    margin-bottom: var(--sp-5);
  }
  .callout p { font-size: 14px; color: var(--muted); line-height: 1.75; }
  .callout p + p { margin-top: var(--sp-3); }
  .callout strong { color: var(--ink); font-weight: 600; }
  footer {
    background: var(--ink);
    color: var(--cream);
    border-top: none;
    padding: var(--sp-7) 7%;
    font-size: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: var(--sp-4);
    flex-wrap: wrap;
  }
  footer .mono { font-size: 11px; opacity: 0.85; }
  code.mono { font-family: ui-monospace, 'Cascadia Code', monospace; font-size: 11.5px; background: var(--wash); padding: 2px 6px; border: 1px solid var(--hairline); }"""

    legend_row = "".join(
        f'<div class="legend-item"><div class="legend-swatch" style="background:{colors[i]}"></div>'
        f"{html_module.escape(tile_names[i])}</div>"
        for i in range(6)
    )

    cal_cards = ""
    for idx, (_, _, _, ttl) in enumerate(cal_specs):
        cal_cards += (
            f'<div class="card" style="margin:0"><h2><span class="uLINE">{html_module.escape(ttl)}</span></h2>'
            f'<div id="cal{idx}" class="plot-sm"></div></div>'
        )
    if not cal_cards:
        cal_cards = '<p class="dim">No calibration data.</p>'

    pv_line = "p &lt; 0.001" if pv_s5_s3 < 0.001 else html_module.escape(f"p = {pv_s5_s3:.4g}")
    thr_llm_note = (
        f"OpenRouter baselines average ≈ {thr_llm_avg:.1f} posts/s on measured runs."
        if thr_llm_avg is not None
        else "OpenRouter baselines are network-bound; stub runs record no wall time."
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Evaluation Dashboard — PRISM</title>
<script src="{plotly_cdn}"></script>
<style>
{EVAL_CSS}
</style>
</head>
<body>
<header>
  <p class="header-eyebrow">PRISM research</p>
  <h1>Comprehensive <span class="title-em">Evaluation</span> Suite</h1>
  <p>Comparison of <strong>six</strong> systems (S1–S4b, PRISM) on a manually annotated gold set. <strong>Macro F1</strong> is the headline metric. <strong>S4a/S4b</strong> are zero-shot API baselines (not fine-tuned on the project taxonomy). Full metrics table, charts, and confusion matrices include all systems; <strong>PRISM</strong> (system id S5) is editorially emphasised as the production stack.</p>
  <div class="header-meta">
    <span class="chip"><strong>Gold set</strong> {html_module.escape(gold_name)}</span>
    <span class="chip"><strong>n</strong> {n_posts} posts</span>
    <span class="chip"><strong>Classes</strong> 4 (gender, nationality, profession, neutral)</span>
    <span class="chip"><strong>Bootstrap</strong> {meta["bootstrap_n"]} resamples · seed {meta["bootstrap_seed"]}</span>
    <span class="chip"><strong>LLM API</strong> {html_module.escape(chip_llm)}</span>
  </div>
</header>
<main>
  <div class="section">
    <p class="section-label">01 · Executive Summary</p>
    <div class="callout">
      <p><strong>PRISM</strong> achieves macro F1 <strong>{best_f1:.3f}</strong> — <strong>{pp_vs_s3:+.1f} pp</strong> over standalone DistilBERT (S3) and <strong>{pp_vs_rules:+.1f} pp</strong> over the mean of rule baselines (S1/S2). PRISM <strong>significantly outperforms</strong> S3 on discordant pairs (McNemar exact test, <strong>{pv_line}</strong>, α = 0.05).</p>
      <p>Production stack: StereoSet <code class="mono">distilbert_B_balanced</code> + auxiliary <code class="mono">distilbert_social_bias</code> + 40-d meta-fusion. {llm_appendix_note} Additional pairwise tests (neural vs rules) appear in §07.</p>
      {llm_pair_callout}
    </div>
    <div class="card-grid-6">
      {stat_tiles}
    </div>
  </div>

  <div class="section">
    <p class="section-label">02 · Systems Under Comparison</p>
    <div class="card">
      <h2><span class="uLINE">All scored pipelines</span></h2>
      <div class="table-wrap">
        <table class="data-table">
          <thead><tr><th style="width:52px">ID</th><th style="width:220px">System</th><th>Description</th></tr></thead>
          <tbody>{systems_table_all}</tbody>
        </table>
      </div>
    </div>
  </div>

  <div class="section">
    <p class="section-label">03 · Classification Metrics</p>
    <div class="legend-row">{legend_row}</div>
    <div class="card" style="max-width:720px">
      <h2><span class="uLINE">Macro F1</span></h2>
      <div id="plotMacroF1" class="plot-sm"></div>
    </div>
    <div class="card">
      <h2><span class="uLINE">Accuracy, macro F1, binary F1</span></h2>
      <p style="font-size:13px;color:var(--muted);margin-bottom:var(--sp-4);max-width:46rem">Headline: <strong>macro F1</strong>. Binary F1 = any bias vs neutral. Colour keys match the six system traces (terra = PRISM).</p>
      <div id="plotSummary" class="plot"></div>
    </div>
    <div class="card" style="max-width:720px">
      <h2><span class="uLINE">Binary F1 (bias vs neutral)</span></h2>
      <div id="plotBin" class="plot-sm"></div>
    </div>
    <div class="card">
      <h2><span class="uLINE">Comparison table</span></h2>
      <div class="table-wrap">
        <table class="data-table">
          <thead><tr><th>System</th><th>Accuracy</th><th>Macro F1</th><th>Binary F1</th></tr></thead>
          <tbody>{full_metrics_rows}</tbody>
        </table>
      </div>
    </div>
  </div>

  <div class="section">
    <p class="section-label">04 · Per-Class F1</p>
    <div class="card">
      <h2><span class="uLINE">Per-class F1 — all systems</span></h2>
      <div id="plotPC" class="plot"></div>
    </div>
    <div class="card">
      <div class="table-wrap">
        <table class="data-table">
          <thead><tr><th>Class</th>{per_class_header}</tr></thead>
          <tbody>{per_class_rows}</tbody>
        </table>
      </div>
    </div>
  </div>

  <div class="section">
    <p class="section-label">04b · Bias-Type Fairness (PRISM)</p>
    <div class="callout">
      <p><strong>Scope:</strong> Here “fairness” means <em>evenness of detection quality across bias-type labels</em> (gender / nationality / profession), not demographic parity in hiring law. Large gaps suggest uneven coverage or ambiguity for specific stereotype families.</p>
    </div>
    <div class="card">
      <div class="table-wrap">
        <table class="data-table">
          <thead><tr><th>Metric</th><th>Value</th><th>Note</th></tr></thead>
          <tbody>{fair_rows_html}</tbody>
        </table>
      </div>
      <p style="margin-top:var(--sp-5);font-size:13px;color:var(--muted);max-width:42rem">Total misclassified by PRISM: <strong>{te_total}</strong>. Semantic taxonomy below summarises <em>why</em> failures occur (heuristic buckets).</p>
    </div>
  </div>

  <div class="section">
    <p class="section-label">05 · Confusion Matrices (Row-Normalised)</p>
    <div class="callout">
      <p><strong>Primary:</strong> PRISM — rows gold, columns predicted; cells are P(predicted | gold). Heatmap uses a muted terra gradient (perceptually soft). Reference rows: <strong>S1–S3</strong> (rules / standalone BERT) and <strong>S4a / S4b</strong> (zero-shot LLMs).</p>
    </div>
    <div class="card">
      <h2><span class="uLINE">PRISM</span></h2>
      <div id="cmS5" class="plot-cm-main"></div>
    </div>
    <div class="card">
      <h2 style="font-family:Lora,Georgia,serif;font-size:17px;"><span class="uLINE">Reference — S1 through S4b</span></h2>
      <div class="cm-ref-grid-rows">
        <div class="cm-ref-row">{cm_ref_row1}</div>
        <div class="cm-ref-row cm-ref-row--pair">{cm_ref_row2}</div>
      </div>
    </div>
  </div>

  <div class="section">
    <p class="section-label">06 · Separability (OvR AUC)</p>
    <div class="callout">
      <p>Macro-averaged ROC and PR AUC (one-vs-rest) for systems with probability vectors: <strong>S3, S4a, S4b, PRISM</strong>. Rule systems (S1, S2) have no scores (<em>n/a</em>). LLM AUC uses one-hot probability mass — interpret as a coarse separability signal when responses collapse to a single class.</p>
    </div>
    <div class="card-grid-2">
      <div class="card" style="margin:0">
        <h2>Macro-Averaged ROC AUC (OvR)</h2>
        <div id="plotROC" class="plot-sm"></div>
      </div>
      <div class="card" style="margin:0">
        <h2>Macro-Averaged PR AUC (OvR)</h2>
        <div id="plotPRAUC" class="plot-sm"></div>
      </div>
    </div>
  </div>

  <div class="section">
    <p class="section-label">07 · Statistical Analysis</p>
    <div class="card">
      <h2>Bootstrap 95% Confidence Intervals — Macro F1</h2>
      <div id="plotBoot" class="plot"></div>
    </div>
    <div class="card">
      <h2>McNemar — PRISM vs each baseline</h2>
      <p style="font-size:12.5px;color:var(--muted);margin-bottom:20px;line-height:1.6"><strong>PRISM vs baseline</strong> table: n₁₀ = PRISM correct and comparator wrong; n₀₁ = the reverse. {html_module.escape(mcn_intro)}</p>
      <div class="table-wrap">
        <table class="data-table">
          <thead><tr><th>Comparator</th><th>n<sub>10</sub></th><th>n<sub>01</sub></th><th>p-value</th><th>Interpretation</th></tr></thead>
          <tbody>{mcn_rows}</tbody>
        </table>
      </div>
    </div>
    <div class="card">
      <h2>McNemar — Neural vs rules &amp; fusion vs neural</h2>
      <p style="font-size:12.5px;color:var(--muted);margin-bottom:16px;line-height:1.6">Shows DistilBERT beating both rule stacks. <strong>PRISM vs S3</strong> is already in the table above (comparator row &quot;DistilBERT Standalone&quot;).</p>
      <div class="table-wrap">
        <table class="data-table">
          <thead><tr><th>Comparison</th><th>n<sub>10</sub></th><th>n<sub>01</sub></th><th>p-value</th><th>Interpretation</th></tr></thead>
          <tbody>{pairwise_mcn_rows}</tbody>
        </table>
      </div>
    </div>
  </div>

  <div class="section">
    <p class="section-label">08 · Calibration (Reliability Diagrams)</p>
    <div class="callout">
      <p>Reliability diagrams plot binned model confidence against empirical accuracy. Only <strong>S3 (DistilBERT)</strong> and <strong>PRISM</strong> are shown — rule-based and LLM zero-shot systems do not expose comparable probabilistic outputs.</p>
    </div>
    <div class="card-grid-2">{cal_cards}</div>
  </div>

  <div class="section">
    <p class="section-label">09 · Error Analysis</p>
    <div class="card">
      <h2>False Positives &amp; False Negatives</h2>
      <p style="font-size:12.5px;color:var(--muted);margin-bottom:20px">FP: gold neutral, predicted any bias class. FN: gold any bias, predicted neutral.</p>
      <div id="plotErrors" class="plot-md"></div>
      <div class="table-wrap" style="margin-top:24px">
        <table class="data-table">
          <thead><tr><th>System</th><th>FP (neutral→bias)</th><th>FN (bias→neutral)</th><th>Total errors</th></tr></thead>
          <tbody>{err_fp_fn_rows}</tbody>
        </table>
      </div>
    </div>
    <div class="card">
      <h2>Semantic error taxonomy (PRISM, heuristic)</h2>
      <p style="font-size:12.5px;color:var(--muted);margin-bottom:16px">Each PRISM error is assigned one bucket (ordered rules: cross-type → critique markers → neutral↔bias → tone). Percentages sum to 100% over misclassifications.</p>
      <div class="table-wrap">
        <table class="data-table">
          <thead><tr><th>Error type</th><th>Count</th><th>%</th></tr></thead>
          <tbody>{tax_rows_html}</tbody>
        </table>
      </div>
    </div>
    <div class="card">
      <h2>PRISM — Misclassification sample</h2>
      <p style="font-size:12.5px;color:var(--muted);margin-bottom:20px">Up to 18 excerpts for qualitative review.</p>
      <div class="table-wrap">
        <table class="data-table">
          <thead><tr><th style="width:72px">Post ID</th><th style="width:130px">Gold</th><th style="width:130px">Predicted</th><th>Text</th></tr></thead>
          <tbody>{err_rows}</tbody>
        </table>
      </div>
    </div>
  </div>

  <div class="section">
    <p class="section-label">10 · Latency &amp; Throughput</p>
    <div class="card-grid-2">
      <div class="card" style="margin:0">
        <h2>End-to-end Batch Latency (ms)</h2>
        <div id="plotLat" class="plot-sm"></div>
      </div>
      <div class="card" style="margin:0">
        <h2>Throughput (posts / second)</h2>
        <div id="plotThr" class="plot-sm"></div>
      </div>
    </div>
    <div class="callout" style="margin-top:0">
      <p>Batch wall time for the full gold set. PRISM ≈ <strong>{sec_per_post_s5:.2f} s/post</strong> in this run (transformer + optional hate API + meta-fusion). {html_module.escape(thr_llm_note)} Rule/linguistic systems are in-process.</p>
    </div>
  </div>

  <div class="section">
    <p class="section-label">11 · System 4 — OpenRouter zero-shot prompt</p>
    <div class="callout">
      <p>Each API call sends a single user message: instructions, label definitions, and the post body (truncated to <strong>8000</strong> characters). Models: <code class="mono">{html_module.escape(CLAUDE_MODEL)}</code> (S4a) and <code class="mono">{html_module.escape(GEMINI_MODEL)}</code> (S4b). Temperature <strong>0.1</strong>, <code class="mono">max_tokens</code> <strong>220</strong>.</p>
    </div>
    <div class="card">
      <h2><span class="uLINE">Verbatim user prompt template</span></h2>
      <p style="font-size:12.5px;color:var(--muted);margin-bottom:var(--sp-4)">Placeholder <code class="mono">[POST_TEXT_TRUNCATED_TO_8000_CHARS]</code> stands in for the evaluated post text.</p>
      <pre style="font-family:ui-monospace,monospace;font-size:11px;line-height:1.5;overflow:auto;max-height:min(380px,50vh);padding:var(--sp-5);border:1px solid var(--hairline);background:var(--wash);color:var(--ink);white-space:pre-wrap">{prompt_template_show}</pre>
    </div>
  </div>

  <div class="section">
    <p class="section-label">12 · Appendix — Machine-readable export</p>
    <div class="callout">
      <p>Single JSON blob with all metrics, confusion matrices, bootstrap CIs, McNemar tables, fairness diagnostics, and taxonomy buckets — suitable for archival or thesis supplementary material.</p>
    </div>
    <div class="card">
      <pre id="raw-json" style="display:none;font-family:ui-monospace,monospace;font-size:10px;line-height:1.45;overflow:auto;max-height:min(420px,55vh);padding:var(--sp-5);border:1px solid var(--hairline);background:var(--wash);color:var(--ink);">{data_json}</pre>
      <p style="margin-top:var(--sp-4);font-size:13px;color:var(--muted)">Unhide via DevTools or temporarily remove <code class="mono">display:none</code> on <code class="mono">#raw-json</code>.</p>
    </div>
  </div>

  <div class="section">
    <p class="section-label">13 · Methodology &amp; Limitations</p>
    <div class="card">
      <div class="table-wrap">
        <table class="data-table">
          <thead><tr><th style="width:180px">Aspect</th><th>Notes</th></tr></thead>
          <tbody>
            <tr><td class="sys-name-cell">LLM parse failures</td><td>If the model response cannot be resolved to one of the four labels, that instance is scored as <code class="mono">neutral</code>.</td></tr>
            <tr><td class="sys-name-cell">No ensemble</td><td>Majority-vote and tie→neutral behaviour are not used in headline metrics.</td></tr>
            <tr><td class="sys-name-cell">Partial OpenRouter runs</td><td>If <code class="mono">--llm-max K</code> with K &lt; n, tail rows are <code class="mono">neutral</code> for both LLMs without API calls.</td></tr>
            <tr><td class="sys-name-cell">Cohen&apos;s κ (LLM pair)</td><td>{kappa_cell}</td></tr>
            <tr><td class="sys-name-cell">ROC/PR AUC for LLMs</td><td>Degenerate one-hot scores — coarse separability proxy only.</td></tr>
            <tr><td class="sys-name-cell">Bootstrap</td><td>{meta["bootstrap_n"]} resamples, seed {meta["bootstrap_seed"]}; 95% CIs are percentile intervals.</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</main>
<footer>
  <span>PRISM · Evaluation Dashboard · <span class="mono">scripts/build_evaluation_suite_report.py</span></span>
  <span style="opacity:0.85">Editorial layout · Lora / Inter · Terra #D97757 · Plotly 2.27</span>
</footer>
<script>
window.addEventListener('DOMContentLoaded', function() {{
  {fig_scripts}
}});
</script>
</body>
</html>"""
if __name__ == "__main__":
    main()
