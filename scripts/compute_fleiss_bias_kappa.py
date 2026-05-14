#!/usr/bin/env python3
"""
Fleiss' κ (binary biased vs not) for three raters — minimal setup for thesis §3.2.2.

Two ways to use:

1) **Fully automated (you only need OPENROUTER_API_KEY + network)**  
   Fetches two independent LLM labels per post (same zero-shot task as
   ``build_evaluation_suite_report.py``) and uses the gold ``label`` column as the
   third binary rater (biased = not ``neutral``). Writes a reproducible CSV and prints κ.

   cd PRISM
   export OPENROUTER_API_KEY=...
   python scripts/compute_fleiss_bias_kappa.py --fetch-llm

   Caveat printed in the output: the gold column is **adjudicated final** labels, not a
   blind independent third pass. κ is still a valid descriptive agreement statistic;
   phrase limitations accordingly in Chapter 6.

2) **From your own spreadsheet (no API)**  
   Provide a CSV with exactly three columns of 0/1 (or four-way labels).

   python scripts/compute_fleiss_bias_kappa.py --csv path/to/ratings.csv \\
       --binary-cols biased_claude biased_gemini biased_author

   python scripts/compute_fleiss_bias_kappa.py --csv path/to/ratings.csv \\
       --four-way-cols label_claude label_gemini label_author

Keep the OpenRouter prompt in sync with ``OPENROUTER_EVAL_PROMPT_TEMPLATE`` in
``scripts/build_evaluation_suite_report.py`` if you change definitions there.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.label_config import LABELS, NEUTRAL_LABEL  # noqa: E402

try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

# Sync with scripts/build_evaluation_suite_report.py (evaluation suite S4 prompt).
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
CLAUDE_MODEL_DEFAULT = "anthropic/claude-sonnet-4"
GEMINI_MODEL_DEFAULT = "google/gemini-2.5-pro"

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
            "HTTP-Referer": "https://localhost/prism-fleiss",
            "X-Title": "PRISM Fleiss kappa",
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


def four_way_to_binary(s: object) -> int:
    v = str(s).strip()
    if v not in LABELS:
        raise ValueError(f"Unknown label {v!r}; expected one of {LABELS}")
    return 0 if v == NEUTRAL_LABEL else 1


def fleiss_kappa_binary(counts: np.ndarray) -> float:
    """
    Fleiss' κ for constant N raters per subject.
    counts: shape (n_subjects, 2) with columns [n_unbiased, n_biased], rows sum to N.
    """
    n, k = counts.shape
    if k != 2:
        raise ValueError("This helper expects k=2 categories.")
    row_sums = counts.sum(axis=1)
    if not np.all(row_sums == row_sums[0]):
        raise ValueError("Each row must sum to the same number of raters.")
    n_raters = int(row_sums[0])
    if n_raters < 2:
        raise ValueError("Need at least 2 raters per subject.")
    p_i = np.sum(counts * (counts - 1), axis=1) / (n_raters * (n_raters - 1))
    p_bar = float(np.mean(p_i))
    p_j = np.sum(counts, axis=0) / (n * n_raters)
    p_e = float(np.sum(p_j**2))
    if p_e >= 1.0 - 1e-14:
        return float("nan")
    return (p_bar - p_e) / (1.0 - p_e)


def landis_koch(kappa: float) -> str:
    if np.isnan(kappa):
        return "undefined (chance-level category distribution)"
    if kappa < 0:
        return "below chance for this sample"
    if kappa < 0.21:
        return "slight agreement beyond chance"
    if kappa < 0.41:
        return "fair agreement beyond chance"
    if kappa < 0.61:
        return "moderate agreement beyond chance"
    if kappa < 0.81:
        return "substantial agreement beyond chance"
    return "almost perfect agreement beyond chance"


def triple_agreement_rate(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float(np.mean((a == b) & (b == c)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Fleiss' κ for three binary bias raters.")
    ap.add_argument(
        "--csv",
        type=Path,
        help="Input CSV (required for --binary-cols / --four-way-cols modes).",
    )
    ap.add_argument(
        "--binary-cols",
        nargs=3,
        metavar="COL",
        help="Three column names with 0/1 (biased=1).",
    )
    ap.add_argument(
        "--four-way-cols",
        nargs=3,
        metavar="COL",
        help="Three columns with labels in "
        + str(LABELS)
        + "; biased is encoded as not neutral.",
    )
    ap.add_argument(
        "--fetch-llm",
        action="store_true",
        help="Call OpenRouter for Claude + Gemini; third rater = gold label column (see script docstring).",
    )
    ap.add_argument(
        "--gold",
        type=Path,
        default=_ROOT / "data/evaluation/manual_eval_v3_400_posts.csv",
        help="Gold CSV with post_id, text, label (used with --fetch-llm).",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=_ROOT / "data/evaluation/fleiss_three_raters_binary_400.csv",
        help="Where to write / resume the rater matrix (--fetch-llm).",
    )
    ap.add_argument(
        "--rater-table",
        type=Path,
        metavar="PATH",
        help="Recompute κ only: CSV with columns biased_claude, biased_gemini, biased_author (0/1). "
        "Use a file produced earlier via --fetch-llm --out-csv.",
    )
    ap.add_argument("--claude-model", default=CLAUDE_MODEL_DEFAULT)
    ap.add_argument("--gemini-model", default=GEMINI_MODEL_DEFAULT)
    ap.add_argument("--max-rows", type=int, default=None, help="Cap rows for smoke tests.")
    ap.add_argument("--sleep", type=float, default=0.35, help="Seconds between OpenRouter calls.")
    args = ap.parse_args()

    exclusive = [bool(args.rater_table), bool(args.fetch_llm), bool(args.csv)]
    if sum(int(x) for x in exclusive) > 1:
        print("ERROR: use exactly one of --rater-table, --fetch-llm, or --csv.", file=sys.stderr)
        sys.exit(1)

    fetch_llm_note = False

    if args.rater_table:
        if not args.rater_table.is_file():
            print(f"ERROR: rater table not found: {args.rater_table}", file=sys.stderr)
            sys.exit(1)
        df = pd.read_csv(args.rater_table)
        for c in ("biased_claude", "biased_gemini", "biased_author"):
            if c not in df.columns:
                print(f"ERROR: rater table must include column {c!r}.", file=sys.stderr)
                sys.exit(1)
        if df[["biased_claude", "biased_gemini", "biased_author"]].isna().any().any():
            print("ERROR: rater table has NaN in binary columns.", file=sys.stderr)
            sys.exit(1)
        bc = df["biased_claude"].astype(int).to_numpy()
        bg = df["biased_gemini"].astype(int).to_numpy()
        ba = df["biased_author"].astype(int).to_numpy()
    elif args.fetch_llm:
        fetch_llm_note = True
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            print("ERROR: set OPENROUTER_API_KEY (or add it to PRISM/.env).", file=sys.stderr)
            sys.exit(1)
        gold_path = args.gold
        if not gold_path.is_file():
            print(f"ERROR: gold file not found: {gold_path}", file=sys.stderr)
            sys.exit(1)
        df = pd.read_csv(gold_path)
        for col in ("post_id", "text", "label"):
            if col not in df.columns:
                print(f"ERROR: gold CSV must contain column {col!r}.", file=sys.stderr)
                sys.exit(1)
        if args.max_rows is not None:
            df = df.iloc[: int(args.max_rows)].copy()
        out_path = args.out_csv
        df = df.copy()
        df["biased_author"] = df["label"].map(four_way_to_binary)

        llm_cols = ("label_four_way_claude", "label_four_way_gemini", "biased_claude", "biased_gemini")
        for c in llm_cols:
            if c not in df.columns:
                df[c] = np.nan

        if out_path.is_file():
            prev = pd.read_csv(out_path)
            if "post_id" not in prev.columns:
                print(f"WARNING: existing {out_path} has no post_id; ignoring resume.", flush=True)
            else:
                pick = ["post_id"] + [c for c in llm_cols if c in prev.columns]
                prev_n = prev[pick].drop_duplicates(subset=["post_id"], keep="last")
                df = df.drop(columns=[c for c in llm_cols if c in df.columns], errors="ignore")
                df = df.merge(prev_n, on="post_id", how="left")

        def _cell_missing(x: object) -> bool:
            if pd.isna(x):
                return True
            return str(x).strip() == ""

        for side in ("claude", "gemini"):
            lf, bf = f"label_four_way_{side}", f"biased_{side}"
            m = df[lf].notna() & df[bf].isna()
            if m.any():
                df.loc[m, bf] = df.loc[m, lf].map(four_way_to_binary)

        n = len(df)
        errs_c = errs_g = 0
        idx = df.index
        for i in range(n):
            ii = idx[i]
            tx = str(df.at[ii, "text"])
            if _cell_missing(df.at[ii, "label_four_way_claude"]):
                lc, _raw_c = openrouter_classify(tx, args.claude_model, api_key)
                if lc is None:
                    errs_c += 1
                    lc = NEUTRAL_LABEL
                df.at[ii, "label_four_way_claude"] = lc
                df.at[ii, "biased_claude"] = four_way_to_binary(lc)
                time.sleep(args.sleep)
            if _cell_missing(df.at[ii, "label_four_way_gemini"]):
                lg, _raw_g = openrouter_classify(tx, args.gemini_model, api_key)
                if lg is None:
                    errs_g += 1
                    lg = NEUTRAL_LABEL
                df.at[ii, "label_four_way_gemini"] = lg
                df.at[ii, "biased_gemini"] = four_way_to_binary(lg)
                time.sleep(args.sleep)
            if (i + 1) % 25 == 0 or i == n - 1:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(out_path, index=False)
                print(f"[checkpoint] wrote {out_path} ({i+1}/{n})", flush=True)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Wrote rater table: {out_path}", flush=True)
        if errs_c or errs_g:
            print(
                f"NOTE: {errs_c} Claude + {errs_g} Gemini parse/http fallbacks to neutral "
                "(κ may be conservative).",
                flush=True,
            )

        for col in ("biased_claude", "biased_gemini"):
            if df[col].isna().any():
                print(
                    f"ERROR: column {col!r} has missing values after --fetch-llm "
                    "(interrupted run?). Re-run to resume from "
                    f"{out_path}.",
                    file=sys.stderr,
                )
                sys.exit(1)

        bc = df["biased_claude"].astype(int).to_numpy()
        bg = df["biased_gemini"].astype(int).to_numpy()
        ba = df["biased_author"].astype(int).to_numpy()
    else:
        if args.csv is None:
            print(
                "ERROR: choose one mode: --fetch-llm | --rater-table PATH | --csv PATH with "
                "--binary-cols or --four-way-cols.",
                file=sys.stderr,
            )
            sys.exit(1)
        if not args.csv.is_file():
            print(f"ERROR: CSV not found: {args.csv}", file=sys.stderr)
            sys.exit(1)
        df = pd.read_csv(args.csv)
        if args.binary_cols and args.four_way_cols:
            print("ERROR: pass only one of --binary-cols or --four-way-cols.", file=sys.stderr)
            sys.exit(1)
        if args.binary_cols:
            c1, c2, c3 = args.binary_cols
            for c in (c1, c2, c3):
                if c not in df.columns:
                    print(f"ERROR: missing column {c!r}.", file=sys.stderr)
                    sys.exit(1)
            bc = df[c1].astype(int).to_numpy()
            bg = df[c2].astype(int).to_numpy()
            ba = df[c3].astype(int).to_numpy()
        elif args.four_way_cols:
            c1, c2, c3 = args.four_way_cols
            for c in (c1, c2, c3):
                if c not in df.columns:
                    print(f"ERROR: missing column {c!r}.", file=sys.stderr)
                    sys.exit(1)
            bc = df[c1].map(four_way_to_binary).astype(int).to_numpy()
            bg = df[c2].map(four_way_to_binary).astype(int).to_numpy()
            ba = df[c3].map(four_way_to_binary).astype(int).to_numpy()
        else:
            print("ERROR: with --csv, supply --binary-cols or --four-way-cols (three names each).", file=sys.stderr)
            sys.exit(1)

    counts = np.column_stack(
        (
            3 - (bc + bg + ba),  # n unbiased
            bc + bg + ba,  # n biased
        )
    ).astype(np.int64)
    kappa = fleiss_kappa_binary(counts)
    pact = triple_agreement_rate(bc, bg, ba)

    print()
    print("=== Fleiss' κ (binary biased), three raters ===")
    print(f"N posts: {len(bc)}")
    print(f"Triple-exact agreement (all three same binary): {100.0 * pact:.2f}%")
    if not np.isnan(kappa):
        print(f"Fleiss' κ: {kappa:.4f}")
    else:
        print("Fleiss' κ: undefined (degenerate margins).")
    print(f"Landis & Koch reading: {landis_koch(kappa)}")
    print()
    if fetch_llm_note:
        print(
            "METHODOLOGY NOTE (for thesis):\n"
            "  The third rater is the adjudicated gold label from your benchmark CSV, not an\n"
            "  independent blind human pass collected before seeing the LLM outputs. Report κ as\n"
            "  agreement between two LLM first passes and the accepted adjudicated labels, and\n"
            "  point readers to Chapter 6 for limitations.\n"
        )
    kappa_s = "undefined" if np.isnan(kappa) else f"{kappa:.3f}"
    lk = landis_koch(kappa)
    print("--- Paste into thesis (edit model names if you overrode defaults) ---")
    if fetch_llm_note:
        caveat = (
            "Because LLM-based annotation can diverge from lay human intuitions, this statistic should be read as "
            "internal consistency of the adopted labelling protocol rather than validation against an external "
            "behavioural gold standard (see Chapter 6)."
        )
    else:
        caveat = (
            "As with any multi-rater study, this κ should be read alongside protocol limitations and edge-case "
            "definitions (see Chapter 6)."
        )
    print(
        f"… yielding κ = {kappa_s} on the binary bias judgement (Fleiss, three raters, N = {len(bc)}). "
        f"This indicates {lk} among the three label sources; triple-rater exact agreement on biased vs not "
        f"was {100.0 * pact:.1f}%. "
        + caveat
    )
    print("--- end paste ---")


if __name__ == "__main__":
    main()
