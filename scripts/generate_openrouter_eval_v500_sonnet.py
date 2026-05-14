#!/usr/bin/env python3
"""
Generate 500 balanced-label evaluation posts via OpenRouter + Claude Sonnet 4.6.

Requires OPENROUTER_API_KEY in PRISM/.env or environment.

Default model: anthropic/claude-sonnet-4.6 (override OPENROUTER_MODEL).

Method: **100 API calls**, each generating **5 posts of a single label** (25 waves × 4 labels).
The original 200-post run asked for exact multi-class counts in one CSV; models often miscount.
Here the model only writes one class per call; we **enforce** labels when saving.

Prompt text matches ``generate_openrouter_eval_v2_sonnet.py`` (same definitions and rules), scaled:
- 125 posts per class total → per 5-post batch: subtlety ~2 explicit / ~2 subtle / ~1 borderline;
  platform ~2–3 X-style / ~2–3 LinkedIn-style per batch.

Writes:
  - data/evaluation/manual_eval_v3_500_openrouter.json
  - data/evaluation/manual_eval_v3_500_posts.csv
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.request import Request, urlopen

_ROOT = Path(__file__).resolve().parents[1]
_OUT_DIR = _ROOT / "data" / "evaluation"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "anthropic/claude-sonnet-4.6"

POSTS_PER_CALL = 5
N_WAVES = 25  # 25 * 20 = 500


# Class definitions aligned with v2 (scaled per-batch instructions embedded in build_prompt).
def _build_prompt(label: str, id_list: list[str]) -> str:
    ids_joined = ", ".join(id_list)
    subt = "~2 explicit; ~2 subtle (coded, no words 'bias'/'stereotype'); ~1 borderline ambiguous"
    plat = "~2–3 Twitter/X (short, <280 chars, casual); ~2–3 LinkedIn (2–6 sentences, professional)"

    common_rules = r"""
## CRITICAL QUALITY RULES (apply to every post in this batch)

1. WHOLE-POST INTENT: If the post *refutes* or only *mentions* stereotyping without endorsing it, it is NOT
   gender_bias/nationality_bias/profession_bias — that content belongs in neutral posts, not in this batch.
   For THIS batch you are writing only posts whose **main stance matches the batch label** above.

2. SUBTLE biased posts must not use the words "bias" or "stereotype" in subtle examples.

3. BORDERLINE posts should be plausibly debatable for a human annotator.

4. Each of the 5 posts must be **distinct** in scenario, industry, tone, and claim.

5. COVER DIVERSE INDUSTRIES: healthcare, finance, law, education, retail, construction, hospitality,
   government, academia, creative, military, sports — not only tech.

6. NO hate speech, slurs, incitement, or real named individuals.

7. Output **only** valid CSV — header `post_id,text,label` then exactly 5 data rows.

8. The `label` field must be exactly `""" + label + r"""` on every row (ASCII, lowercase, underscores).

9. Double-quote CSV fields; escape `"` in text as `""`.
"""

    if label == "gender_bias":
        spec = rf"""
### gender_bias (these 5 posts)
Posts that reinforce, assume, or generalise based on gender — including but not limited to:
- Assuming men are more technical, logical, or suited for leadership
- Assuming women are less ambitious, less capable, or more emotional
- Gendered assumptions about family responsibilities
- Double standards in hiring or promotion
- Dismissing women's ideas or credentials based on gender
- Coded language: "she's surprisingly good for a woman", "he's a natural leader"
- Gender norms in non-tech sectors: nursing as female, engineering as male
- Assumptions about non-binary or gender-nonconforming individuals in workplaces

Within these **5** posts, spread: {subt}.
Platform mix: {plat}.
Where natural, include **at least one** post that touches non-binary, trans, or gender-nonconforming workplace bias.
"""
    elif label == "nationality_bias":
        spec = rf"""
### nationality_bias (these 5 posts)
Posts that generalise, stereotype, or make assumptions based on national origin,
immigration status, or ethnicity-as-nationality, including:
- Work ethic stereotypes tied to nationality
- Immigrants framed as threats, undercutters, or burdens
- Assumptions about assimilation, loyalty, or reliability from origin
- Positive stereotypes that still generalise
- Nationalities framed as more/less disciplined, ambitious, trustworthy
- Immigration as zero-sum threat stated as fact
- Subtle: "I always prefer hiring locally — you know what you're getting"

Within these **5** posts: {subt}.
Platform mix: {plat}.
Cover **varied regions** — South Asian, East Asian, African, Eastern European, Latin American, Middle Eastern —
not only US/UK.
"""
    elif label == "profession_bias":
        spec = rf"""
### profession_bias (these 5 posts)
Posts that assign worth, intelligence, status, or moral value based on professional identity or education:
- Vocational/trade vs university graduates
- Teachers, nurses, social workers framed as less intelligent or valuable
- "MBA brain" / consultants mocked
- Artists, humanities dismissed as unserious
- Engineers or STEM assumed smarter
- Sector hierarchies: finance > tech > arts > trades
- Manual workers stereotyped as lacking ambition
- Academic elitism (Ivy / Oxbridge)
- Subtle: "Oh you studied philosophy? That's... brave."

Within these **5** posts: {subt}.
Platform mix: {plat}.
Include **intra-professional** snobbery sometimes (not only white- vs blue-collar).
"""
    else:  # neutral
        spec = rf"""
### neutral (these 5 posts)
Posts that discuss bias, diversity, hiring, immigration, or professional topics **without**
reinforcing any harmful stereotype. Includes:
- Counter-speech naming and rejecting a stereotype
- Policy discussion without demeaning groups
- Inclusive observations that don't generalise harmful traits
- Workplace commentary without group-level assumptions
- Structural change, research, reform
- Career/hiring advice without biased framing

Sound like real people — not preachy or robotic. **Not** all counter-speech: mix policy debate, career reflection,
plain hiring observations without group generalisation.

Within these **5** posts, vary length and tone. Platform mix: {plat}.
"""

    return f"""You are a data generation assistant helping build a bias detection evaluation dataset.

## SINGLE-BATCH TASK

Generate **exactly {POSTS_PER_CALL}** social media posts — a realistic mix of Twitter/X-style (short, casual,
sometimes hashtags) and LinkedIn-style (longer, professional, reflective).

**Every post in this batch must belong to the class `{label}`** and match its definition.

## POST IDs (use exactly these five, in this order)

{ids_joined}

{spec}

{common_rules}

## CSV OUTPUT

post_id,text,label

Include the header row, then exactly 5 rows. Nothing before the header; nothing after the last row.
"""


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(_ROOT / ".env")
    except ImportError:
        pass


def _strip_code_fences(raw: str) -> str:
    t = raw.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:csv)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def _call_openrouter(api_key: str, model: str, user_content: str) -> str:
    body = json.dumps(
        {
            "model": model,
            "temperature": 0.68,
            "max_tokens": 16384,
            "messages": [
                {
                    "role": "system",
                    "content": "You follow instructions exactly. Output only the CSV requested; no preamble or suffix.",
                },
                {"role": "user", "content": user_content},
            ],
        }
    ).encode("utf-8")
    req = Request(OPENROUTER_URL, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("HTTP-Referer", "https://github.com/")
    req.add_header("X-Title", "PRISM eval v3 single-class")
    with urlopen(req, timeout=900) as r:
        resp = json.loads(r.read().decode("utf-8"))
    choices = resp.get("choices") or []
    if not choices:
        raise RuntimeError(f"OpenRouter: no choices in response keys={list(resp.keys())}")
    return str(choices[0].get("message", {}).get("content") or "")


def _parse_csv_loose(content: str) -> list[dict[str, str]]:
    text = _strip_code_fences(content)
    reader = csv.DictReader(io.StringIO(text))
    rows: list[dict[str, str]] = []
    for row in reader:
        pid = str(row.get("post_id", "")).strip()
        txt = str(row.get("text", "")).strip()
        lab = str(row.get("label", "")).strip()
        rows.append({"post_id": pid, "text": txt, "label": lab})
    return rows


def _sort_key_post_id(post_id: str) -> tuple[int, str]:
    m = re.match(r"^p3_(\d+)$", str(post_id).strip())
    if m:
        return (int(m.group(1)), str(post_id))
    return (99999, str(post_id))


def _validate_single_class(
    rows: list[dict[str, str]],
    expected_ids: list[str],
    forced_label: str,
) -> list[str]:
    errs: list[str] = []
    if len(rows) != POSTS_PER_CALL:
        errs.append(f"expected {POSTS_PER_CALL} rows, got {len(rows)}")
    rows = sorted(rows, key=lambda r: _sort_key_post_id(r["post_id"]))
    got_ids = [r["post_id"] for r in rows]
    if got_ids != expected_ids:
        errs.append(f"post_id sequence: want {expected_ids}, got {got_ids}")
    for r in rows:
        if not r["text"].strip():
            errs.append(f"empty text for {r['post_id']}")
        # label checked at save time
    return errs


def main() -> int:
    _load_dotenv()
    api_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if not api_key:
        print("Set OPENROUTER_API_KEY in .env", file=sys.stderr)
        return 2

    model = (os.environ.get("OPENROUTER_MODEL") or DEFAULT_MODEL).strip()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    archive_calls: list[dict[str, object]] = []
    all_rows: list[dict[str, str]] = []
    label_order = ["gender_bias", "nationality_bias", "profession_bias", "neutral"]
    max_attempts = 5
    call_idx = 0

    for wave in range(N_WAVES):
        base = wave * 20
        for j, label in enumerate(label_order):
            lo = base + j * POSTS_PER_CALL + 1
            hi = base + (j + 1) * POSTS_PER_CALL
            expected_ids = [f"p3_{k:03d}" for k in range(lo, hi + 1)]
            prompt = _build_prompt(label, expected_ids)
            call_idx += 1

            for attempt in range(1, max_attempts + 1):
                print(
                    f"Call {call_idx}/100  wave {wave + 1}/{N_WAVES}  "
                    f"{label}  p3_{lo:03d}–p3_{hi:03d}  attempt {attempt}/{max_attempts}",
                    flush=True,
                )
                raw = _call_openrouter(api_key, model, prompt)
                archive_calls.append(
                    {
                        "call": call_idx,
                        "wave": wave,
                        "label": label,
                        "expected_ids": expected_ids,
                        "attempt": attempt,
                        "raw_assistant_content": raw,
                    }
                )
                rows = _parse_csv_loose(raw)
                errs = _validate_single_class(rows, expected_ids, label)
                if not errs:
                    for r in sorted(rows, key=lambda x: _sort_key_post_id(x["post_id"])):
                        all_rows.append(
                            {"post_id": r["post_id"], "text": r["text"], "label": label}
                        )
                    break
                print("  errors:", errs, flush=True)
                if attempt < max_attempts:
                    time.sleep(2.0)
            else:
                outp = _OUT_DIR / "manual_eval_v3_500_openrouter.json"
                outp.write_text(json.dumps({"model": model, "calls": archive_calls}, indent=2), encoding="utf-8")
                print(json.dumps({"ok": False, "failed_wave": wave, "label": label}, indent=2))
                return 1
            time.sleep(0.35)

    final_errs: list[str] = []
    if len(all_rows) != 500:
        final_errs.append(f"Expected 500 rows, got {len(all_rows)}")
    counts = Counter(r["label"] for r in all_rows)
    for lab in label_order:
        if counts.get(lab, 0) != 125:
            final_errs.append(f"{lab}: expected 125, got {counts.get(lab, 0)}")
    exp = [f"p3_{i:03d}" for i in range(1, 501)]
    got = [r["post_id"] for r in all_rows]
    if got != exp:
        final_errs.append("Final ID list mismatch")

    archive_path = _OUT_DIR / "manual_eval_v3_500_openrouter.json"
    archive_path.write_text(
        json.dumps(
            {
                "model": model,
                "provider": "openrouter.ai",
                "method": "100_calls_x5_single_label",
                "calls": archive_calls,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    out_csv = _OUT_DIR / "manual_eval_v3_500_posts.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["post_id", "text", "label"])
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    print(json.dumps(
        {
            "csv": str(out_csv),
            "archive": str(archive_path),
            "model": model,
            "n_rows": len(all_rows),
            "label_counts": dict(counts),
            "validation_errors": final_errs,
            "valid": len(final_errs) == 0,
        },
        indent=2,
    ))
    return 0 if not final_errs else 1


if __name__ == "__main__":
    raise SystemExit(main())
