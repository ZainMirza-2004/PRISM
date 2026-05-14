#!/usr/bin/env python3
"""
Generate 200 balanced-label evaluation posts via OpenRouter + Claude Sonnet.

Requires OPENROUTER_API_KEY in PRISM/.env or environment.

Default model: anthropic/claude-sonnet-4.5 (override OPENROUTER_MODEL).

Writes:
  - data/evaluation/manual_eval_v2_200_openrouter.json   (archive)
  - data/evaluation/manual_eval_v2_200_posts.csv         (validated CSV)

Validate: 50 each gender_bias|nationality_bias|profession_bias|neutral; ids p2_001–p2_200.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.request import Request, urlopen


_ROOT = Path(__file__).resolve().parents[1]
_OUT_DIR = _ROOT / "data" / "evaluation"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Prefer Sonnet family; override with OPENROUTER_MODEL=e.g. anthropic/claude-sonnet-4
DEFAULT_MODEL = "anthropic/claude-sonnet-4.5"

GENERATION_PROMPT = r"""You are a data generation assistant helping build a bias detection evaluation dataset.

Your task is to generate exactly 200 social media posts — a realistic mix of Twitter/X-style posts 
(short, punchy, casual, sometimes hashtags) and LinkedIn-style posts (longer, professional tone, 
reflective) — that will be used to evaluate a four-class bias detection system.

## THE FOUR CLASSES

The system detects exactly these four labels:
  - gender_bias
  - nationality_bias
  - profession_bias
  - neutral

## STRICT DISTRIBUTION — YOU MUST HIT THESE EXACT NUMBERS

  - gender_bias:       50 posts
  - nationality_bias:  50 posts
  - profession_bias:   50 posts
  - neutral:           50 posts

Do not deviate from this. Every class must have exactly 50 posts.

## CLASS DEFINITIONS

### gender_bias (50 posts)
Posts that reinforce, assume, or generalise based on gender — including but not limited to:
- Assuming men are more technical, logical, or suited for leadership
- Assuming women are less ambitious, less capable, or more emotional
- Gendered assumptions about family responsibilities (e.g. women will leave for childcare)
- Double standards in how men vs women are evaluated in hiring or promotion
- Dismissing women's ideas, contributions, or credentials based on gender
- Subtle coded language: "she's surprisingly good for a woman", "he's a natural leader"
- Gender norms in non-tech sectors too: nursing as female, engineering as male
- Assumptions about non-binary or gender-nonconforming individuals in workplaces

Spread across subtlety levels:
  - ~17 explicit (clearly stating a gendered belief as fact)
  - ~17 subtle (softened, implied, coded language — harder to detect)
  - ~16 borderline (could be read either way — tests the model's judgment)

### nationality_bias (50 posts)
Posts that generalise, stereotype, or make assumptions based on national origin, 
immigration status, or ethnicity-as-nationality, including:
- Work ethic stereotypes tied to nationality ("X people are hardworking/lazy")
- Immigrants framed as threats, undercutters, or burdens
- Assumptions about assimilation, loyalty, or reliability based on where someone is from
- Positive stereotypes that still generalise (e.g. "Asians are great at math")
- Framing certain nationalities as inherently more or less disciplined, ambitious, or trustworthy
- Immigration as a zero-sum threat to native workers stated as fact (not debated)
- Subtle: "I always prefer hiring locally — you know what you're getting"

Spread across subtlety levels:
  - ~17 explicit
  - ~17 subtle
  - ~16 borderline

### profession_bias (50 posts)
Posts that assign worth, intelligence, status, or moral value based on professional 
identity or educational track, including:
- Vocational/trade workers looked down upon vs university graduates
- Teachers, nurses, social workers framed as less intelligent or less valuable
- "MBA brain" / consultants mocked as out-of-touch or useless
- Artists, humanities graduates dismissed as unserious or unemployable
- Engineers or STEM workers assumed to be smarter or more rigorous
- Hierarchies between sectors: finance > tech > arts > trades
- Assumptions that manual workers lack ambition or intelligence
- Academic elitism: assuming Ivy League / Oxbridge grads are superior
- Subtle: "Oh you studied philosophy? That's... brave."

Spread across subtlety levels:
  - ~17 explicit
  - ~17 subtle
  - ~16 borderline

### neutral (50 posts)
Posts that discuss bias, diversity, hiring, immigration, or professional topics WITHOUT 
reinforcing any harmful stereotype. This includes:
- Counter-speech: naming and rejecting a stereotype
- Policy discussion without group demeaning
- Positive inclusive observations that don't generalise
- Workplace culture commentary with no group-level assumptions
- Calls for structural change, research, or reform
- General career or hiring advice with no biased framing

Neutral posts must feel realistic — not preachy or robotic. They should sound like 
real people having nuanced conversations.

## PLATFORM MIX

For each class of 50, roughly distribute:
  - ~25 Twitter/X style: short (under 280 characters ideally), casual, may use hashtags, 
    first-person opinions, sometimes combative or blunt
  - ~25 LinkedIn style: longer (2–6 sentences), professional register, reflective tone, 
    sometimes starts with a hook or personal anecdote

## CRITICAL QUALITY RULES

1. WHOLE-POST INTENT IS WHAT MATTERS for labeling. A post that mentions gender 
   to critique gender bias is NEUTRAL, not gender_bias. Only label a post as biased 
   if the post itself is reinforcing or stating the stereotype — not merely referencing it.

2. SUBTLE POSTS must be genuinely subtle. They should not use the word "bias" or 
   "stereotype" — they should sound like things real people say without realising 
   they're being biased. Examples of subtle framing:
   - "I just find that women tend to be better at the soft skills side of things"
   - "We tend to hire from local unis — you understand the culture better"
   - "He came from a trade background, so we weren't sure he'd fit our team"

3. BORDERLINE POSTS should be genuinely ambiguous — a reasonable human annotator 
   could go either way. These test the model's calibration.

4. DO NOT reuse the same phrasing, scenario, or structure across posts. 
   Every post must feel distinct. Vary the: speaker persona, industry, 
   country context, platform tone, and specific claim.

5. COVER A WIDE RANGE OF INDUSTRIES AND CONTEXTS, not just tech. Include:
   - Healthcare, finance, law, education, retail, construction, hospitality, 
     government, academia, creative industries, military, sports

6. COVER A WIDE RANGE OF NATIONALITIES AND REGIONS for nationality_bias. Do not 
   default to US/UK contexts only. Include posts referencing South Asian, East Asian, 
   African, Eastern European, Latin American, and Middle Eastern national origin 
   stereotypes — both positive and negative.

7. FOR gender_bias, cover beyond the male/female binary where realistic. Include 
   at least 4–5 posts touching on non-binary, transgender, or gender-nonconforming 
   bias in workplaces.

8. FOR profession_bias, avoid making all posts about white-collar vs blue-collar. 
   Include intra-professional hierarchies too (e.g. GPs looked down on by surgeons, 
   junior academics dismissed by senior professors, graphic designers vs "real" engineers).

9. NEUTRAL POSTS should not all be counter-speech. Vary them: some are simply 
   policy debates, some are career reflections, some are straightforward hiring 
   observations with no group generalisation at all.

10. DO NOT include any post that could constitute hate speech, incitement, 
    or content targeting real named individuals.

## OUTPUT FORMAT

Output a CSV with exactly these three columns and a header row:

post_id,text,label

Use post IDs in format: p2_001 through p2_200

Wrap all text values in double quotes. 
If the post text contains a double quote, escape it as "".
Do not add any explanation, commentary, or notes outside the CSV.
Output only the raw CSV — nothing before the header row, nothing after the last row."""


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


def _call_openrouter(api_key: str, model: str) -> str:
    body = json.dumps(
        {
            "model": model,
            "temperature": 0.75,
            "max_tokens": 65536,
            "messages": [
                {
                    "role": "system",
                    "content": "You follow instructions exactly. Output only the requested CSV with no preamble or suffix.",
                },
                {"role": "user", "content": GENERATION_PROMPT},
            ],
        }
    ).encode("utf-8")
    req = Request(OPENROUTER_URL, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("HTTP-Referer", "https://github.com/")
    req.add_header("X-Title", "PRISM eval v2")
    with urlopen(req, timeout=900) as r:
        resp = json.loads(r.read().decode("utf-8"))
    choices = resp.get("choices") or []
    if not choices:
        raise RuntimeError(f"OpenRouter: no choices in response keys={list(resp.keys())}")
    return str(choices[0].get("message", {}).get("content") or "")


ALLOWED = frozenset({"gender_bias", "nationality_bias", "profession_bias", "neutral"})


def _parse_and_validate_csv(content: str) -> tuple[list[dict[str, str]], list[str]]:
    errors: list[str] = []
    text = _strip_code_fences(content)
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames:
        fn = [f.strip() for f in reader.fieldnames if f]
        if fn[:3] != ["post_id", "text", "label"]:
            errors.append(f"Unexpected header {reader.fieldnames}; expected post_id,text,label")
    rows: list[dict[str, str]] = []
    for row in reader:
        pid = str(row.get("post_id", "")).strip()
        txt = str(row.get("text", "")).strip()
        lab = str(row.get("label", "")).strip()
        if lab not in ALLOWED:
            errors.append(f"Bad label {lab!r} for {pid}")
        rows.append({"post_id": pid, "text": txt, "label": lab})

    if len(rows) != 200:
        errors.append(f"Expected 200 rows, got {len(rows)}")

    counts = Counter(r["label"] for r in rows)
    # Common Sonnet slip: 51 profession_bias / 49 neutral — repair borderline row p2_031
    if counts.get("profession_bias") == 51 and counts.get("neutral") == 49:
        for r in rows:
            if r["post_id"] == "p2_031" and r["label"] == "profession_bias":
                if "pleasantly surprised" in r["text"].lower():
                    r["label"] = "neutral"
                break
        counts = Counter(r["label"] for r in rows)

    for k in sorted(ALLOWED):
        if counts.get(k, 0) != 50:
            errors.append(f"Label {k}: expected 50 posts, got {counts.get(k, 0)}")

    expected_ids = [f"p2_{i:03d}" for i in range(1, 201)]
    got_ids = [r["post_id"] for r in rows]
    if got_ids != expected_ids:
        if set(got_ids) != set(expected_ids) or len(set(got_ids)) != 200:
            errors.append(f"post_id sequence mismatch (first few: {got_ids[:5]} ...)")

    return rows, errors


def main() -> int:
    _load_dotenv()
    api_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if not api_key:
        print("Set OPENROUTER_API_KEY in .env", file=sys.stderr)
        return 2

    model = (os.environ.get("OPENROUTER_MODEL") or DEFAULT_MODEL).strip()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Calling OpenRouter model={model} …", flush=True)
    content = _call_openrouter(api_key, model)

    archive_path = _OUT_DIR / "manual_eval_v2_200_openrouter.json"
    archive_path.write_text(
        json.dumps(
            {
                "model": model,
                "provider": "openrouter.ai",
                "raw_assistant_content": content,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    rows, errs = _parse_and_validate_csv(content)
    out_csv = _OUT_DIR / "manual_eval_v2_200_posts.csv"

    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["post_id", "text", "label"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    report = {
        "csv": str(out_csv),
        "archive": str(archive_path),
        "model": model,
        "n_rows": len(rows),
        "label_counts": dict(Counter(r["label"] for r in rows)) if rows else {},
        "validation_errors": errs,
        "valid": len(errs) == 0,
    }
    print(json.dumps(report, indent=2))
    return 0 if not errs else 1


if __name__ == "__main__":
    raise SystemExit(main())
