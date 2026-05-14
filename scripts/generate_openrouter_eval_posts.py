#!/usr/bin/env python3
"""
Generate 200 evaluation posts via OpenRouter (openai/gpt-4.1-mini).

Requires OPENROUTER_API_KEY in the environment or .env alongside PRISM/.

Writes:
  - data/evaluation/manual_eval_200_generated.json   (API response archive)
  - data/evaluation/manual_eval_200_posts.csv         (post_id, text, label blanks)
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from pathlib import Path
from urllib.request import Request, urlopen


_ROOT = Path(__file__).resolve().parents[1]
_OUT_DIR = _ROOT / "data" / "evaluation"

SYSTEM_USER_PROMPT = r"""You are generating a high-quality evaluation dataset for a research system that detects bias in text.

Your task is to produce 200 independent social media posts (like X/Twitter or LinkedIn posts).

The posts will be manually annotated later, so they must be realistic, diverse, and not trivially obvious.

---

## IMPORTANT GOAL

We are NOT trying to make obvious biased or neutral statements.

We are trying to create a dataset that includes:

1. Neutral posts (no bias)
2. Subtle bias (implicit generalisations, soft stereotypes)
3. Explicit bias (clear stereotyping or exclusionary claims)
4. Counter-bias / anti-stereotype posts (rebuttals or corrections)

The dataset must include ambiguity and realistic human tone variation.

---

## TOPICS TO INCLUDE

- immigration and labor markets
- gender and hiring in tech/business
- nationality and work ethic stereotypes
- education and STEM talent
- workplace culture and "fit"
- economic policy and labor competition
- startups and global talent mobility

---

## CRITICAL RULES

- Each post must be between 1–3 sentences
- Must sound like real social media writing
- Must NOT mention that it is AI-generated
- Must NOT include explanations or labels
- Must NOT be repetitive or templated
- Must NOT always be extreme — include nuance and subtlety
- Must include disagreement, opinions, and uncertainty in some posts
- Mix tones: casual, professional, argumentative, reflective

---

## BALANCE REQUIREMENT (IMPORTANT)

Approximate distribution across 200 posts:

- 50 posts: neutral (informational or opinion without bias)
- 50 posts: subtle bias (implicit generalisations, soft stereotypes)
- 50 posts: explicit bias (clear group generalisations or exclusionary framing)
- 50 posts: counter-bias / rebuttal (challenging stereotypes or correcting them)

---

## OUTPUT FORMAT (STRICT)

Return ONLY a valid JSON array.

Each item must be:

{
  "id": "p1_001",
  "text": "post content here"
}

No extra commentary. No headings. No markdown.

IDs must be sequential from p1_001 to p1_200.

---

## QUALITY STANDARD

Posts should resemble real-world X / LinkedIn discourse:
- some emotionally charged
- some analytical
- some sarcastic or rhetorical
- some neutral informational

Avoid being overly clean or textbook-like.
Real human inconsistency is required.

Begin now."""

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(_ROOT / ".env")
    except ImportError:
        pass


def _strip_json_fences(raw: str) -> str:
    t = raw.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def _call_openrouter(api_key: str) -> str:
    body = json.dumps(
        {
            "model": "openai/gpt-4.1-mini",
            "temperature": 0.85,
            "max_tokens": 32000,
            "messages": [
                {"role": "system", "content": "Follow the user instructions exactly. Output parseable JSON only."},
                {"role": "user", "content": SYSTEM_USER_PROMPT},
            ],
        }
    ).encode("utf-8")
    req = Request(OPENROUTER_URL, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("HTTP-Referer", "https://github.com/")
    req.add_header("X-Title", "PRISM eval generator")
    with urlopen(req, timeout=600) as r:
        resp = json.loads(r.read().decode("utf-8"))
    choices = resp.get("choices") or []
    if not choices:
        raise RuntimeError(f"OpenRouter: no choices: {resp!r:.20000}")
    return str(choices[0].get("message", {}).get("content") or "")


def _parse_posts(content: str) -> list[dict]:
    raw = _strip_json_fences(content)
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("Parsed JSON root must be an array.")
    out: list[dict] = []
    for row in data:
        if isinstance(row, dict) and "id" in row and "text" in row:
            out.append({"id": str(row["id"]).strip(), "text": str(row["text"]).strip()})
        else:
            raise ValueError(f"Bad row: {row!r:.120}")
    return out


def _validate_exactly_200(rows: list[dict]) -> None:
    if len(rows) != 200:
        raise ValueError(f"Expected 200 posts, got {len(rows)}")
    expected = {f"p1_{i:03d}" for i in range(1, 201)}
    got = [str(r["id"]).strip() for r in rows]
    if len(set(got)) != 200 or set(got) != expected:
        raise ValueError(f"IDs must be exactly {{p1_001..p1_200}}, unique; got troubles in sample: {got[:5]}…")


def _sorted_by_id(rows: list[dict]) -> list[dict]:
    def key(r: dict) -> tuple[int, ...]:
        m = re.match(r"^p1_(\d+)$", str(r["id"]).strip())
        return (int(m.group(1)),) if m else (999,)

    return sorted(rows, key=key)


def main() -> int:
    _load_dotenv()
    api_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if not api_key:
        print("Set OPENROUTER_API_KEY (e.g. in PRISM/.env)", file=sys.stderr)
        return 2

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Calling OpenRouter (openai/gpt-4.1-mini)...", flush=True)
    content = _call_openrouter(api_key)

    archive = {
        "model": "openai/gpt-4.1-mini",
        "provider": "openrouter.ai",
        "raw_assistant_content": content,
    }
    (_OUT_DIR / "manual_eval_200_generated.json").write_text(json.dumps(archive, indent=2), encoding="utf-8")

    rows = _parse_posts(content)
    _validate_exactly_200(rows)
    rows = _sorted_by_id(rows)

    csv_path = _OUT_DIR / "manual_eval_200_posts.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["post_id", "text", "label"])
        w.writeheader()
        for r in rows:
            w.writerow({"post_id": r["id"], "text": r["text"], "label": ""})

    print(json.dumps({"ok": True, "csv": str(csv_path), "n": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
