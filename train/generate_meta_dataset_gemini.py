"""Generate a high-quality 300-row meta-classifier dataset using Gemini API.

Output:
  data/meta_training/gemini_300_posts.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = _ROOT / "data" / "meta_training" / "gemini_300_posts.csv"

LABELS = ("neutral", "gender_bias", "nationality_bias", "profession_bias")
PLATFORMS = ("x", "linkedin")
TOTAL_PER_LABEL = 75

# Style mix: close to 50/50 while preserving strict class totals.
PER_PLATFORM_BY_LABEL = {
    "neutral": {"x": 38, "linkedin": 37},
    "gender_bias": {"x": 37, "linkedin": 38},
    "nationality_bias": {"x": 38, "linkedin": 37},
    "profession_bias": {"x": 37, "linkedin": 38},
}

SYSTEM_PROMPT = """You are a dataset generator for social-bias classification research.

TASK
- Generate realistic social-media posts and assign one label per post.
- You are NOT a chatbot. Do not explain or discuss. Output only JSON.

LABEL DEFINITIONS
- neutral: no stereotyping against gender, nationality, or profession groups.
- gender_bias: subtle/implicit stereotype, preference, or generalisation tied to gender.
- nationality_bias: subtle/implicit stereotype, preference, or generalisation tied to nationality/immigration/origin.
- profession_bias: subtle/implicit stereotype, preference, or generalisation tied to occupation/profession/role group.

QUALITY RULES
- Prefer subtle, implicit, realistic bias (not explicit hate speech/slurs).
- Include natural patterns such as soft preferences, coded phrasing, implicit generalisations.
- For LinkedIn style, include polished corporate tone and workplace context where appropriate.
- Avoid repetitive templates, repeated openings, or robotic wording.
- Keep text clean (no markdown bullets, no numbering, no labels in text, no code fences).
- Use varied topics across hiring, tech, education, politics, economy, culture, workplace dynamics, and adjacent public discourse.
- Keep posts concise and realistic for the requested platform.
- For neutral, include many group-mention posts that are policy-focused, pro-inclusion, and anti-stereotype.
- Avoid labeling supportive group statements as bias.

OUTPUT FORMAT
- Return ONLY a JSON array.
- Each item must be: {"text": "<post>", "label": "<label>"}.
- Label must be exactly one of: neutral, gender_bias, nationality_bias, profession_bias.
"""


class RequestPacer:
    """Simple request-rate limiter (requests per minute)."""

    def __init__(self, rpm: int) -> None:
        if rpm <= 0:
            raise ValueError("rpm must be > 0")
        self._min_interval_s = 60.0 / float(rpm)
        self._last_call_ts = 0.0

    def wait_turn(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_call_ts
        wait_s = self._min_interval_s - elapsed
        if wait_s > 0:
            time.sleep(wait_s)
        self._last_call_ts = time.monotonic()


def _load_dotenv() -> None:
    """Load .env if python-dotenv is installed; no-op otherwise."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = _ROOT / ".env"
    if env_path.is_file():
        load_dotenv(env_path)


def _require_api_key() -> str:
    _load_dotenv()
    key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError(
            "Missing GEMINI_API_KEY. Add it to PRISM/.env or environment before running."
        )
    return key


def _require_openrouter_key() -> str:
    _load_dotenv()
    key = (
        os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENROUTER_KEY")
        or ""
    ).strip()
    if not key:
        raise RuntimeError(
            "Missing OPENROUTER_API_KEY. Add it to PRISM/.env or environment before running."
        )
    return key


def _request_payload(label: str, platform: str, count: int) -> Dict:
    platform_hint = (
        "X/Twitter style: short, punchy, casual, internet-native phrasing."
        if platform == "x"
        else "LinkedIn style: professional, corporate, workplace and hiring context."
    )
    user_prompt = (
        f"Generate exactly {count} unique posts.\n"
        f"Target label for every item: {label}.\n"
        f"Platform style for every item: {platform}.\n"
        f"{platform_hint}\n"
        "Return only a JSON array of objects with keys text and label.\n"
        "No extra keys, no markdown, no comments."
    )

    return {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": 1.05,
            "topP": 0.95,
            "responseMimeType": "application/json",
        },
    }


def _extract_json_array(text: str) -> List[Dict[str, str]]:
    def _coerce(obj: object) -> List[Dict[str, str]]:
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            for k in ("data", "items", "posts", "results", "samples"):
                v = obj.get(k)
                if isinstance(v, list):
                    return v
            if "text" in obj and "label" in obj:
                return [obj]  # tolerate single-object payload
        return []

    raw = (text or "").strip()
    if not raw:
        return []
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        payload = json.loads(raw)
        return _coerce(payload)
    except json.JSONDecodeError:
        pass

    start = raw.find("[")
    end = raw.rfind("]")
    if start >= 0 and end > start:
        snippet = raw[start : end + 1]
        try:
            payload = json.loads(snippet)
            return _coerce(payload)
        except json.JSONDecodeError:
            return []
    return []


def _normalize_text(text: str) -> str:
    t = str(text or "")
    t = t.replace("\u200b", " ").replace("\ufeff", " ")
    t = t.replace("\r", " ").replace("\n", " ")
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"^\s*[-*•\d\.\)\(]+\s*", "", t)
    t = t.strip(" \"'")
    return t


def _request_json(url: str, payload: Dict, headers: Dict[str, str], timeout_s: int) -> Dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _openrouter_list_models(api_key: str, timeout_s: int = 30) -> List[str]:
    url = "https://openrouter.ai/api/v1/models"
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    ids: List[str] = []
    for row in data.get("data", []):
        model_id = str(row.get("id", "")).strip()
        if model_id:
            ids.append(model_id)
    return ids


def _select_best_openrouter_model(available: List[str]) -> str:
    preferred = [
        "openai/gpt-5",
        "openai/o3-pro",
        "openai/o3",
        "anthropic/claude-opus-4.1",
        "anthropic/claude-opus-4",
        "google/gemini-2.5-pro",
        "anthropic/claude-sonnet-4",
        "google/gemini-2.5-flash",
        "openai/gpt-4.1",
        "meta-llama/llama-4-maverick",
    ]
    avail_set = set(available)
    for m in preferred:
        if m in avail_set:
            return m
    if available:
        return sorted(available)[0]
    raise RuntimeError("No OpenRouter models available for this API key.")


def _call_gemini_with_backoff(
    api_key: str,
    model: str,
    payload: Dict,
    timeout_s: int,
    pacer: RequestPacer,
    *,
    max_retries: int,
    base_backoff_s: float,
    max_backoff_s: float,
) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    body = json.dumps(payload).encode("utf-8")
    attempt = 0
    while True:
        attempt += 1
        pacer.wait_turn()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            is_retryable = exc.code in {429, 500, 502, 503, 504}
            if not is_retryable or attempt > max_retries:
                raise RuntimeError(f"Gemini HTTP error {exc.code}: {exc.reason}") from exc
            retry_after = 0.0
            try:
                ra = (exc.headers.get("Retry-After") or "").strip()
                retry_after = float(ra) if ra else 0.0
            except (TypeError, ValueError):
                retry_after = 0.0
            exp = min(base_backoff_s * (2 ** (attempt - 1)), max_backoff_s)
            jitter = random.uniform(0.0, 0.75)
            time.sleep(max(retry_after, exp + jitter))
            continue
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt > max_retries:
                raise RuntimeError(f"Gemini request failed after retries: {exc}") from exc
            exp = min(base_backoff_s * (2 ** (attempt - 1)), max_backoff_s)
            jitter = random.uniform(0.0, 0.75)
            time.sleep(exp + jitter)
            continue
        break

    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"Gemini response missing candidates: {json.dumps(data)[:600]}")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(str(p.get("text", "")) for p in parts)
    if not text.strip():
        raise RuntimeError(f"Gemini returned empty text: {json.dumps(data)[:600]}")
    return text


def _call_openrouter_with_backoff(
    api_key: str,
    model: str,
    payload: Dict,
    timeout_s: int,
    pacer: RequestPacer,
    *,
    max_retries: int,
    base_backoff_s: float,
    max_backoff_s: float,
) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Generate exactly {payload['count']} unique posts.\n"
                f"Target label for every item: {payload['label']}.\n"
                f"Platform style for every item: {payload['platform']}.\n"
                "Return ONLY a JSON object with this exact shape:\n"
                "{\"items\": [{\"text\": \"...\", \"label\": \"...\"}]}\n"
                f"The items array must contain exactly {payload['count']} objects.\n"
                "No extra keys, no markdown, no comments."
            ),
        },
    ]
    body = {
        "model": model,
        "messages": messages,
        "temperature": 1.0,
        "top_p": 0.95,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://local.cognitive-bias-detector",
        "X-Title": "PRISM",
    }

    attempt = 0
    while True:
        attempt += 1
        pacer.wait_turn()
        try:
            data = _request_json(url=url, payload=body, headers=headers, timeout_s=timeout_s)
        except urllib.error.HTTPError as exc:
            is_retryable = exc.code in {429, 500, 502, 503, 504}
            if not is_retryable or attempt > max_retries:
                raise RuntimeError(f"OpenRouter HTTP error {exc.code}: {exc.reason}") from exc
            exp = min(base_backoff_s * (2 ** (attempt - 1)), max_backoff_s)
            time.sleep(exp + random.uniform(0.0, 0.75))
            continue
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt > max_retries:
                raise RuntimeError(f"OpenRouter request failed after retries: {exc}") from exc
            exp = min(base_backoff_s * (2 ** (attempt - 1)), max_backoff_s)
            time.sleep(exp + random.uniform(0.0, 0.75))
            continue
        break

    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError(f"OpenRouter response missing choices: {json.dumps(data)[:600]}")
    content = str(choices[0].get("message", {}).get("content", ""))
    if not content.strip():
        raise RuntimeError(f"OpenRouter returned empty content: {json.dumps(data)[:600]}")
    return content


def _generate_bucket(
    api_key: str,
    model: str,
    label: str,
    platform: str,
    target_count: int,
    *,
    pacer: RequestPacer,
    max_retries: int,
    base_backoff_s: float,
    max_backoff_s: float,
    provider: str,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    seen = set()
    attempts = 0
    max_attempts = 12

    print(f"[START] {label} | {platform} | target={target_count}", flush=True)
    while len(rows) < target_count and attempts < max_attempts:
        attempts += 1
        if attempts > 6:
            raise RuntimeError("Too many attempts — stopping to avoid API overuse")
        remaining = target_count - len(rows)
        request_n = min(remaining + 8, 25)
        payload = _request_payload(label=label, platform=platform, count=request_n)
        print(
            f"[ATTEMPT] {label} | {platform} | attempt={attempts} | remaining={remaining} | request_n={request_n}",
            flush=True,
        )

        try:
            if provider == "gemini":
                text = _call_gemini_with_backoff(
                    api_key=api_key,
                    model=model,
                    payload=payload,
                    timeout_s=90,
                    pacer=pacer,
                    max_retries=max_retries,
                    base_backoff_s=base_backoff_s,
                    max_backoff_s=max_backoff_s,
                )
            else:
                text = _call_openrouter_with_backoff(
                    api_key=api_key,
                    model=model,
                    payload={"label": label, "platform": platform, "count": request_n},
                    timeout_s=90,
                    pacer=pacer,
                    max_retries=max_retries,
                    base_backoff_s=base_backoff_s,
                    max_backoff_s=max_backoff_s,
                )
        except RuntimeError as exc:
            print(f"[RETRY] {label} | {platform} | attempt={attempts} | reason={exc}", flush=True)
            if attempts >= max_attempts:
                raise RuntimeError(f"{provider} request failed for {label}/{platform}: {exc}") from exc
            time.sleep(min(2.0 * attempts, 10.0))
            continue

        batch = _extract_json_array(text)
        if not batch:
            if attempts >= max_attempts:
                raise RuntimeError(f"Gemini produced invalid JSON for {label}/{platform}.")
            continue

        for item in batch:
            if len(rows) >= target_count:
                break
            if not isinstance(item, dict):
                continue
            sample_label = str(item.get("label", "")).strip()
            if sample_label not in LABELS or sample_label != label:
                continue
            clean = _normalize_text(item.get("text", ""))
            if len(clean) < 12:
                continue
            key = clean.casefold()
            if key in seen:
                continue
            seen.add(key)
            rows.append({"text": clean, "label": sample_label, "platform": platform})
            print(f"[PROGRESS] {label} | {platform} -> {len(rows)}/{target_count}", flush=True)

    if len(rows) != target_count:
        raise RuntimeError(
            f"Could not satisfy bucket {label}/{platform}: generated {len(rows)} of {target_count} unique rows."
        )
    return rows


def _validate_global(rows: List[Dict[str, str]]) -> None:
    if len(rows) != 300:
        raise RuntimeError(f"Expected exactly 300 rows, got {len(rows)}.")

    counts = defaultdict(int)
    seen = set()
    for r in rows:
        label = str(r.get("label", "")).strip()
        text = _normalize_text(r.get("text", ""))
        if label not in LABELS:
            raise RuntimeError(f"Invalid label found: {label}")
        if not text:
            raise RuntimeError("Empty text found after cleaning.")
        key = text.casefold()
        if key in seen:
            raise RuntimeError("Duplicate text found after global validation.")
        seen.add(key)
        counts[label] += 1

    for label in LABELS:
        if counts[label] != TOTAL_PER_LABEL:
            raise RuntimeError(f"Class count mismatch for {label}: expected 75, got {counts[label]}.")


def _infer_bias_label(text: str) -> str:
    t = text.lower()
    if re.search(r"\b(women|men|female|male|gender)\b", t):
        return "gender_bias"
    if re.search(r"\b(immigrant|foreign|nationality|visa|local candidates?)\b", t):
        return "nationality_bias"
    return "profession_bias"


def _audit_and_relabel(rows: List[Dict[str, str]]) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    anti_stereo = re.compile(r"\b(not\s+inferior|not\s+stealing\s+jobs|myth\s+that|stereotype\s+that|creating\s+jobs)\b", re.I)
    supportive = re.compile(r"\b(encouraging|inclusive|support|welcome|equity|fair\s+hiring|diverse\s+teams)\b", re.I)
    exclusion = re.compile(r"\b(should\s+not\s+hire|don't\s+trust|prefer\s+\w+\s+over|not\s+a\s+fit|keep\s+out)\b", re.I)

    audited = []
    flags: List[Dict[str, str]] = []
    for row in rows:
        text = str(row.get("text", ""))
        label = str(row.get("label", "")).strip()
        new_label = label
        if label != "neutral" and (anti_stereo.search(text) or supportive.search(text)):
            new_label = "neutral"
            flags.append({"reason": "supportive_or_antistereotype_mislabeled_bias", "text": text, "from": label, "to": new_label})
        elif label == "neutral" and exclusion.search(text):
            new_label = _infer_bias_label(text)
            flags.append({"reason": "harmful_exclusion_mislabeled_neutral", "text": text, "from": label, "to": new_label})
        audited.append({"text": text, "label": new_label, "platform": row.get("platform", "")})

    # Rebalance to strict 75/class by deterministic reassignment from overrepresented labels.
    counts = defaultdict(int)
    for r in audited:
        counts[r["label"]] += 1
    deficits = [lab for lab in LABELS for _ in range(max(0, TOTAL_PER_LABEL - counts[lab]))]
    if deficits:
        for r in audited:
            if not deficits:
                break
            if counts[r["label"]] > TOTAL_PER_LABEL:
                replacement = deficits.pop(0)
                counts[r["label"]] -= 1
                counts[replacement] += 1
                flags.append({"reason": "rebalance", "text": r["text"], "from": r["label"], "to": replacement})
                r["label"] = replacement

    return audited, flags


def _write_csv(rows: List[Dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        for r in rows:
            writer.writerow({"text": r["text"], "label": r["label"]})


def generate_dataset(
    output_csv: Path,
    model: str,
    *,
    rpm: int,
    max_retries: int,
    base_backoff_s: float,
    max_backoff_s: float,
    provider: str,
) -> Path:
    if provider == "gemini":
        api_key = _require_api_key()
        selected_model = model
    else:
        api_key = _require_openrouter_key()
        if model == "auto":
            raise RuntimeError(
                "OpenRouter auto model selection is disabled. Pass a fixed model explicitly, e.g. --model openai/gpt-4.1-mini"
            )
        selected_model = model
        print(f"[CONFIG] provider=openrouter | model={selected_model}", flush=True)
    all_rows: List[Dict[str, str]] = []
    pacer = RequestPacer(rpm=rpm)

    for label in LABELS:
        for platform in PLATFORMS:
            n = PER_PLATFORM_BY_LABEL[label][platform]
            bucket = _generate_bucket(
                api_key=api_key,
                model=selected_model,
                label=label,
                platform=platform,
                target_count=n,
                pacer=pacer,
                max_retries=max_retries,
                base_backoff_s=base_backoff_s,
                max_backoff_s=max_backoff_s,
                provider=provider,
            )
            all_rows.extend(bucket)

    # deterministic shuffle keeps reproducibility while mixing classes/platforms
    all_rows = sorted(all_rows, key=lambda r: (r["label"], r["platform"], r["text"].casefold()))

    audited_rows, flags = _audit_and_relabel(all_rows)
    _validate_global(audited_rows)
    _write_csv(audited_rows, output_csv)
    if flags:
        audit_path = output_csv.with_name(output_csv.stem + "_audit_flags.json")
        audit_path.write_text(json.dumps(flags, indent=2), encoding="utf-8")
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Gemini-based meta-classifier training data (300 rows).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT, help="Output CSV path.")
    parser.add_argument(
        "--provider",
        choices=("gemini", "openrouter"),
        default="gemini",
        help="LLM provider for generation.",
    )
    parser.add_argument(
        "--model",
        default="gemini-1.5-pro",
        help="Model id. For OpenRouter, pass --model auto to select best available.",
    )
    parser.add_argument("--rpm", type=int, default=12, help="Max Gemini requests per minute.")
    parser.add_argument("--max-retries", type=int, default=6, help="Retries per request on 429/5xx/network issues.")
    parser.add_argument("--base-backoff", type=float, default=2.0, help="Initial exponential backoff seconds.")
    parser.add_argument("--max-backoff", type=float, default=45.0, help="Maximum backoff seconds.")
    args = parser.parse_args()

    out = generate_dataset(
        output_csv=args.output,
        model=args.model,
        rpm=args.rpm,
        max_retries=args.max_retries,
        base_backoff_s=args.base_backoff,
        max_backoff_s=args.max_backoff,
        provider=args.provider,
    )
    print(f"Wrote 300 rows to {out}")


if __name__ == "__main__":
    main()
