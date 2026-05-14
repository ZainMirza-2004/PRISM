"""Print stratified random samples per label from a benchmark training CSV (sanity check)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from models.label_config import LABELS

LOGGER = logging.getLogger(__name__)


def sample_benchmark_csv(
    csv_path: str | Path,
    per_label: int = 20,
    seed: int = 42,
) -> dict[str, list[tuple[str, str, str]]]:
    """Return label -> list of (text, source, group_id) for display."""
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV needs columns: text, label")
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(str).str.strip()
    out: dict[str, list[tuple[str, str, str]]] = {}
    for lab in LABELS:
        sub = df[df["label"] == lab]
        n = min(per_label, len(sub))
        if n == 0:
            out[lab] = []
            continue
        take = sub.sample(n=n, random_state=seed)
        rows = []
        for _, r in take.iterrows():
            src = str(r.get("source", "")) if "source" in take.columns else ""
            gid = str(r.get("group_id", "")) if "group_id" in take.columns else ""
            txt = str(r["text"])
            if len(txt) > 500:
                txt = txt[:500] + "…"
            rows.append((txt, src, gid))
        out[lab] = rows
    return out


def format_samples_md(samples: dict[str, list[tuple[str, str, str]]], title: str) -> str:
    lines = [f"# {title}", ""]
    for lab in LABELS:
        lines.append(f"## `{lab}` ({len(samples[lab])} samples)")
        lines.append("")
        for i, (txt, src, gid) in enumerate(samples[lab], 1):
            meta = []
            if src:
                meta.append(f"source={src}")
            if gid:
                meta.append(f"group_id={gid}")
            suffix = f" ({', '.join(meta)})" if meta else ""
            lines.append(f"{i}. {txt}{suffix}")
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Inspect stereoset_crows_*.csv samples per label.")
    p.add_argument("--data", required=True, help="Path to CSV (text,label[,group_id,source])")
    p.add_argument("--per-label", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None, help="Optional path to write Markdown (e.g. for thesis appendix).")
    args = p.parse_args()

    df_full = pd.read_csv(args.data)
    df_full.columns = [str(c).strip().lower() for c in df_full.columns]
    counts = df_full["label"].astype(str).str.strip().value_counts().reindex(LABELS, fill_value=0)
    LOGGER.info("Row counts per label in file:\n%s", counts.to_string())

    samples = sample_benchmark_csv(args.data, per_label=args.per_label, seed=args.seed)
    for lab in LABELS:
        LOGGER.info("%s: showing %s samples (cap %s)", lab, len(samples[lab]), args.per_label)
        for i, (txt, src, gid) in enumerate(samples[lab], 1):
            meta = f" [{src}]" if src else ""
            print(f"\n--- {lab} #{i}{meta} ---\n{txt}\n")

    if args.output:
        md = format_samples_md(samples, title=f"Benchmark CSV samples: {args.data}")
        Path(args.output).write_text(md, encoding="utf-8")
        LOGGER.info("Wrote Markdown: %s", args.output)


if __name__ == "__main__":
    main()
