#!/usr/bin/env python3
"""
Colab fallback when ``main.py`` on Google Drive is outdated and lacks
``build-benchmark-csv`` / ``inspect-benchmark-csv`` / ``evaluate-crows-stereo``.

Run from repo root (same as ``main.py``):

    python scripts/colab_stereoset_pipeline.py verify-files
    python scripts/colab_stereoset_pipeline.py diagnose
    python scripts/colab_stereoset_pipeline.py build --output data/training/stereoset_crows_combined.csv
    python scripts/colab_stereoset_pipeline.py build --output data/training/stereoset_crows_balanced.csv --balance
    python scripts/colab_stereoset_pipeline.py inspect --data data/training/stereoset_crows_combined.csv
    python scripts/colab_stereoset_pipeline.py train --data ... --output ... --epochs 5 --batch-size 16 --seed 42
    python scripts/colab_stereoset_pipeline.py eval-crows --model models/distilbert_B_balanced
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Minimal set for StereoSet+CrowS training (Colab). ``verify_colab_layout.py`` delegates here.
REQUIRED_SUBPATHS = [
    "main.py",
    "requirements.txt",
    "models/label_config.py",
    "train/train_model.py",
    "train/build_stereoset_crows_csv.py",
    "train/inspect_benchmark_csv.py",
    "train/eval_type_head_on_crows_stereo.py",
    "data/training/STEREOSET_CROWS_LABEL_MAP.md",
    "scripts/colab_stereoset_pipeline.py",
]


def run_verify_files(root: Path | None = None) -> int:
    """Return 0 if OK, 1 if not repo root, 2 if files missing."""
    base = (root or REPO_ROOT).resolve()
    if not (base / "main.py").is_file():
        print("ERROR: not repo root — main.py not found:", base / "main.py", file=sys.stderr)
        print("  cwd:", Path.cwd(), file=sys.stderr)
        return 1
    bad = False
    for rel in REQUIRED_SUBPATHS:
        p = base / rel
        ok = p.is_file()
        print(("OK " if ok else "MISSING"), p)
        if not ok:
            bad = True
    if bad:
        print("\nUpload the missing paths (or clone full repo). See docs/COLAB_GEMINI_RUNBOOK.md", file=sys.stderr)
        return 2
    mp = base / "main.py"
    txt = mp.read_text(encoding="utf-8", errors="replace")
    for needle in ("build-benchmark-csv", "inspect-benchmark-csv", "evaluate-crows-stereo"):
        if needle not in txt:
            print(f"NOTE: main.py missing {needle!r} — use this script for build/inspect/train/eval-crows (Drive-safe).")
            break
    else:
        print("\nmain.py includes benchmark subcommands (optional; pipeline script still works if False).")
    print("\nRepo root:", base)
    return 0


def _chdir_root() -> Path:
    os.chdir(REPO_ROOT)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return REPO_ROOT


def cmd_verify_files(_: argparse.Namespace) -> None:
    code = run_verify_files()
    if code != 0:
        raise SystemExit(code)


def cmd_diagnose(_: argparse.Namespace) -> None:
    root = _chdir_root()
    mp = root / "main.py"
    print("cwd:", os.getcwd())
    print("main.py path:", mp.resolve())
    print("main.py exists:", mp.is_file())
    if mp.is_file():
        txt = mp.read_text(encoding="utf-8", errors="replace")
        for needle in ("build-benchmark-csv", "inspect-benchmark-csv", "evaluate-crows-stereo"):
            print(f"  main.py contains {needle!r}:", needle in txt)
    print("\nIf build-benchmark-csv is False, Drive has an OLD main.py — use this script for build/inspect/train/eval, or replace main.py.")


def cmd_build(args: argparse.Namespace) -> None:
    _chdir_root()
    from train.build_stereoset_crows_csv import build_combined_csv

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cap = None if args.no_neutral_cap else args.neutral_cap_ratio
    build_combined_csv(
        args.output,
        balance=args.balance,
        neutral_cap_ratio=cap,
        seed=args.seed,
        mix_csv=args.mix_csv,
        mix_fraction=args.mix_fraction,
        mix_source_tag=args.mix_source_tag,
    )
    print("Wrote:", args.output)


def cmd_inspect(args: argparse.Namespace) -> None:
    _chdir_root()
    from train.inspect_benchmark_csv import format_samples_md, sample_benchmark_csv

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    samples = sample_benchmark_csv(args.data, per_label=args.per_label, seed=args.seed)
    from models.label_config import LABELS

    for lab in LABELS:
        for i, (txt, src, gid) in enumerate(samples[lab], 1):
            meta = f" [{src}]" if src else ""
            print(f"\n--- {lab} #{i}{meta} ---\n{txt}\n")
    if args.output:
        md = format_samples_md(samples, title=f"Samples: {args.data}")
        Path(args.output).write_text(md, encoding="utf-8")
        print("Wrote:", args.output)


def cmd_train(args: argparse.Namespace) -> None:
    _chdir_root()
    from utils.logging_config import configure_logging

    configure_logging()
    from train.train_model import train

    train(
        data_path=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print("Training finished ->", args.output)


def cmd_eval_crows(args: argparse.Namespace) -> None:
    _chdir_root()
    from train.eval_type_head_on_crows_stereo import evaluate_type_head_crows

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rep = evaluate_type_head_crows(args.model, batch_size=args.batch_size)
    print(rep.get("classification_report_text", ""))
    print(json.dumps({"macro_f1": rep["macro_f1"], "n_examples": rep["n_examples"]}, indent=2))
    if args.json_out:
        p = Path(args.json_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        serializable = {k: v for k, v in rep.items() if k != "classification_report_text"}
        p.write_text(json.dumps(serializable, indent=2, default=str), encoding="utf-8")
        print("Wrote:", args.json_out)


def main() -> None:
    p = argparse.ArgumentParser(description="Colab StereoSet+CrowS pipeline (bypasses outdated main.py).")
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser(
        "verify-files",
        help="Check required paths exist (replaces scripts/verify_colab_layout.py on Drive).",
    )
    v.set_defaults(func=cmd_verify_files)

    d = sub.add_parser("diagnose", help="Print which main.py is used and whether it is new enough.")
    d.set_defaults(func=cmd_diagnose)

    b = sub.add_parser("build", help="Build stereoset_crows CSV (same as build-benchmark-csv).")
    b.add_argument("--output", required=True)
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--balance", action="store_true")
    b.add_argument("--no-neutral-cap", action="store_true")
    b.add_argument("--neutral-cap-ratio", type=float, default=2.5)
    b.add_argument("--mix-csv", default=None)
    b.add_argument("--mix-fraction", type=float, default=0.0)
    b.add_argument("--mix-source-tag", default="mixed_noisy")
    b.set_defaults(func=cmd_build)

    i = sub.add_parser("inspect", help="Sample rows per label (same as inspect-benchmark-csv).")
    i.add_argument("--data", required=True)
    i.add_argument("--per-label", type=int, default=20)
    i.add_argument("--seed", type=int, default=42)
    i.add_argument("--output", default=None)
    i.set_defaults(func=cmd_inspect)

    t = sub.add_parser("train", help="Train type head (same as main.py train for CSV path).")
    t.add_argument("--data", required=True)
    t.add_argument("--output", required=True)
    t.add_argument("--epochs", type=int, default=5)
    t.add_argument("--batch-size", type=int, default=16)
    t.add_argument("--seed", type=int, default=42)
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("eval-crows", help="CrowS stereo external eval (same as evaluate-crows-stereo).")
    e.add_argument("--model", required=True)
    e.add_argument("--batch-size", type=int, default=16)
    e.add_argument("--json-out", default=None)
    e.set_defaults(func=cmd_eval_crows)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
