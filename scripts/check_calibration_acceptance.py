#!/usr/bin/env python3
"""Verify sanity FPR/FNR after calibration; non-zero exit when constraints fail."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
MAX_FPR = 0.20
MAX_FNR = 0.20


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sanity-metrics",
        default=str(_ROOT / "data/output/sanity_eval_calibrated/sanity_metrics.json"),
        help="sanity_metrics.json from evaluate-sanity --calibrated --use-thresholds",
    )
    args = ap.parse_args()
    p = Path(args.sanity_metrics)
    if not p.is_file():
        print(f"Missing metrics file: {p}", file=sys.stderr)
        return 2
    data = json.loads(p.read_text(encoding="utf-8"))
    if "post_calibration_thresholded" in data:
        blk = data["post_calibration_thresholded"]
    elif "fpr_neutral_policy_positive_group" in data:
        blk = data
    else:
        print("Unrecognized metrics schema", file=sys.stderr)
        return 3
    fpr = float(blk.get("fpr_neutral_policy_positive_group", 999))
    fnr = float(blk.get("fnr_explicit_or_implicit_bias", 999))
    ok = fpr <= MAX_FPR + 1e-9 and fnr <= MAX_FNR + 1e-9
    print(json.dumps({"sanity_fpr": fpr, "sanity_fnr": fnr, "pass": ok}, indent=2))
    if not ok:
        cm = blk.get("overall", {}).get("confusion_matrix", [])
        print("Confusion matrix:", file=sys.stderr)
        print(json.dumps(cm), file=sys.stderr)
        thr_file = _ROOT / "models" / "meta_fusion" / "optimal_thresholds.json"
        if thr_file.is_file():
            thr_data = json.loads(thr_file.read_text(encoding="utf-8"))
            print("Per-class thresholds:", file=sys.stderr)
            print(json.dumps(thr_data.get("thresholds", thr_data), indent=2), file=sys.stderr)
            meta = thr_data.get("meta")
            if meta:
                print("Threshold optimisation meta:", file=sys.stderr)
                print(json.dumps(meta, indent=2, default=str), file=sys.stderr)
        else:
            print(f"(No {thr_file} for threshold diagnostics)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
