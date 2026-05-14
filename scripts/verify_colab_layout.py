#!/usr/bin/env python3
"""Verify required files exist for StereoSet+CrowS Colab training. Run from repo root."""

from __future__ import annotations

import sys
from pathlib import Path

# Repo root = parent of scripts/
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.colab_stereoset_pipeline import run_verify_files  # noqa: E402


def main() -> int:
    return run_verify_files(_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
