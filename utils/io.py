from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict

import pandas as pd


def save_results(results: List[Dict], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".json":
        with path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    elif path.suffix.lower() == ".csv":
        pd.DataFrame(results).to_csv(path, index=False)
    else:
        raise ValueError("Output format must be .json or .csv")

    return path
