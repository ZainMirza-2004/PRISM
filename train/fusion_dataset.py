"""Build class-balanced 4-way training CSV from binary hate corpus + synthetic bias data."""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.label_config import LABELS

LOGGER = logging.getLogger(__name__)

# Heuristic routing: hate (binary 1) → one of three bias types (not neutral).
_GENDER_PAT = re.compile(
    r"\b(?:woman|women|man|men|girl|boy|she|her|he|him|his|hers|gay|lesbian|bitch|twat|vagina|"
    r"cunt|slut|whore|feminazi|dyke|faggot|fag|gender|female|male|mother|father|wife|husband|"
    r"feminism|misogyn|misandr|boobs|bra|dress|pretty|ugly\s+woman|guys?\s+are|girls?\s+are)\b",
    re.I,
)
_NAT_PAT = re.compile(
    r"\b(?:immigrant|foreign|foreigner|latino|latina|mexican|muslim|islam|china|chink|ching|"
    r"deport|country|shithole|accent|somalia|irish|jew|jewish|white\s+people|black\s+people|"
    r"asian|ethnic|race|racial|national|border|visa|refugee|minority|spic|afro|mongol)\b",
    re.I,
)
_PROF_PAT = re.compile(
    r"\b(?:retard|mongoloid|idiot|moron|engineer|worker|workers|job|jobs|workplace|manager|"
    r"hire|skill|skills|stupid|dumb|iq|smart|lazy|customer\s+service|professional|career|"
    r"office|blue[\s-]?collar|white[\s-]?collar|tech|coder|developer)\b",
    re.I,
)


def load_real_binary_csv(path: str | Path) -> pd.DataFrame:
    """Expect binary labels 0 / 1 and a text column (Content or text)."""
    p = Path(path)
    df = pd.read_csv(p, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    text_col = "content" if "content" in df.columns else "text"
    if text_col not in df.columns:
        raise ValueError(f"Need Content or text column in {path}, got {list(df.columns)}")
    lab_col = "label" if "label" in df.columns else None
    if lab_col is None:
        raise ValueError(f"Need label column in {path}")
    out = pd.DataFrame({"text": df[text_col].astype(str), "binary_label": pd.to_numeric(df[lab_col], errors="coerce")})
    out = out.dropna(subset=["text", "binary_label"])
    out["binary_label"] = out["binary_label"].astype(int)
    out = out[out["binary_label"].isin([0, 1])].reset_index(drop=True)
    return out


def map_binary_to_four_way(real_df: pd.DataFrame, *, seed: int = 42) -> pd.DataFrame:
    """label 0 → neutral; hate → gender/nationality/profession via regex scores + tie-break rotation."""
    texts_s = real_df["text"].astype(str)
    gc = texts_s.str.count(_GENDER_PAT.pattern).values.astype(np.int64)
    nc = texts_s.str.count(_NAT_PAT.pattern).values.astype(np.int64)
    pc = texts_s.str.count(_PROF_PAT.pattern).values.astype(np.int64)
    mx = np.maximum.reduce([gc, nc, pc])
    hate = real_df["binary_label"].values.astype(np.int64) == 1

    labels = np.empty(len(real_df), dtype=object)
    labels[~hate] = "neutral"
    rr = 0
    for i in np.flatnonzero(hate):
        if mx[i] == 0:
            labels[i] = ["gender_bias", "nationality_bias", "profession_bias"][rr % 3]
            rr += 1
            continue
        g, n, p = int(gc[i]), int(nc[i]), int(pc[i])
        if g >= n and g >= p:
            labels[i] = "gender_bias"
        elif n >= g and n >= p:
            labels[i] = "nationality_bias"
        else:
            labels[i] = "profession_bias"

    mapped = pd.DataFrame({"text": texts_s.values, "label": labels})
    LOGGER.info("Mapped real binary → 4-way:\n%s", mapped["label"].value_counts())
    return mapped


def load_synthetic_four_way(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Synthetic CSV needs text,label; got {list(df.columns)}")
    syn = pd.DataFrame({"text": df["text"].astype(str), "label": df["label"].astype(str)})
    syn = syn[syn["label"].isin(LABELS)].reset_index(drop=True)
    return syn


def fuse_real_synthetic_8020(
    real_four: pd.DataFrame,
    syn_four: pd.DataFrame,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Concatenate real + synthetic so synthetic ≈ 20% of rows (oversample synthetic with replacement).
    Then subsample each class to the same count (balanced 4-way).
    """
    rng = np.random.RandomState(seed)
    real = real_four.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    r = len(real)
    syn_target = max(1, int(round(r * 0.25)))  # S/(R+S)=0.2 → S = R/4
    syn_pool = syn_four.reset_index(drop=True)
    if len(syn_pool) == 0:
        raise ValueError("Synthetic dataset is empty.")
    idx = rng.randint(0, len(syn_pool), size=syn_target)
    syn_sample = syn_pool.iloc[idx].reset_index(drop=True)

    fused = pd.concat([real, syn_sample], axis=0, ignore_index=True)
    fused = fused.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    LOGGER.info(
        "Fusion mix: real_rows=%s synthetic_rows=%s (synthetic_frac=%.4f)",
        r,
        len(syn_sample),
        len(syn_sample) / max(1, len(fused)),
    )

    counts = fused["label"].value_counts()
    target = int(counts.min())
    if target < 1:
        raise ValueError("Fusion produced empty class after concat.")
    balanced_parts: list[pd.DataFrame] = []
    for lab in LABELS:
        sub = fused[fused["label"] == lab]
        if len(sub) >= target:
            balanced_parts.append(sub.sample(n=target, random_state=seed))
        else:
            balanced_parts.append(sub.sample(n=target, replace=True, random_state=seed))
    out = pd.concat(balanced_parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    LOGGER.info("Balanced fusion (per-class n=%s):\n%s", target, out["label"].value_counts())
    return out


def build_fusion_csv(
    real_path: str | Path,
    synthetic_path: str | Path,
    output_csv: str | Path,
    *,
    seed: int = 42,
    max_real_rows: int | None = None,
) -> Path:
    LOGGER.info("Loading real binary data from %s", real_path)
    real_bin = load_real_binary_csv(real_path)
    LOGGER.info("Real rows (binary): %s", len(real_bin))
    if max_real_rows is not None and len(real_bin) > max_real_rows:
        real_bin = real_bin.sample(n=max_real_rows, random_state=seed).reset_index(drop=True)
        LOGGER.info("Subsampled real rows to %s", len(real_bin))
    real_four = map_binary_to_four_way(real_bin, seed=seed)
    syn_four = load_synthetic_four_way(synthetic_path)
    LOGGER.info("Synthetic rows: %s", len(syn_four))
    fused = fuse_real_synthetic_8020(real_four, syn_four, seed=seed)
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fused.to_csv(out_path, index=False)
    LOGGER.info("Wrote balanced fusion CSV: %s (%s rows)", out_path, len(fused))
    return out_path


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build fused 4-class balanced CSV for DistilBERT training")
    ap.add_argument("--real", default="train_data", help="Binary CSV (Content,text + Label,label)")
    ap.add_argument("--synthetic", default="data/training/generated_social_bias_data.csv")
    ap.add_argument("--output", default="data/training/fused_real_synthetic_balanced.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--max-real",
        type=int,
        default=None,
        help="Optional cap on real rows before 80/20 fusion (stratified per-class pre-cap).",
    )
    return ap.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    build_fusion_csv(
        args.real,
        args.synthetic,
        args.output,
        seed=args.seed,
        max_real_rows=args.max_real,
    )


if __name__ == "__main__":
    main()
