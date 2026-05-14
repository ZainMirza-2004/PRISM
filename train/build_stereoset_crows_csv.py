"""Build a unified text,label CSV from StereoSet (HF) + CrowS-Pairs (official CSV).

StereoSet: https://huggingface.co/datasets/McGill-NLP/stereoset
CrowS-Pairs: https://github.com/nyu-mll/crows-pairs (anonymized CSV; no HF script loader in recent `datasets`).

Labels match ``models.label_config.LABELS`` (4-way). See ``data/training/STEREOSET_CROWS_LABEL_MAP.md``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from models.label_config import LABELS

LOGGER = logging.getLogger(__name__)

CROWS_PAIRS_CSV_URL = (
    "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv"
)

# StereoSet domain → project label (race/religion → nationality_bias bucket; document in thesis)
STEREOTYPE_DOMAIN_TO_LABEL = {
    "gender": "gender_bias",
    "profession": "profession_bias",
    "race": "nationality_bias",
    "religion": "nationality_bias",
}

# CrowS-Pairs bias_type column (CSV strings). Socioeconomic → profession_bias (class/occupation-adjacent stereotypes).
CROWS_TYPE_TO_LABEL = {
    "gender": "gender_bias",
    "nationality": "nationality_bias",
    "race-color": "nationality_bias",
    "religion": "nationality_bias",
    "sexual-orientation": "nationality_bias",
    "age": "nationality_bias",
    "disability": "nationality_bias",
    "physical-appearance": "nationality_bias",
    "socioeconomic": "profession_bias",
}

# StereoSet gold_label int (ClassLabel order: anti-stereotype, stereotype, unrelated)
SS_STEREOTYPE_INT = 1


def _stereoset_bias_to_label(bias_type: str) -> str:
    key = bias_type.strip().lower()
    if key not in STEREOTYPE_DOMAIN_TO_LABEL:
        raise ValueError(f"Unknown StereoSet bias_type: {bias_type!r} (expected one of {sorted(STEREOTYPE_DOMAIN_TO_LABEL)})")
    return STEREOTYPE_DOMAIN_TO_LABEL[key]


def _combine_context_sentence(context: str, sentence: str) -> str:
    c, s = context.strip(), sentence.strip()
    if not c:
        return s
    if not s:
        return c
    return f"{c} {s}"


def rows_from_stereoset() -> list[dict]:
    out: list[dict] = []
    for config in ("intrasentence", "intersentence"):
        ds = load_dataset("McGill-NLP/stereoset", config, split="validation")
        for ex in ds:
            group_id = f"stereoset-{config}-{ex['id']}"
            bias_type = str(ex["bias_type"])
            label_for_stereo = _stereoset_bias_to_label(bias_type)
            sents = ex["sentences"]
            if not isinstance(sents, dict):
                raise TypeError("Unexpected StereoSet `sentences` format (expected dict of lists). Upgrade/downgrade `datasets` or file an issue.")
            texts = sents["sentence"]
            golds = sents["gold_label"]
            if len(texts) != len(golds):
                raise ValueError(f"StereoSet id={ex['id']}: sentence/gold_label length mismatch")
            context = str(ex.get("context") or "")
            for sent, gl in zip(texts, golds, strict=True):
                gl = int(gl)
                label = label_for_stereo if gl == SS_STEREOTYPE_INT else "neutral"
                text = _combine_context_sentence(context, str(sent))
                if not text:
                    continue
                out.append(
                    {
                        "text": text,
                        "label": label,
                        "group_id": group_id,
                        "source": f"stereoset_{config}",
                    }
                )
    return out


def _load_crows_pairs_df() -> pd.DataFrame:
    df = pd.read_csv(CROWS_PAIRS_CSV_URL)
    # First column in the public CSV is an unnamed index
    first = df.columns[0]
    if first.startswith("Unnamed") or first == "":
        df = df.drop(columns=first)
    return df


def rows_from_crows_pairs(*, stereo_only: bool = True) -> list[dict]:
    """CrowS-Pairs: ``sent_more`` → bias label, ``sent_less`` → neutral.

    Training uses **stereo** direction only (``stereo_antistereo == stereo``). **antistereo** pairs are
    dropped to reduce label noise (see ``STEREOSET_CROWS_LABEL_MAP.md``).
    """
    df = _load_crows_pairs_df()
    required = {"sent_more", "sent_less", "bias_type", "stereo_antistereo"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CrowS-Pairs CSV missing columns {missing}")
    n_skip = 0
    out: list[dict] = []
    for idx, row in df.iterrows():
        direction = str(row["stereo_antistereo"]).strip().lower()
        if stereo_only and direction != "stereo":
            n_skip += 1
            continue
        group_id = f"crows_pairs-{int(idx)}"
        btype = str(row["bias_type"]).strip().lower()
        if btype not in CROWS_TYPE_TO_LABEL:
            LOGGER.warning("Skipping CrowS-Pairs row %s: unknown bias_type %r", idx, row["bias_type"])
            continue
        lab = CROWS_TYPE_TO_LABEL[btype]
        sm = str(row["sent_more"]).strip()
        sl = str(row["sent_less"]).strip()
        if sm:
            out.append({"text": sm, "label": lab, "group_id": group_id, "source": "crows_pairs"})
        if sl:
            out.append({"text": sl, "label": "neutral", "group_id": group_id, "source": "crows_pairs"})
    if stereo_only:
        LOGGER.info("CrowS-Pairs: kept stereo rows; dropped %s antistereo (or other) rows.", n_skip)
    return out


def rows_from_crows_pairs_stereo_only() -> list[dict]:
    """Alias for external eval / scripts that want the same training filter."""
    return rows_from_crows_pairs(stereo_only=True)


def _ensure_all_labels(df: pd.DataFrame) -> None:
    present = set(df["label"].unique())
    need = set(LABELS)
    if present < need:
        raise ValueError(f"After merge, not all labels present. Have {sorted(present)}, need {LABELS}")


def _balance_per_label(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Undersample each class to min count so all four labels are equally represented."""
    counts = df["label"].value_counts()
    k = int(counts.min())
    if k < 50:
        LOGGER.warning("Very small per-class count after balancing (k=%s). Consider --no-balance for more data.", k)
    parts = []
    for lab in LABELS:
        sub = df[df["label"] == lab]
        parts.append(sub.sample(n=min(len(sub), k), random_state=seed))
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _cap_neutral(df: pd.DataFrame, max_ratio_vs_median_bias: float, seed: int) -> pd.DataFrame:
    """Cap neutral count to max_ratio_vs_median_bias × median count of non-neutral labels."""
    bias_mask = df["label"] != "neutral"
    bias_counts = df.loc[bias_mask, "label"].value_counts()
    if bias_counts.empty:
        return df
    med = float(bias_counts.median())
    cap = int(med * max_ratio_vs_median_bias)
    neu = df[~bias_mask]
    rest = df[bias_mask]
    if len(neu) > cap:
        neu = neu.sample(n=cap, random_state=seed)
    return pd.concat([rest, neu], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _mix_in_rows(
    df: pd.DataFrame,
    mix_csv: str | Path,
    mix_fraction: float,
    seed: int,
    *,
    source_tag: str = "mixed_noisy",
) -> pd.DataFrame:
    """Append ~``mix_fraction`` × len(df) rows from a second CSV (columns text, label).

    ``mix_fraction`` is relative to the benchmark dataframe size *before* mixing: we add
    ``round(len(df) * mix_fraction)`` rows so the extra mass scales with corpus size (typ. 5–10%).
    """
    if mix_fraction <= 0:
        return df
    path = Path(mix_csv)
    if not path.is_file():
        raise FileNotFoundError(f"--mix-csv not found: {path}")
    mix = pd.read_csv(path)
    mix.columns = [str(c).strip().lower() for c in mix.columns]
    if "text" not in mix.columns or "label" not in mix.columns:
        raise ValueError(f"{path} must have columns: text, label")
    mix = mix.dropna(subset=["text", "label"]).copy()
    mix["label"] = mix["label"].astype(str).str.strip()
    mix = mix[mix["label"].isin(LABELS)].reset_index(drop=True)
    if len(mix) == 0:
        raise ValueError(f"No valid labeled rows in {path}")
    n_add = max(0, int(round(len(df) * float(mix_fraction))))
    if n_add == 0:
        return df
    take = mix.sample(n=n_add, replace=len(mix) < n_add, random_state=seed)
    extra = pd.DataFrame(
        {
            "text": take["text"].astype(str).str.strip().values,
            "label": take["label"].values,
            "group_id": [f"{source_tag}-{seed}-{i}" for i in range(len(take))],
            "source": [source_tag] * len(take),
        }
    )
    out = pd.concat([df, extra], ignore_index=True)
    LOGGER.info("Mixed in %s rows from %s (mix_fraction=%s on base n=%s).", len(extra), path, mix_fraction, len(df))
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def build_combined_csv(
    output_path: str | Path,
    *,
    balance: bool = False,
    neutral_cap_ratio: float | None = 2.5,
    seed: int = 42,
    mix_csv: str | Path | None = None,
    mix_fraction: float = 0.0,
    mix_source_tag: str = "mixed_noisy",
) -> Path:
    LOGGER.info("Loading StereoSet (intrasentence + intersentence)…")
    ss_rows = rows_from_stereoset()
    LOGGER.info("StereoSet rows: %s", len(ss_rows))
    LOGGER.info("Loading CrowS-Pairs from %s", CROWS_PAIRS_CSV_URL)
    cp_rows = rows_from_crows_pairs(stereo_only=True)
    LOGGER.info("CrowS-Pairs rows (stereo-only): %s", len(cp_rows))

    df = pd.DataFrame(ss_rows + cp_rows)
    df = df.dropna(subset=["text", "label", "group_id"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    _ensure_all_labels(df)

    if mix_fraction > 0:
        if not mix_csv:
            raise ValueError("mix_csv is required when mix_fraction > 0")
        df = _mix_in_rows(df, mix_csv, mix_fraction, seed, source_tag=mix_source_tag)
        _ensure_all_labels(df)

    if balance:
        LOGGER.info("Balancing classes (equal undersampling to min class count).")
        df = _balance_per_label(df, seed)
    elif neutral_cap_ratio is not None and neutral_cap_ratio > 0:
        LOGGER.info("Capping neutral to %.2f × median bias-class count.", neutral_cap_ratio)
        df = _cap_neutral(df, neutral_cap_ratio, seed)

    _ensure_all_labels(df)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    LOGGER.info("Wrote %s rows to %s", len(df), out_path)
    LOGGER.info("Label counts:\n%s", df["label"].value_counts())
    return out_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Build StereoSet + CrowS-Pairs CSV for DistilBERT training.")
    p.add_argument(
        "--output",
        default="data/training/stereoset_crows_combined.csv",
        help="Output CSV path (columns: text,label,group_id,source)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--balance",
        action="store_true",
        help="Force equal class counts (undersample to minimum class size). Smaller but balanced.",
    )
    p.add_argument(
        "--no-neutral-cap",
        action="store_true",
        help="Do not undersample neutral (only applies when --balance is off).",
    )
    p.add_argument(
        "--neutral-cap-ratio",
        type=float,
        default=2.5,
        help="When not using --balance, cap neutral to this × median bias-class count (default 2.5).",
    )
    p.add_argument(
        "--mix-csv",
        default=None,
        help="Optional CSV (text,label) e.g. generated_social_bias_data.csv — append ~mix-fraction × base rows.",
    )
    p.add_argument(
        "--mix-fraction",
        type=float,
        default=0.0,
        help="Extra rows ≈ this × benchmark row count before mix (e.g. 0.075 for ~7.5%% added).",
    )
    p.add_argument(
        "--mix-source-tag",
        default="mixed_noisy",
        help="Value for `source` column on mixed-in rows.",
    )
    args = p.parse_args()
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


if __name__ == "__main__":
    main()
