"""Merge labeled gold column into texts from archived JSON — one-shot for manual_eval_200."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
ARCH = _ROOT / "data" / "evaluation" / "manual_eval_200_generated.json"

# Annotator labels (post_id → label); must match annotated export.
LABELS_FROM_ANNOTATOR: dict[str, str] = {}

def _populate_labels():
    txt = LABELS_LINES.strip().splitlines()
    for ln in txt:
        parts = ln.split(",", 1)
        if len(parts) != 2:
            raise ValueError(ln)
        pid, lab = parts[0].strip(), parts[1].strip()
        LABELS_FROM_ANNOTATOR[pid] = lab


LABELS_LINES = """
p1_001,neutral
p1_002,neutral
p1_003,neutral
p1_004,neutral
p1_005,gender_bias
p1_006,neutral
p1_007,neutral
p1_008,neutral
p1_009,neutral
p1_010,neutral
p1_011,profession_bias
p1_012,neutral
p1_013,neutral
p1_014,neutral
p1_015,neutral
p1_016,neutral
p1_017,neutral
p1_018,neutral
p1_019,neutral
p1_020,neutral
p1_021,neutral
p1_022,neutral
p1_023,neutral
p1_024,neutral
p1_025,neutral
p1_026,gender_bias
p1_027,neutral
p1_028,neutral
p1_029,neutral
p1_030,neutral
p1_031,neutral
p1_032,neutral
p1_033,neutral
p1_034,neutral
p1_035,neutral
p1_036,neutral
p1_037,neutral
p1_038,gender_bias
p1_039,neutral
p1_040,neutral
p1_041,neutral
p1_042,neutral
p1_043,neutral
p1_044,neutral
p1_045,neutral
p1_046,neutral
p1_047,neutral
p1_048,neutral
p1_049,neutral
p1_050,neutral
p1_051,nationality_bias
p1_052,neutral
p1_053,neutral
p1_054,neutral
p1_055,neutral
p1_056,neutral
p1_057,neutral
p1_058,neutral
p1_059,neutral
p1_060,neutral
p1_061,neutral
p1_062,neutral
p1_063,neutral
p1_064,neutral
p1_065,neutral
p1_066,neutral
p1_067,neutral
p1_068,neutral
p1_069,neutral
p1_070,gender_bias
p1_071,neutral
p1_072,neutral
p1_073,neutral
p1_074,neutral
p1_075,neutral
p1_076,neutral
p1_077,neutral
p1_078,neutral
p1_079,neutral
p1_080,neutral
p1_081,neutral
p1_082,neutral
p1_083,neutral
p1_084,neutral
p1_085,neutral
p1_086,neutral
p1_087,neutral
p1_088,neutral
p1_089,gender_bias
p1_090,neutral
p1_091,neutral
p1_092,neutral
p1_093,neutral
p1_094,neutral
p1_095,neutral
p1_096,neutral
p1_097,neutral
p1_098,neutral
p1_099,neutral
p1_100,neutral
p1_101,neutral
p1_102,neutral
p1_103,neutral
p1_104,neutral
p1_105,neutral
p1_106,gender_bias
p1_107,neutral
p1_108,neutral
p1_109,neutral
p1_110,neutral
p1_111,neutral
p1_112,neutral
p1_113,neutral
p1_114,neutral
p1_115,neutral
p1_116,neutral
p1_117,neutral
p1_118,neutral
p1_119,neutral
p1_120,neutral
p1_121,neutral
p1_122,neutral
p1_123,neutral
p1_124,neutral
p1_125,neutral
p1_126,neutral
p1_127,neutral
p1_128,neutral
p1_129,neutral
p1_130,neutral
p1_131,neutral
p1_132,neutral
p1_133,neutral
p1_134,neutral
p1_135,neutral
p1_136,neutral
p1_137,neutral
p1_138,neutral
p1_139,neutral
p1_140,neutral
p1_141,neutral
p1_142,neutral
p1_143,neutral
p1_144,neutral
p1_145,neutral
p1_146,neutral
p1_147,neutral
p1_148,neutral
p1_149,neutral
p1_150,neutral
p1_151,gender_bias
p1_152,neutral
p1_153,neutral
p1_154,neutral
p1_155,gender_bias
p1_156,neutral
p1_157,neutral
p1_158,neutral
p1_159,neutral
p1_160,nationality_bias
p1_161,neutral
p1_162,neutral
p1_163,neutral
p1_164,neutral
p1_165,neutral
p1_166,gender_bias
p1_167,neutral
p1_168,neutral
p1_169,neutral
p1_170,neutral
p1_171,neutral
p1_172,neutral
p1_173,neutral
p1_174,neutral
p1_175,neutral
p1_176,neutral
p1_177,neutral
p1_178,neutral
p1_179,neutral
p1_180,neutral
p1_181,neutral
p1_182,neutral
p1_183,neutral
p1_184,neutral
p1_185,neutral
p1_186,neutral
p1_187,gender_bias
p1_188,neutral
p1_189,neutral
p1_190,neutral
p1_191,neutral
p1_192,neutral
p1_193,neutral
p1_194,neutral
p1_195,neutral
p1_196,neutral
p1_197,neutral
p1_198,neutral
p1_199,neutral
p1_200,neutral
"""


def main() -> int:
    _populate_labels()
    if len(LABELS_FROM_ANNOTATOR) != 200:
        print(len(LABELS_FROM_ANNOTATOR), file=sys.stderr)
        raise SystemExit(f"Expected 200 labels, got {len(LABELS_FROM_ANNOTATOR)}")

    raw = json.loads(ARCH.read_text(encoding="utf-8"))
    blob = raw["raw_assistant_content"].strip()
    if blob.startswith("```"):
        blob = blob.split("\n", 1)[1] if "\n" in blob else blob[3:]
        blob = blob.rsplit("```", 1)[0]
    rows_data = json.loads(blob)
    by_id = {str(r["id"]): str(r["text"]).strip() for r in rows_data}

    out_rows: list[dict[str, str]] = []
    for i in range(1, 201):
        pid = f"p1_{i:03d}"
        lab = LABELS_FROM_ANNOTATOR.get(pid)
        tx = by_id.get(pid)
        if not lab or tx is None:
            raise SystemExit(f"Missing label or text for {pid}")
        out_rows.append({"post_id": pid, "text": tx, "label": lab})

    outp = _ROOT / "data" / "evaluation" / "manual_eval_200_labeled.csv"
    with outp.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["post_id", "text", "label"])
        w.writeheader()
        w.writerows(out_rows)

    print(json.dumps({"wrote": str(outp), "n": len(out_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
