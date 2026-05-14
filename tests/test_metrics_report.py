from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from model_evaluation.metrics import save_report


class ConfusionMatrixReportFormatTests(unittest.TestCase):
    def test_confusion_rows_are_human_readable_numbers(self) -> None:
        multiclass = {
            "labels_order": ["gender_bias", "nationality_bias", "profession_bias", "neutral"],
            "confusion_matrix": [[10, 0, 1, 2], [0, 9, 0, 1]],
        }
        binary = {"confusion_matrix_binary": [[7, 1], [2, 11]]}

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            save_report(out_dir, multiclass, binary, errors=[])
            txt = (out_dir / "confusion_matrices.txt").read_text(encoding="utf-8")

        self.assertIn("10 0 1 2", txt)
        self.assertIn("0 9 0 1", txt)
        self.assertIn("7 1", txt)
        self.assertIn("2 11", txt)
        self.assertNotIn("<generator object", txt)


if __name__ == "__main__":
    unittest.main()
