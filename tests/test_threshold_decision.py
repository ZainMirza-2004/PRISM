from __future__ import annotations

import unittest

from models.label_config import LABELS, NEUTRAL_LABEL
from models.threshold_optimizer import predicted_class_with_thresholds


class ThresholdDecisionTests(unittest.TestCase):
    def test_none_pass_falls_back_neutral(self) -> None:
        post = {c: 0.05 for c in LABELS}
        th = {c: 0.5 for c in LABELS}
        self.assertEqual(predicted_class_with_thresholds(post, th, classes_ordered=LABELS), NEUTRAL_LABEL)

    def test_highest_among_passing(self) -> None:
        post = {"gender_bias": 0.4, "nationality_bias": 0.7, "profession_bias": 0.2, "neutral": 0.1}
        th = {"gender_bias": 0.35, "nationality_bias": 0.6, "profession_bias": 0.5, "neutral": 0.5}
        self.assertEqual(predicted_class_with_thresholds(post, th, classes_ordered=LABELS), "nationality_bias")

    def test_ambiguity_margin_prefers_neutral(self) -> None:
        post = {"gender_bias": 0.45, "nationality_bias": 0.1, "profession_bias": 0.1, "neutral": 0.42}
        th = {"gender_bias": 0.4, "nationality_bias": 0.5, "profession_bias": 0.5, "neutral": 0.5}
        self.assertEqual(
            predicted_class_with_thresholds(post, th, classes_ordered=LABELS, ambiguity_margin=0.1),
            NEUTRAL_LABEL,
        )
        self.assertEqual(
            predicted_class_with_thresholds(post, th, classes_ordered=LABELS, ambiguity_margin=None),
            "gender_bias",
        )


if __name__ == "__main__":
    unittest.main()
