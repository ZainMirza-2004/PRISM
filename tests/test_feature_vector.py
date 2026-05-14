"""Meta feature vector contract."""

from __future__ import annotations

import unittest

from models.feature_vector import (
    FEATURE_SCHEMA_VERSION,
    META_FEATURE_NAMES,
    META_FEATURE_NAMES_CORE,
    build_meta_feature_row,
)
from models.rule_signals import RuleFusionSignals


class MetaFeatureVectorTests(unittest.TestCase):
    def test_dimension_and_schema_version(self) -> None:
        self.assertEqual(len(META_FEATURE_NAMES_CORE), 36)
        self.assertEqual(len(META_FEATURE_NAMES), 40)
        self.assertEqual(FEATURE_SCHEMA_VERSION, 6)

    def test_row_length_matches_names(self) -> None:
        dist = {k: 0.25 for k in ("gender_bias", "nationality_bias", "profession_bias", "neutral")}
        rules = RuleFusionSignals(1, 1, 1, 1, 1, 1, 1)
        row = build_meta_feature_row(dist, 0.5, rules, clean_text="women engineers tend to lead")
        self.assertEqual(row.shape[1], len(META_FEATURE_NAMES_CORE))

    def test_row_with_social_aux_is_40(self) -> None:
        dist = {k: 0.25 for k in ("gender_bias", "nationality_bias", "profession_bias", "neutral")}
        dist_s = {"gender_bias": 0.1, "nationality_bias": 0.7, "profession_bias": 0.1, "neutral": 0.1}
        rules = RuleFusionSignals(0, 0, 0, 0, 0, 0, 0)
        row = build_meta_feature_row(
            dist, 0.0, rules, clean_text="test", dist_social=dist_s
        )
        self.assertEqual(row.shape[1], 40)


if __name__ == "__main__":
    unittest.main()
