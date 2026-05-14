from __future__ import annotations

import unittest

from models.linguistic_features import compute_linguistic_features
from models.preprocess import preprocess_social_post
from models.rule_signals import extract_rule_fusion_signals


class LinguisticFeatureTests(unittest.TestCase):
    def test_exclusion_intent_detected(self) -> None:
        text = "We should not hire immigrants for client-facing roles."
        clean = preprocess_social_post(text)
        rules = extract_rule_fusion_signals(clean)
        feat = compute_linguistic_features(clean, rules)
        self.assertEqual(feat.exclusion_intent, 1.0)

    def test_anti_stereotype_cue_detected(self) -> None:
        text = "It is a myth that immigrants are stealing jobs."
        clean = preprocess_social_post(text)
        rules = extract_rule_fusion_signals(clean)
        feat = compute_linguistic_features(clean, rules)
        self.assertEqual(feat.anti_stereotype_cue, 1.0)

    def test_target_negative_sentiment_high_for_group_attack(self) -> None:
        text = "Immigrants are a harmful burden and a serious problem."
        clean = preprocess_social_post(text)
        rules = extract_rule_fusion_signals(clean)
        feat = compute_linguistic_features(clean, rules)
        self.assertGreater(feat.target_negative_sentiment, 0.0)


if __name__ == "__main__":
    unittest.main()
