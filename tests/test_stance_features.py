from __future__ import annotations

import unittest

from models.stance_features import extract_stance_features


class StanceFeatureTests(unittest.TestCase):
    def test_denial_is_detected(self) -> None:
        s = extract_stance_features("Immigrants are not stealing jobs.")
        self.assertEqual(s.attribution_denial, 1.0)
        self.assertEqual(s.attribution_critique_of_stereotype, 0.0)

    def test_critique_is_detected(self) -> None:
        s = extract_stance_features("It is a stereotype that women are weak leaders.")
        self.assertEqual(s.attribution_critique_of_stereotype, 1.0)

    def test_endorsement_and_normative_detected(self) -> None:
        s = extract_stance_features("We should prefer local candidates over foreigners.")
        self.assertEqual(s.attribution_endorsement, 1.0)
        self.assertGreater(s.normative_language_score, 0.0)


if __name__ == "__main__":
    unittest.main()
