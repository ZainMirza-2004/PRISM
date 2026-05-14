"""Regression tests for rule_signals + critique gating (see data/eval/rule_signal_fixtures.json)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from models.preprocess import preprocess_social_post
from models.rule_signals import extract_rule_fusion_signals

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE = _ROOT / "data" / "eval" / "rule_signal_fixtures.json"


class RuleSignalFixtureTests(unittest.TestCase):
    def test_fixture_cases(self) -> None:
        data = json.loads(_FIXTURE.read_text(encoding="utf-8"))
        for row in data:
            with self.subTest(case=row.get("id")):
                raw = str(row.get("raw_text") or row.get("text", ""))
                exp = dict(row["expected"])
                rules = extract_rule_fusion_signals(preprocess_social_post(raw))
                got = rules.as_dict()
                for k, v in exp.items():
                    self.assertEqual(
                        got.get(k),
                        v,
                        msg=f"{row.get('id')}: key {k!r} expected {v!r} got {got.get(k)!r}",
                    )


class RuleSignalSanityTests(unittest.TestCase):
    def test_empty_is_zero(self) -> None:
        r = extract_rule_fusion_signals("")
        self.assertEqual(sum(r.as_dict().values()), 0)

    def test_structure_score_unchanged_formula(self) -> None:
        from models.rule_signals import RuleFusionSignals, structure_score_from_rules

        r = RuleFusionSignals(1, 1, 1, 1, 1, 1, 1)
        s = structure_score_from_rules(r)
        self.assertAlmostEqual(s, 0.4 + 0.3 + 0.2 + 0.1)

    def test_pro_group_statement_is_not_generalisation_bias(self) -> None:
        text = "Immigrants are not stealing jobs; they are creating them."
        r = extract_rule_fusion_signals(preprocess_social_post(text))
        self.assertEqual(r.generalisation, 0)

    def test_policy_discussion_is_neutral_in_rules(self) -> None:
        text = "The committee discussed immigration policy and labor protections."
        r = extract_rule_fusion_signals(preprocess_social_post(text))
        self.assertEqual(r.generalisation, 0)
        self.assertEqual(r.preference, 0)

    def test_subtle_harmful_stereotype_triggers_generalisation(self) -> None:
        text = "Women are usually too emotional for crisis leadership."
        r = extract_rule_fusion_signals(preprocess_social_post(text))
        self.assertEqual(r.generalisation, 1)


if __name__ == "__main__":
    unittest.main()
