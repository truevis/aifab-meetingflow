import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.town_board_eval import (
    score_petition_evidence,
    score_utterance_coverage,
)


class TownBoardEvalHelperTests(unittest.TestCase):
    def test_glm_style_log_is_coverage_miss(self) -> None:
        payload = {"utterances": [{"id": index} for index in range(1, 22)]}
        misses = score_utterance_coverage(payload, 98)
        self.assertEqual(len(misses), 1)
        self.assertIn("21/98", misses[0])

    def test_matching_coverage_has_no_miss(self) -> None:
        payload = {"utterances": [{"id": index} for index in range(1, 99)]}
        self.assertEqual(score_utterance_coverage(payload, 98), [])

    def test_petition_citing_unrelated_ids_is_evidence_miss(self) -> None:
        payload = {
            "alternatives": [
                {"label": "Voter petition", "source_ids": ["U13", "U14"]},
            ]
        }
        misses = score_petition_evidence(payload)
        self.assertTrue(
            any("U13" in miss or "unrelated" in miss.casefold() for miss in misses)
        )

    def test_petition_citing_u56_passes(self) -> None:
        payload = {
            "alternatives": [
                {"label": "Voter petition to force election", "source_ids": ["U56"]},
            ]
        }
        self.assertEqual(score_petition_evidence(payload), [])


if __name__ == "__main__":
    unittest.main()
