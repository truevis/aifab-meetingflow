import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.meeting_analyzer import merge_utterance_annotations


def _source_turns() -> list[dict]:
    """Return two source utterances with identity fields only."""
    return [
        {"id": 1, "timestamp": "00:01", "speaker": "Chair", "text": "We will warn the meeting."},
        {"id": 2, "timestamp": "00:02", "speaker": "Clerk", "text": "Notice needs thirty days."},
    ]


class UtteranceMergeTests(unittest.TestCase):
    def test_id_mismatch_returns_source_list(self) -> None:
        source = _source_turns()
        model = [
            {
                "id": 1,
                "timestamp": "00:01",
                "speaker": "Chair",
                "text": "We will warn the meeting.",
                "is_business": True,
                "relevance_score": 0.9,
                "action_type": "add",
                "detected_step": "Warn meeting",
                "detected_role": "Chair",
            }
        ]
        merged = merge_utterance_annotations(source, model)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["text"], source[0]["text"])
        self.assertNotIn("is_business", merged[0])
        self.assertNotIn("detected_step", merged[1])

    def test_text_mismatch_keeps_source_row(self) -> None:
        source = _source_turns()
        model = [
            {
                "id": 1,
                "timestamp": "99:99",
                "speaker": "Other",
                "text": "Summarized opening remarks.",
                "is_business": True,
                "relevance_score": 0.9,
                "action_type": "add",
                "detected_step": "Warn meeting",
                "detected_role": "Chair",
            },
            {
                "id": 2,
                "timestamp": "00:02",
                "speaker": "Clerk",
                "text": "Notice needs thirty days.",
                "is_business": True,
                "relevance_score": 0.8,
                "action_type": "add",
                "detected_step": "Issue notice",
                "detected_role": "Clerk",
            },
        ]
        merged = merge_utterance_annotations(source, model)
        self.assertEqual(merged[0]["timestamp"], "00:01")
        self.assertEqual(merged[0]["speaker"], "Chair")
        self.assertEqual(merged[0]["text"], source[0]["text"])
        self.assertNotIn("is_business", merged[0])
        self.assertEqual(merged[1]["detected_step"], "Issue notice")
        self.assertEqual(merged[1]["text"], source[1]["text"])

    def test_matching_ids_copy_annotation_fields_only(self) -> None:
        source = _source_turns()
        model = [
            {
                "id": 1,
                "timestamp": "99:99",
                "speaker": "Other",
                "text": "We will warn the meeting.",
                "is_business": True,
                "relevance_score": 0.91,
                "action_type": "add",
                "detected_step": "Warn meeting",
                "detected_role": "Selectboard",
            },
            {
                "id": 2,
                "timestamp": "88:88",
                "speaker": "Other",
                "text": "Notice needs thirty days.",
                "is_business": False,
                "relevance_score": 0.1,
                "action_type": "none",
                "detected_step": None,
                "detected_role": None,
            },
        ]
        merged = merge_utterance_annotations(source, model)
        self.assertEqual(merged[0]["timestamp"], "00:01")
        self.assertEqual(merged[0]["speaker"], "Chair")
        self.assertEqual(merged[0]["text"], source[0]["text"])
        self.assertTrue(merged[0]["is_business"])
        self.assertEqual(merged[0]["detected_step"], "Warn meeting")
        self.assertEqual(merged[0]["detected_role"], "Selectboard")
        self.assertEqual(merged[1]["timestamp"], "00:02")
        self.assertFalse(merged[1]["is_business"])
        self.assertEqual(merged[1]["action_type"], "none")


if __name__ == "__main__":
    unittest.main()
