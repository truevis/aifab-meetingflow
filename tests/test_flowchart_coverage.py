"""Unit tests for Mermaid side blob, disconnected processes, and coverage filler."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.flowchart_generator import (
    SIDE_BLOB_CAP,
    SIDE_BLOB_TITLE,
    build_mermaid_flowchart,
    weakly_connected_components,
)
from utils.meeting_analyzer import (
    _fill_side_notes_and_excerpts,
    _finalize_workflow_payload,
)


def _two_process_payload() -> dict:
    """Return two processes with no connecting edge."""
    return {
        "lanes": ["Clerk", "Board"],
        "nodes": [
            {
                "id": "N1",
                "lane": "Clerk",
                "label": "Issue appreciation certificate",
                "node_type": "action",
                "source_ids": ["U1"],
            },
            {
                "id": "N2",
                "lane": "Board",
                "label": "Appoint constable",
                "node_type": "action",
                "source_ids": ["U2"],
            },
        ],
        "edges": [],
        "confirmations": [],
        "alternatives": [],
        "facts": [],
        "utterances": [
            {
                "id": 1,
                "text": "Issue appreciation certificate",
                "is_business": True,
                "action_type": "add",
            },
            {
                "id": 2,
                "text": "Appoint constable",
                "is_business": True,
                "action_type": "add",
            },
        ],
    }


class SideBlobMermaidTests(unittest.TestCase):
    def test_side_subgraph_uses_s_ids_without_edges(self) -> None:
        chart = build_mermaid_flowchart(
            [
                {
                    "id": "N1",
                    "lane": "Board",
                    "label": "Warn election",
                    "node_type": "action",
                }
            ],
            [],
            lanes=["Board"],
            side_notes=[
                {"label": "Board appointment"},
                {"label": "Voter petition"},
            ],
        )
        self.assertIn(SIDE_BLOB_TITLE, chart)
        self.assertIn('S1["Board appointment"]', chart)
        self.assertIn('S2["Voter petition"]', chart)
        self.assertNotIn("S1 -->", chart)
        self.assertNotIn("--> S1", chart)
        self.assertNotIn("N1 --> S", chart)

    def test_disconnected_processes_stay_unconnected(self) -> None:
        payload = _two_process_payload()
        components = weakly_connected_components(payload["nodes"], payload["edges"])
        self.assertEqual(len(components), 2)
        chart = build_mermaid_flowchart(
            payload["nodes"],
            payload["edges"],
            lanes=payload["lanes"],
        )
        self.assertIn("Issue appreciation certificate", chart)
        self.assertIn("Appoint constable", chart)
        self.assertNotIn("N1 --> N2", chart)
        self.assertNotIn("N2 --> N1", chart)


class UncitedCoverageFillerTests(unittest.TestCase):
    def test_uncited_business_becomes_original_side_note(self) -> None:
        payload = {
            "lanes": ["Board"],
            "nodes": [
                {
                    "id": "N1",
                    "lane": "Board",
                    "label": "Warn election",
                    "node_type": "action",
                    "source_ids": ["U1"],
                }
            ],
            "edges": [],
            "confirmations": [],
            "alternatives": [],
            "facts": [],
            "utterances": [
                {
                    "id": 1,
                    "text": "Warn the election",
                    "is_business": True,
                    "action_type": "add",
                },
                {
                    "id": 2,
                    "text": "Voters could petition for the meeting",
                    "is_business": True,
                    "action_type": "add",
                },
                {
                    "id": 3,
                    "text": "Nice weather today",
                    "is_business": False,
                    "action_type": "none",
                },
            ],
        }
        segments = [
            {"source_id": "U1", "text": "Warn the election"},
            {"source_id": "U2", "text": "Voters could petition for the meeting"},
            {"source_id": "U3", "text": "Nice weather today"},
        ]
        filled = _finalize_workflow_payload(payload, segments)
        labels = [str(note.get("label") or "") for note in filled["side_notes"]]
        self.assertTrue(
            any("Voters could petition for the meeting" in label for label in labels)
        )
        self.assertFalse(any("Nice weather" in label for label in labels))

    def test_overflow_past_cap_goes_to_uncited_excerpts(self) -> None:
        alternatives = [
            {"label": f"Alternative option {index}", "source_ids": []}
            for index in range(1, SIDE_BLOB_CAP + 4)
        ]
        payload = {
            "lanes": [],
            "nodes": [],
            "edges": [],
            "confirmations": [],
            "alternatives": alternatives,
            "facts": [],
            "utterances": [],
        }
        filled = _fill_side_notes_and_excerpts(payload, [])
        self.assertEqual(len(filled["side_notes"]), SIDE_BLOB_CAP)
        self.assertEqual(len(filled["uncited_excerpts"]), 3)
        self.assertEqual(
            filled["uncited_excerpts"][0]["label"],
            f"Alternative option {SIDE_BLOB_CAP + 1}",
        )

    def test_finalize_strips_disputed_day_counts_from_edges(self) -> None:
        payload = {
            "lanes": ["Board"],
            "nodes": [
                {
                    "id": "N1",
                    "lane": "Board",
                    "label": "Warn election",
                    "node_type": "action",
                    "source_ids": ["U1"],
                },
                {
                    "id": "N2",
                    "lane": "Board",
                    "label": "Hold meeting",
                    "node_type": "action",
                    "source_ids": ["U2"],
                },
            ],
            "edges": [
                {
                    "source": "N1",
                    "target": "N2",
                    "label": "30 day notice sent",
                    "source_ids": ["U2"],
                }
            ],
            "confirmations": [
                {
                    "id": 1,
                    "question": "Is the notice period 30 or 45 days from warning?",
                    "suggested_role": "Board",
                    "confidence": 0.5,
                    "source_ids": ["U2"],
                }
            ],
            "alternatives": [],
            "facts": [],
            "utterances": [],
        }
        segments = [
            {"source_id": "U1", "text": "Warn election"},
            {"source_id": "U2", "text": "Maybe thirty days"},
        ]
        filled = _finalize_workflow_payload(payload, segments)
        edge_label = str((filled["edges"][0].get("label") or "")).casefold()
        self.assertNotRegex(edge_label, r"\b30\b")
        self.assertNotRegex(edge_label, r"\bdays?\b")


if __name__ == "__main__":
    unittest.main()
