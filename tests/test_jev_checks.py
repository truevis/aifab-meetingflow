import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.meeting_analyzer import (
    _alternative_is_similar,
    _apply_jev_verdicts,
    _collect_jev_claims,
    _merge_jev_alternatives,
    _shortlist_unused_alternative_segments,
    _strip_day_counts,
)


def _segments() -> list[dict[str, str]]:
    """Return cited source segments for offline Jev-check fixtures."""
    return [
        {
            "source_id": "U1",
            "speaker": "Chair",
            "text": "We will warn the meeting.",
        },
        {
            "source_id": "U2",
            "speaker": "Clerk",
            "text": "Notice needs thirty days.",
        },
        {
            "source_id": "U56",
            "speaker": "Voter",
            "text": "The voters could petition to have this meeting.",
        },
        {
            "source_id": "U73",
            "speaker": "Clerk",
            "text": "Not less than 30, no more than 40 days from the date it is warned.",
        },
    ]


def _graph_payload() -> dict:
    """Return a small Mercury-style graph with an edge, number, and alternative."""
    return {
        "lanes": ["Board"],
        "nodes": [
            {
                "id": "N1",
                "lane": "Board",
                "label": "Start",
                "node_type": "start",
                "source_ids": ["U1"],
                "status": "planned",
            },
            {
                "id": "N2",
                "lane": "Board",
                "label": "Warn Meeting (30-day rule)",
                "node_type": "action",
                "source_ids": ["U2", "U73"],
                "status": "adopted",
            },
            {
                "id": "N3",
                "lane": "Board",
                "label": "Special election concluded",
                "node_type": "end",
                "source_ids": ["U1"],
                "status": "planned",
            },
        ],
        "edges": [
            {
                "source": "N1",
                "target": "N2",
                "label": "30-45 day wait",
                "source_ids": ["U73"],
                "status": "planned",
            },
            {
                "source": "N2",
                "target": "N3",
                "label": None,
                "source_ids": ["U1"],
                "status": "planned",
            },
        ],
        "confirmations": [],
        "alternatives": [
            {
                "label": "Board Appointment",
                "source_ids": ["U32"],
                "status": "alternative",
            }
        ],
        "facts": [
            {
                "text": "Notice is a 30-day rule",
                "source_ids": ["U2"],
            }
        ],
    }


class JevClaimCollectionTests(unittest.TestCase):
    def test_collects_edges_nodes_alternatives_and_facts(self) -> None:
        claims = _collect_jev_claims(_graph_payload())
        kinds = [claim["kind"] for claim in claims]
        texts = [claim["text"] for claim in claims]
        self.assertIn("edge", kinds)
        self.assertIn("node", kinds)
        self.assertIn("alternative", kinds)
        self.assertIn("fact", kinds)
        self.assertTrue(any("leads to" in text for text in texts))
        self.assertTrue(any("30-45 day wait" in text for text in texts))
        self.assertIn("Board Appointment", texts)
        self.assertIn("Notice is a 30-day rule", texts)


class JevNumberStripTests(unittest.TestCase):
    def test_strip_day_counts_removes_range_and_rule_phrase(self) -> None:
        self.assertEqual(_strip_day_counts("Warn Meeting (30-day rule)"), "Warn Meeting")
        self.assertEqual(_strip_day_counts("30-45 day wait"), "")
        self.assertEqual(_strip_day_counts("Hold election"), "Hold election")


class JevVerdictTests(unittest.TestCase):
    def test_high_confidence_contradiction_drops_edge(self) -> None:
        payload = _graph_payload()
        claims = _collect_jev_claims(payload)
        answers = {
            "c_e0": {"choice": "contradicts", "confidence": 0.91},
            "c_e1": {"choice": "supports", "confidence": 0.95},
        }
        _apply_jev_verdicts(payload, answers, claims, [], _segments())
        remaining = {
            (edge.get("source"), edge.get("target"))
            for edge in payload["edges"]
            if isinstance(edge, dict)
        }
        self.assertNotIn(("N1", "N2"), remaining)
        self.assertIn(("N2", "N3"), remaining)
        self.assertTrue(payload["confirmations"])
        self.assertEqual(payload["confirmations"][0]["confidence"], 0.91)

    def test_uncertain_or_says_nothing_keeps_edge(self) -> None:
        payload = _graph_payload()
        claims = _collect_jev_claims(payload)
        answers = {
            "c_e0": {"choice": "says_nothing", "confidence": 0.4},
            "c_e1": {"choice": "contradicts", "confidence": 0.5},
        }
        _apply_jev_verdicts(payload, answers, claims, [], _segments())
        remaining = {
            (edge.get("source"), edge.get("target"))
            for edge in payload["edges"]
            if isinstance(edge, dict)
        }
        self.assertEqual({("N1", "N2"), ("N2", "N3")}, remaining)
        self.assertGreaterEqual(len(payload["confirmations"]), 2)

    def test_contradicted_number_is_stripped_and_unresolved(self) -> None:
        payload = _graph_payload()
        claims = _collect_jev_claims(payload)
        answers = {
            "c_nN2": {"choice": "contradicts", "confidence": 0.88},
        }
        _apply_jev_verdicts(payload, answers, claims, [], _segments())
        node = next(item for item in payload["nodes"] if item["id"] == "N2")
        self.assertNotIn("30", str(node.get("label") or ""))
        self.assertEqual(node.get("status"), "unresolved")
        self.assertTrue(
            any("30-day" in str(item.get("question") or "") for item in payload["confirmations"])
        )

    def test_unestablished_end_is_demoted(self) -> None:
        payload = _graph_payload()
        claims = _collect_jev_claims(payload)
        answers = {
            "p_nN3": {"noul": 0.12},
        }
        _apply_jev_verdicts(payload, answers, claims, [], _segments())
        node = next(item for item in payload["nodes"] if item["id"] == "N3")
        self.assertEqual(node.get("node_type"), "action")
        self.assertEqual(node.get("status"), "planned")
        self.assertEqual(payload["confirmations"][0]["confidence"], 0.12)

    def test_uncertain_noul_does_not_demote(self) -> None:
        payload = _graph_payload()
        claims = _collect_jev_claims(payload)
        answers = {
            "p_nN3": {"noul": 0.5},
        }
        _apply_jev_verdicts(payload, answers, claims, [], _segments())
        node = next(item for item in payload["nodes"] if item["id"] == "N3")
        self.assertEqual(node.get("node_type"), "end")
        self.assertEqual(node.get("status"), "planned")


class JevAlternativeMergeTests(unittest.TestCase):
    def test_shortlist_keeps_only_token_matches(self) -> None:
        candidates = _shortlist_unused_alternative_segments(
            _segments()
            + [{"source_id": "U10", "speaker": "Chair", "text": "Please take your seats."}]
        )
        source_ids = {item["source_id"] for item in candidates}
        self.assertIn("U56", source_ids)
        self.assertNotIn("U10", source_ids)
        self.assertNotIn("U1", source_ids)

    def test_high_confidence_unused_alternative_is_appended(self) -> None:
        payload = _graph_payload()
        candidates = [
            {
                "source_id": "U56",
                "text": "The voters could petition to have this meeting.",
            }
        ]
        answers = {
            "alt_U56": {"choice": "unused_alternative", "confidence": 0.86},
        }
        _merge_jev_alternatives(payload, answers, candidates)
        labels = [str(item.get("label") or "") for item in payload["alternatives"]]
        self.assertTrue(any("petition" in label.casefold() for label in labels))
        self.assertTrue(any("Appointment" in label for label in labels))

    def test_similar_alternative_is_not_duplicated(self) -> None:
        payload = _graph_payload()
        payload["alternatives"].append(
            {
                "label": "Voter petition for an earlier meeting",
                "source_ids": ["U67"],
                "status": "alternative",
            }
        )
        candidates = [
            {
                "source_id": "U56",
                "text": "The voters could petition to have this meeting.",
            }
        ]
        answers = {
            "alt_U56": {"choice": "unused_alternative", "confidence": 0.9},
        }
        _merge_jev_alternatives(payload, answers, candidates)
        petition_alts = [
            item
            for item in payload["alternatives"]
            if "petition" in str(item.get("label") or "").casefold()
        ]
        self.assertEqual(len(petition_alts), 1)

    def test_same_source_is_treated_as_similar(self) -> None:
        existing = {"label": "Board Appointment", "source_ids": ["U32"]}
        self.assertTrue(_alternative_is_similar(existing, "U32", "Appoint someone today"))

    def test_explicit_reject_drops_mercury_alternative(self) -> None:
        payload = _graph_payload()
        candidates = [
            {
                "source_id": "U32",
                "text": "We already decided to appoint someone today.",
            }
        ]
        answers = {
            "alt_U32": {"choice": "adopted_step", "confidence": 0.93},
        }
        _merge_jev_alternatives(payload, answers, candidates)
        self.assertEqual(payload["alternatives"], [])


if __name__ == "__main__":
    unittest.main()
