import unittest

from utils.workflow_validator import validate_workflow_graph


def _decision_fixture() -> dict:
    """Return a small graph with an intentionally incomplete fork."""
    return {
        "lanes": ["Reviewer"],
        "nodes": [
            {"id": "N1", "lane": "Reviewer", "label": "Approved?",
             "node_type": "decision"},
            {"id": "N2", "lane": "Reviewer", "label": "Release document",
             "node_type": "action"},
            {"id": "N3", "lane": "Reviewer", "label": "Request revisions",
             "node_type": "action"},
        ],
        "edges": [{"source": "N1", "target": "N2", "label": "Yes"}],
        "confirmations": [],
    }


class WorkflowGraphTests(unittest.TestCase):
    def test_rejects_single_exit_decision(self) -> None:
        with self.assertRaisesRegex(ValueError, "distinct next steps"):
            validate_workflow_graph(_decision_fixture())

    def test_accepts_labeled_fork(self) -> None:
        graph = _decision_fixture()
        graph["edges"].append(
            {"source": "N1", "target": "N3", "label": "No"}
        )
        self.assertIs(validate_workflow_graph(graph), graph)

    def test_accepts_empty_graph(self) -> None:
        graph = {
            "lanes": [],
            "nodes": [],
            "edges": [],
            "confirmations": [],
        }
        self.assertIs(validate_workflow_graph(graph), graph)

    def test_rejects_unknown_source_ids(self) -> None:
        graph = {
            "lanes": ["Clerk"],
            "nodes": [
                {
                    "id": "N1",
                    "lane": "Clerk",
                    "label": "File notice",
                    "node_type": "action",
                    "source_ids": ["U99"],
                }
            ],
            "edges": [],
            "confirmations": [],
        }
        with self.assertRaisesRegex(ValueError, "Unknown source_id"):
            validate_workflow_graph(graph, known_source_ids={"U1"})

    def test_warns_disputed_day_count_vs_confirmation(self) -> None:
        graph = {
            "lanes": ["Clerk"],
            "nodes": [
                {
                    "id": "N1",
                    "lane": "Clerk",
                    "label": "Issue notice (30-day rule)",
                    "node_type": "action",
                    "status": "adopted",
                }
            ],
            "edges": [],
            "confirmations": [
                {
                    "id": 1,
                    "question": "Is the notice period 30 or 45 days from resignation?",
                    "suggested_role": "Clerk",
                    "confidence": 0.5,
                    "source_ids": ["U1"],
                }
            ],
        }
        result = validate_workflow_graph(graph, known_source_ids={"U1"})
        self.assertIs(result, graph)
        self.assertTrue(
            any("day count" in warning.casefold() for warning in result["validation_warnings"])
        )

    def test_warns_without_failing_on_field_agreement(self) -> None:
        graph = {
            "lanes": ["Board"],
            "nodes": [
                {
                    "id": "N1",
                    "lane": "Board",
                    "label": "Special election concluded",
                    "node_type": "end",
                    "status": "planned",
                }
            ],
            "edges": [],
            "confirmations": [],
        }
        result = validate_workflow_graph(graph)
        self.assertIs(result, graph)
        self.assertTrue(
            any("completed result" in warning.casefold() for warning in result["validation_warnings"])
        )

    def test_rejects_duplicate_node_ids(self) -> None:
        graph = {
            "lanes": ["Clerk"],
            "nodes": [
                {"id": "N1", "lane": "Clerk", "label": "File notice", "node_type": "action"},
                {"id": "N1", "lane": "Clerk", "label": "Post notice", "node_type": "action"},
            ],
            "edges": [],
            "confirmations": [],
        }
        with self.assertRaisesRegex(ValueError, "Duplicate node ID"):
            validate_workflow_graph(graph)


if __name__ == "__main__":
    unittest.main()
