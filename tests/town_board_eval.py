"""Shared live-evaluation helper for the town-board sample transcript."""

from __future__ import annotations

import json
import os
import re
import sys
import time
import tomllib
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.meeting_analyzer import (
    analyze_transcript_with_gemini,
    analyze_transcript_with_glm,
    analyze_transcript_with_jev,
    analyze_transcript_with_mercury,
    _workflow_source_segments,
)
from utils.workflow_validator import validate_workflow_graph

SAMPLE_TRANSCRIPT_PATH = ROOT / "vids" / "sample transcript.txt"


def load_openrouter_api_key() -> str:
    """Read OPENROUTER_API_KEY from the environment or .streamlit/secrets.toml."""
    env_key = str(os.getenv("OPENROUTER_API_KEY") or "").strip()
    if env_key:
        return env_key
    secrets_path = ROOT / ".streamlit" / "secrets.toml"
    if not secrets_path.is_file():
        return ""
    data = tomllib.loads(secrets_path.read_text(encoding="utf-8"))
    return str(data.get("OPENROUTER_API_KEY") or "").strip()


def load_sample_transcript() -> str:
    """Load the town-board sample transcript from the repo."""
    return SAMPLE_TRANSCRIPT_PATH.read_text(encoding="utf-8")


def _joined_text(items: list[Any], keys: tuple[str, ...]) -> str:
    """Join string fields from dict or string items."""
    parts: list[str] = []
    for item in items:
        if isinstance(item, str):
            parts.append(item)
            continue
        if not isinstance(item, dict):
            continue
        for key in keys:
            value = item.get(key)
            if value:
                parts.append(str(value))
    return " ".join(parts).casefold()


def score_town_board_expectations(payload: dict[str, Any]) -> list[str]:
    """Return expectation misses for the town-board sample; not schema failures."""
    misses: list[str] = []
    nodes = payload.get("nodes") or []
    labels = [str(node.get("label") or "") for node in nodes if isinstance(node, dict)]
    label_blob = " ".join(labels).casefold()
    alternatives = payload.get("alternatives") or []
    confirmations = payload.get("confirmations") or []
    side_blob = _joined_text(alternatives, ("label", "text")) + " " + _joined_text(
        confirmations, ("question",)
    )

    if any(label.strip().casefold() == "complete" for label in labels):
        misses.append("Invented Complete endpoint.")
    if "special election" not in label_blob and "election" not in label_blob:
        misses.append("Chosen special-election direction is missing from node labels.")
    appoint_vs_elect = False
    for node in nodes:
        if not isinstance(node, dict) or node.get("node_type") != "decision":
            continue
        label = str(node.get("label") or "").casefold()
        if "appoint" in label and "elect" in label:
            appoint_vs_elect = True
    if appoint_vs_elect:
        misses.append("Appointment vs election is modeled as a decision diamond.")
    if "appoint" not in side_blob:
        misses.append("Appointment proposal is not kept as an alternative or confirmation.")
    if "petition" not in side_blob:
        misses.append("Voter petition is not kept as an alternative or confirmation.")
    if not any(
        token in side_blob for token in ("notice", "timing", "deadline", "days")
    ):
        misses.append("Notice timing is not marked unresolved in confirmations.")
    lanes = [str(lane) for lane in payload.get("lanes") or []]
    node_lanes = [
        str(node.get("lane") or "")
        for node in nodes
        if isinstance(node, dict)
    ]
    if node_lanes and all(lane == "Participant" for lane in node_lanes):
        misses.append("Unknown owners were mapped onto Participant instead of Unassigned.")
    meaningful = [
        node
        for node in nodes
        if isinstance(node, dict)
        and str(node.get("node_type") or "") not in {"start", "end"}
    ]
    if not (5 <= len(meaningful) <= 8) and not (5 <= len(nodes) <= 8):
        misses.append(
            f"Node count {len(nodes)} (meaningful {len(meaningful)}) is outside the 5–8 check."
        )
    for node in nodes:
        if not isinstance(node, dict):
            continue
        label = str(node.get("label") or "").casefold()
        status = str(node.get("status") or "").casefold()
        node_type = str(node.get("node_type") or "").casefold()
        completed = any(token in label for token in ("concluded", "seated", "complete"))
        if completed and (status == "planned" or node_type == "end"):
            misses.append(
                f"Completed-result wording on planned/end node {node.get('id')}."
            )
    graph_blob = label_blob + " " + _joined_text(payload.get("edges") or [], ("label",))
    conf_blob = _joined_text(confirmations, ("question",))
    has_day_count = bool(
        re.search(
            r"\b\d{1,3}\s*(?:[-–/]\s*\d{1,3})?\s*-?\s*days?\b",
            graph_blob,
            re.IGNORECASE,
        )
    )
    timing_disputed = any(
        token in conf_blob for token in ("days", "timing", "deadline", "notice")
    )
    if has_day_count and timing_disputed:
        misses.append(
            "A day count is asserted in a label while confirmations still dispute timing."
        )
    return misses


def score_utterance_coverage(payload: dict[str, Any], source_count: int) -> list[str]:
    """Flag a model log that would replace the app feed with fewer entries."""
    utterances = payload.get("utterances")
    if not isinstance(utterances, list):
        return ["Model did not return an utterance list."]
    if len(utterances) != source_count:
        return [
            f"Utterance coverage {len(utterances)}/{source_count}; "
            "the app may show a shortened transcript feed."
        ]
    return []


def score_petition_evidence(payload: dict[str, Any]) -> list[str]:
    """Check the fixture's known petition excerpts, not arbitrary IDs."""
    petition_sources = {"U56", "U67"}
    unrelated_sources = {"U13", "U14"}
    for item in payload.get("alternatives") or []:
        if not isinstance(item, dict):
            continue
        if "petition" not in str(item.get("label") or "").casefold():
            continue
        cited = set(item.get("source_ids") or [])
        misses = []
        if not cited & petition_sources:
            misses.append("Petition alternative cites no petition excerpt (U56 or U67).")
        if cited & unrelated_sources:
            misses.append("Petition alternative cites the unrelated seat discussion (U13/U14).")
        return misses
    return ["Petition alternative is missing."]


def _engine_fn(engine: str) -> Callable[..., dict[str, Any]]:
    """Return the analyzer function for a named engine."""
    if engine == "jev":
        return analyze_transcript_with_jev
    if engine == "glm":
        return analyze_transcript_with_glm
    if engine == "gemini":
        return analyze_transcript_with_gemini
    return analyze_transcript_with_mercury


def _sample_json_path(engine: str) -> Path:
    """Return the JSON output path for a live engine sample in the tests folder."""
    return TESTS_DIR / f"test_{engine}_sample.json"


def save_engine_sample_json(engine: str, report: dict[str, Any]) -> Path:
    """Write a live engine sample report to JSON in the tests folder."""
    path = _sample_json_path(engine)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return path


def _failure_report(engine: str, elapsed_seconds: float, error: str) -> dict[str, Any]:
    """Build a JSON report for a failed engine sample run."""
    return {
        "engine": engine,
        "ok": False,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "error": error,
    }


def _success_report(
    engine: str,
    payload: dict[str, Any],
    elapsed_seconds: float,
    misses: list[str],
) -> dict[str, Any]:
    """Build a JSON report for a successful engine sample run."""
    nodes = payload.get("nodes") or []
    return {
        "engine": engine,
        "ok": True,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "node_count": len(nodes),
        "edge_count": len(payload.get("edges") or []),
        "lane_count": len(payload.get("lanes") or []),
        "labels": [node.get("label") for node in nodes if isinstance(node, dict)],
        "confirmations": payload.get("confirmations") or [],
        "alternatives": payload.get("alternatives") or [],
        "validation_warnings": payload.get("validation_warnings") or [],
        "expectation_misses": misses,
        "payload": payload,
    }


def print_workflow_report(report: dict[str, Any]) -> None:
    """Print a compact workflow evaluation report."""
    print(f"engine: {report['engine']}")
    print(f"elapsed_seconds: {report['elapsed_seconds']:.1f}")
    if not report.get("ok"):
        print(f"FAILURE: {report.get('error')}")
        return
    print(
        f"nodes: {report['node_count']} edges: {report['edge_count']} "
        f"lanes: {report['lane_count']}"
    )
    print(f"labels: {report['labels']}")
    print(f"confirmations: {report['confirmations']}")
    print(f"alternatives: {report['alternatives']}")
    print(f"validation_warnings: {report['validation_warnings']}")
    misses = report.get("expectation_misses") or []
    if misses:
        print("expectation_misses:")
        for miss in misses:
            print(f"- {miss}")
    else:
        print("expectation_misses: none")


def _write_and_print_report(engine: str, report: dict[str, Any]) -> Path:
    """Save the report JSON in tests/ and print the compact summary."""
    path = save_engine_sample_json(engine, report)
    print_workflow_report(report)
    print(f"wrote: {path}")
    return path


def run_engine_sample(engine: str) -> int:
    """Run one engine on the town-board transcript and save the evaluation JSON."""
    api_key = load_openrouter_api_key()
    if not api_key:
        error = "OPENROUTER_API_KEY is missing from the environment and .streamlit/secrets.toml."
        _write_and_print_report(engine, _failure_report(engine, 0.0, error))
        return 1
    transcript = load_sample_transcript().strip()
    if not transcript:
        error = f"Sample transcript is empty: {SAMPLE_TRANSCRIPT_PATH}"
        _write_and_print_report(engine, _failure_report(engine, 0.0, error))
        return 1
    started = time.perf_counter()
    try:
        payload = _engine_fn(engine)(transcript, api_key=api_key, translate_to_english=False)
        validate_workflow_graph(payload)
    except Exception as error:
        elapsed = time.perf_counter() - started
        _write_and_print_report(engine, _failure_report(engine, elapsed, str(error)))
        return 1
    elapsed = time.perf_counter() - started
    source_count = len(_workflow_source_segments(transcript))
    misses = score_town_board_expectations(payload)
    misses.extend(score_utterance_coverage(payload, source_count))
    misses.extend(score_petition_evidence(payload))
    _write_and_print_report(engine, _success_report(engine, payload, elapsed, misses))
    return 0
