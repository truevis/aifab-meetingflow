"""Structural validation for transcript-extracted workflow graphs."""

from __future__ import annotations

import re
from typing import Any

_NODE_ID_RE = re.compile(r"N[1-9]\d*")
_ALLOWED_NODE_TYPES = {"start", "action", "decision", "end"}
_ALLOWED_STATUS = {"adopted", "planned", "proposed", "alternative", "unresolved"}


def normalize_incomplete_decisions(payload: dict[str, Any]) -> dict[str, Any]:
    """Demote decisions that lack two distinct labeled outcomes; do not invent branches."""
    if not isinstance(payload, dict):
        return payload
    nodes = payload.get("nodes")
    edges = payload.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        return payload
    confirmations = payload.get("confirmations")
    if not isinstance(confirmations, list):
        confirmations = []
        payload["confirmations"] = confirmations

    outgoing: dict[str, list[dict[str, Any]]] = {}
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        source = edge.get("source")
        if isinstance(source, str):
            outgoing.setdefault(source, []).append(edge)

    for node in nodes:
        if not isinstance(node, dict) or node.get("node_type") != "decision":
            continue
        node_id = node.get("id")
        if not isinstance(node_id, str):
            continue
        branches = outgoing.get(node_id, [])
        targets: set[str] = set()
        conditions: list[str] = []
        for edge in branches:
            target = edge.get("target")
            if isinstance(target, str) and target.strip():
                targets.add(target)
            label = edge.get("label")
            if label is not None and not isinstance(label, str):
                conditions.append("")
            else:
                conditions.append((label or "").strip().casefold())
        unique_labels = [condition for condition in conditions if condition]
        has_two_labeled_outcomes = (
            len(targets) >= 2
            and len(conditions) >= 2
            and all(conditions)
            and len(set(conditions)) == len(conditions)
            and len(unique_labels) == len(conditions)
        )
        if has_two_labeled_outcomes:
            continue
        node["node_type"] = "action"
        label = str(node.get("label") or node_id).strip() or node_id
        confirmations.append(
            {
                "id": len(confirmations) + 1,
                "question": (
                    f"Which distinct conditional outcomes follow '{label}'? "
                    "The transcript did not support a complete decision branch."
                ),
                "suggested_role": str(node.get("lane") or "Unassigned"),
                "confidence": 0.0,
            }
        )
    return payload


def _iter_source_id_holders(payload: dict[str, Any]) -> list[tuple[str, list[Any]]]:
    """Collect objects that may carry source_ids lists."""
    holders: list[tuple[str, list[Any]]] = []
    for kind in ("nodes", "edges", "alternatives", "facts"):
        items = payload.get(kind)
        if isinstance(items, list):
            holders.append((kind, items))
    return holders


def _collect_semantic_warnings(
    nodes_by_id: dict[str, dict[str, Any]],
    outgoing: dict[str, list[tuple[str, str]]],
    incoming: dict[str, list[str]],
) -> list[str]:
    """Return non-fatal notes for reachability, cycles, and disconnected parts."""
    warnings: list[str] = []
    if not nodes_by_id:
        return warnings

    adjacency = {node_id: [target for target, _ in targets] for node_id, targets in outgoing.items()}
    starts = [node_id for node_id, node in nodes_by_id.items() if node.get("node_type") == "start"]
    if not starts:
        starts = [node_id for node_id in nodes_by_id if not incoming.get(node_id)]
    if not starts:
        starts = list(nodes_by_id)

    reachable: set[str] = set()
    stack = list(starts)
    while stack:
        current = stack.pop()
        if current in reachable:
            continue
        reachable.add(current)
        stack.extend(adjacency.get(current, []))
    unreachable = [node_id for node_id in nodes_by_id if node_id not in reachable]
    if unreachable:
        warnings.append("Unreachable nodes: " + ", ".join(unreachable) + ".")

    visiting: set[str] = set()
    visited: set[str] = set()
    cycle_found = False

    def _visit(node_id: str) -> None:
        nonlocal cycle_found
        if cycle_found or node_id in visited:
            return
        visiting.add(node_id)
        for target in adjacency.get(node_id, []):
            if target in visiting:
                cycle_found = True
                return
            _visit(target)
        visiting.discard(node_id)
        visited.add(node_id)

    for node_id in nodes_by_id:
        _visit(node_id)
        if cycle_found:
            warnings.append("Graph contains a cycle; confirm it is an evidenced rework loop.")
            break

    undirected: dict[str, set[str]] = {node_id: set() for node_id in nodes_by_id}
    for source, targets in adjacency.items():
        for target in targets:
            undirected.setdefault(source, set()).add(target)
            undirected.setdefault(target, set()).add(source)
    seen_components: set[str] = set()
    component_count = 0
    for node_id in nodes_by_id:
        if node_id in seen_components:
            continue
        component_count += 1
        queue = [node_id]
        while queue:
            current = queue.pop()
            if current in seen_components:
                continue
            seen_components.add(current)
            queue.extend(undirected.get(current, ()))
    if component_count > 1:
        warnings.append(
            f"Graph has {component_count} disconnected components; confirm they are separate processes."
        )
    return warnings


def validate_workflow_graph(
    payload: Any,
    known_source_ids: set[str] | frozenset[str] | list[str] | None = None,
) -> dict[str, Any]:
    """Return the payload after structural checks, or raise on unsafe graph structure."""
    if not isinstance(payload, dict):
        raise ValueError("Workflow response must be an object.")
    for key in ("nodes", "edges", "lanes", "confirmations"):
        if not isinstance(payload.get(key), list):
            raise ValueError(f"Workflow {key} must be a list.")

    lanes = payload["lanes"]
    if any(not isinstance(lane, str) or not lane.strip() for lane in lanes):
        raise ValueError("Every lane must be a nonempty string.")
    if len(lanes) != len(set(lanes)):
        raise ValueError("Workflow lanes must be unique.")

    nodes_by_id: dict[str, dict[str, Any]] = {}
    for node in payload["nodes"]:
        if not isinstance(node, dict):
            raise ValueError("Every node must be an object.")
        node_id = node.get("id")
        if not isinstance(node_id, str) or not _NODE_ID_RE.fullmatch(node_id):
            raise ValueError("Node IDs must use N1, N2, ... notation.")
        if node_id in nodes_by_id:
            raise ValueError(f"Duplicate node ID: {node_id}.")
        node_type = node.get("node_type")
        if node_type not in _ALLOWED_NODE_TYPES:
            raise ValueError(f"Invalid node type for {node_id}.")
        label = node.get("label")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"Missing label for {node_id}.")
        if node.get("lane") not in lanes:
            raise ValueError(f"Undeclared lane for {node_id}.")
        status = node.get("status")
        if status is not None and status not in _ALLOWED_STATUS:
            raise ValueError(f"Invalid status for {node_id}.")
        nodes_by_id[node_id] = node

    outgoing: dict[str, list[tuple[str, str]]] = {node_id: [] for node_id in nodes_by_id}
    incoming: dict[str, list[str]] = {node_id: [] for node_id in nodes_by_id}
    seen_edges: set[tuple[str, str, str]] = set()
    for edge in payload["edges"]:
        if not isinstance(edge, dict):
            raise ValueError("Every edge must be an object.")
        source, target = edge.get("source"), edge.get("target")
        if not isinstance(source, str) or not isinstance(target, str):
            raise ValueError("Edge endpoints must be node ID strings.")
        if source not in nodes_by_id or target not in nodes_by_id:
            raise ValueError(f"Unknown edge endpoint: {source} -> {target}.")
        label = edge.get("label")
        if label is not None and not isinstance(label, str):
            raise ValueError("Edge labels must be strings or null.")
        condition = (label or "").strip().casefold()
        edge_key = (source, target, condition)
        if edge_key in seen_edges:
            raise ValueError(f"Duplicate edge: {source} -> {target}.")
        seen_edges.add(edge_key)
        outgoing[source].append((target, condition))
        incoming[target].append(source)
        status = edge.get("status")
        if status is not None and status not in _ALLOWED_STATUS:
            raise ValueError(f"Invalid status for edge {source} -> {target}.")

    for node_id, node in nodes_by_id.items():
        if node["node_type"] != "decision":
            continue
        branches = outgoing[node_id]
        targets = {target for target, _ in branches}
        conditions = [condition for _, condition in branches]
        if len(targets) < 2:
            raise ValueError(f"Decision {node_id} needs distinct next steps.")
        if not all(conditions) or len(set(conditions)) != len(conditions):
            raise ValueError(f"Decision {node_id} needs unique branch labels.")

    if known_source_ids is not None:
        allowed = set(known_source_ids)
        for kind, items in _iter_source_id_holders(payload):
            for item in items:
                if not isinstance(item, dict):
                    continue
                source_ids = item.get("source_ids")
                if source_ids is None:
                    continue
                if not isinstance(source_ids, list):
                    raise ValueError(f"Workflow {kind} source_ids must be a list.")
                for source_id in source_ids:
                    if source_id not in allowed:
                        raise ValueError(f"Unknown source_id {source_id!r}.")

    payload["validation_warnings"] = _collect_semantic_warnings(
        nodes_by_id, outgoing, incoming
    )
    return payload
