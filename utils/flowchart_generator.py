"""Generate GitHub/Streamlit compatible Mermaid Markdown flowcharts with swimlanes."""

from __future__ import annotations

from typing import Any

SIDE_BLOB_TITLE = "Discussed, not a sequence"
SIDE_BLOB_CAP = 12


def sanitize_mermaid_label(text: str) -> str:
    """Sanitize labels to prevent Mermaid syntax breakage and escape quotes."""
    safe = str(text).replace('"', "'").replace("\n", " ").strip()
    # Replace LaTeX '$' with fullwidth '＄'
    safe = safe.replace("$", "＄")
    # Avoid unmatched brackets or parens breaking node definition
    safe = safe.replace("[", "［").replace("]", "］")
    safe = safe.replace("(", "（").replace(")", "）")
    safe = safe.replace("{", "｛").replace("}", "｝")
    safe = safe.replace("<", "＜").replace(">", "＞")
    return safe


def _wrap_mermaid_label(text: str, width: int = 26) -> str:
    """Insert HTML line breaks so long node labels stay readable in the chart."""
    words = [word for word in text.split() if word]
    if not words:
        return text
    lines: list[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        if current and len(trial) > width:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return "<br>".join(lines)


def _normalize_flowchart_direction(direction: str) -> str:
    """Return a valid Mermaid flowchart direction, defaulting to top-down."""
    value = str(direction).strip().upper()
    if value in {"TD", "TB", "BT", "LR", "RL"}:
        return value
    return "TD"


def _mermaid_init_directive(font_size: str = "22px") -> str:
    """Return a Mermaid init line that enlarges labels without invalidating exported markdown."""
    return (
        "%%{init: {\"theme\": \"base\", "
        "\"themeVariables\": {\"fontSize\": \""
        + font_size
        + "\"}, "
        "\"flowchart\": {\"useMaxWidth\": true, \"htmlLabels\": true, "
        "\"nodeSpacing\": 56, \"rankSpacing\": 72, \"padding\": 24}}}%%"
    )


def _node_lookup(nodes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index workflow nodes by id string."""
    lookup: dict[str, dict[str, Any]] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id") or "").strip()
        if node_id:
            lookup[node_id] = node
    return lookup


def _undirected_adjacency(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> dict[str, set[str]]:
    """Build an undirected adjacency map from directed workflow edges."""
    adjacency: dict[str, set[str]] = {
        str(node.get("id") or ""): set()
        for node in nodes
        if isinstance(node, dict) and str(node.get("id") or "").strip()
    }
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        source = str(edge.get("source") or "").strip()
        target = str(edge.get("target") or "").strip()
        if source not in adjacency or target not in adjacency:
            continue
        adjacency[source].add(target)
        adjacency[target].add(source)
    return adjacency


def weakly_connected_components(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Return weakly connected node groups in first-seen node order."""
    lookup = _node_lookup(nodes)
    if not lookup:
        return []
    adjacency = _undirected_adjacency(nodes, edges)
    seen: set[str] = set()
    components: list[list[dict[str, Any]]] = []
    for node_id in lookup:
        if node_id in seen:
            continue
        queue = [node_id]
        member_ids: list[str] = []
        while queue:
            current = queue.pop()
            if current in seen or current not in lookup:
                continue
            seen.add(current)
            member_ids.append(current)
            queue.extend(sorted(adjacency.get(current, ())))
        components.append([lookup[member_id] for member_id in member_ids])
    return components


def _component_caption(component_nodes: list[dict[str, Any]]) -> str:
    """Build a caption from the component's own labels, not a made-up topic."""
    labels: list[str] = []
    for node in component_nodes:
        label = str(node.get("label") or "").strip()
        node_type = str(node.get("node_type") or "")
        if not label:
            continue
        if node_type in {"start", "end"} and len(component_nodes) > 1:
            continue
        labels.append(label)
        if len(labels) >= 3:
            break
    if not labels:
        for node in component_nodes:
            label = str(node.get("label") or "").strip()
            if label:
                labels.append(label)
                break
    caption = "; ".join(labels) if labels else "Process"
    if len(caption) > 72:
        caption = caption[:69].rstrip() + "..."
    return sanitize_mermaid_label(caption)


def _lane_order_for_nodes(
    nodes: list[dict[str, Any]],
    lanes: list[str] | None,
) -> list[str]:
    """Return lane order preferring caller order, then first appearance."""
    if lanes:
        return list(lanes)
    order: list[str] = []
    seen: set[str] = set()
    for node in nodes:
        lane = str(node.get("lane") or "General")
        if lane and lane not in seen:
            seen.add(lane)
            order.append(lane)
    return order if order else ["General"]


def _append_lane_subgraphs(
    lines: list[str],
    nodes: list[dict[str, Any]],
    lanes: list[str] | None,
    chart_direction: str,
    subgraph_index: int,
) -> int:
    """Append swimlane subgraphs for the given nodes; return next subgraph index."""
    lane_order = _lane_order_for_nodes(nodes, lanes)
    nodes_by_lane: dict[str, list[dict[str, Any]]] = {lane: [] for lane in lane_order}
    other_nodes: list[dict[str, Any]] = []
    for node in nodes:
        lane = str(node.get("lane") or "General")
        if lane in nodes_by_lane:
            nodes_by_lane[lane].append(node)
        else:
            other_nodes.append(node)
    for lane, lane_nodes in nodes_by_lane.items():
        if not lane_nodes:
            continue
        safe_lane = sanitize_mermaid_label(lane)
        lines.append(f'    subgraph SG{subgraph_index} ["{safe_lane}"]')
        lines.append(f"        direction {chart_direction}")
        for node in lane_nodes:
            lines.append(f"        {_format_node_str(node)}")
        lines.append("    end")
        subgraph_index += 1
    for node in other_nodes:
        lines.append(f"    {_format_node_str(node)}")
    return subgraph_index


def _append_component_subgraphs(
    lines: list[str],
    components: list[list[dict[str, Any]]],
    chart_direction: str,
    subgraph_index: int,
) -> int:
    """Append one subgraph per weakly connected component."""
    for component in components:
        if not component:
            continue
        caption = _component_caption(component)
        lines.append(f'    subgraph SG{subgraph_index} ["{caption}"]')
        lines.append(f"        direction {chart_direction}")
        for node in component:
            lines.append(f"        {_format_node_str(node)}")
        lines.append("    end")
        subgraph_index += 1
    return subgraph_index


def _side_note_label(note: Any) -> str:
    """Extract a display label from a side-note string or dict."""
    if isinstance(note, str):
        return note.strip()
    if not isinstance(note, dict):
        return ""
    return str(note.get("label") or note.get("text") or "").strip()


def _append_side_blob_subgraph(
    lines: list[str],
    side_notes: list[Any] | None,
    chart_direction: str,
    subgraph_index: int,
) -> int:
    """Append the non-sequence side notes subgraph using S1, S2, ... ids."""
    notes = [
        note
        for note in (side_notes or [])
        if _side_note_label(note)
    ][:SIDE_BLOB_CAP]
    if not notes:
        return subgraph_index
    title = sanitize_mermaid_label(SIDE_BLOB_TITLE)
    lines.append(f'    subgraph SG{subgraph_index} ["{title}"]')
    lines.append(f"        direction {chart_direction}")
    for index, note in enumerate(notes, start=1):
        label = _wrap_mermaid_label(sanitize_mermaid_label(_side_note_label(note)))
        lines.append(f'        S{index}["{label}"]')
    lines.append("    end")
    return subgraph_index + 1


def build_mermaid_flowchart(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    lanes: list[str] | None = None,
    direction: str = "TD",
    side_notes: list[Any] | None = None,
) -> str:
    """Build a Mermaid flowchart with lanes or components plus a side notes blob."""
    chart_direction = _normalize_flowchart_direction(direction)
    lines: list[str] = [
        _mermaid_init_directive(),
        f"flowchart {chart_direction}",
    ]

    valid_nodes = [node for node in nodes if isinstance(node, dict) and node.get("id")]
    components = weakly_connected_components(valid_nodes, edges)
    subgraph_index = 1
    if len(components) > 1:
        subgraph_index = _append_component_subgraphs(
            lines, components, chart_direction, subgraph_index
        )
    elif valid_nodes:
        subgraph_index = _append_lane_subgraphs(
            lines, valid_nodes, lanes, chart_direction, subgraph_index
        )

    lines.append("")
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        source_id = edge.get("source")
        target_id = edge.get("target")
        label = edge.get("label")
        if not source_id or not target_id:
            continue
        if label:
            safe_edge_label = sanitize_mermaid_label(label)
            lines.append(f'    {source_id} -->|"{safe_edge_label}"| {target_id}')
        else:
            lines.append(f"    {source_id} --> {target_id}")

    _append_side_blob_subgraph(lines, side_notes, chart_direction, subgraph_index)

    return "\n".join(lines)


def _format_node_str(node: dict[str, Any]) -> str:
    """Format single node into Mermaid shape notation."""
    node_id = node.get("id", "N")
    label = _wrap_mermaid_label(sanitize_mermaid_label(node.get("label", "")))
    node_type = node.get("node_type", "action")

    if node_type == "start" or node_type == "end":
        return f'{node_id}(["{label}"])'
    elif node_type == "decision":
        return f'{node_id}{{"{label}"}}'
    else:  # action / standard box
        return f'{node_id}["{label}"]'
