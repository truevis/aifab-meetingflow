"""Generate GitHub/Streamlit compatible Mermaid Markdown flowcharts with swimlanes."""

from __future__ import annotations

import re
from typing import Any


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


def build_mermaid_flowchart(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    lanes: list[str] | None = None,
    direction: str = "TD",
) -> str:
    """Build a valid Mermaid markdown flowchart code block with optional swimlane subgraphs."""
    chart_direction = _normalize_flowchart_direction(direction)
    lines: list[str] = [
        _mermaid_init_directive(),
        f"flowchart {chart_direction}",
    ]

    # Determine unique lanes if not specified
    if lanes is None:
        lane_order = []
        seen = set()
        for node in nodes:
            lane = node.get("lane", "General")
            if lane and lane not in seen:
                seen.add(lane)
                lane_order.append(lane)
        lanes = lane_order if lane_order else ["General"]

    # Group nodes by lane
    nodes_by_lane: dict[str, list[dict[str, Any]]] = {lane: [] for lane in lanes}
    other_nodes: list[dict[str, Any]] = []

    for node in nodes:
        lane = node.get("lane", "General")
        if lane in nodes_by_lane:
            nodes_by_lane[lane].append(node)
        else:
            other_nodes.append(node)

    # Render subgraphs for swimlanes
    subgraph_index = 1
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

    # Render any unassigned nodes
    if other_nodes:
        for node in other_nodes:
            lines.append(f"    {_format_node_str(node)}")

    # Render edges
    lines.append("")
    for edge in edges:
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
