"""Streamlit interface for Meeting Flow Live: Meeting Video/Audio to Markdown Flowchart."""

from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path
from typing import Any

import streamlit as st

from utils.flowchart_generator import build_mermaid_flowchart
from utils.meeting_analyzer import (
    DEFAULT_MODEL,
    GEMINI_FLASH_MODEL,
    GLM_FLASH_LATEST_MODEL,
    JEV_MODEL,
    analyze_transcript_with_gemini,
    analyze_transcript_with_glm,
    analyze_transcript_with_jev,
    analyze_transcript_with_mercury,
    get_default_vl_model,
    get_sample_meeting_data,
    get_vl_model_label,
    get_vl_transcription_models,
    is_ling_vl_model,
    iter_media_transcript_chunks,
    normalize_pasted_transcript,
    transcribe_media_with_openrouter,
    workflow_cache_fingerprint,
)
from utils.workflow_validator import validate_workflow_graph

ENGINE_JEV = "jev"
ENGINE_MERCURY = "mercury"
ENGINE_GLM = "glm"
ENGINE_GEMINI = "gemini"
FLOWCHART_ENGINES = (ENGINE_JEV, ENGINE_MERCURY, ENGINE_GLM, ENGINE_GEMINI)
ENGINE_LABELS = {
    ENGINE_JEV: "Jev hybrid",
    ENGINE_MERCURY: "Mercury",
    ENGINE_GLM: "GLM Flash latest",
    ENGINE_GEMINI: "Gemini 3.8 Flash",
}
DEFAULT_FLOWCHART_ENGINE = ENGINE_MERCURY
SOURCE_UPLOAD = "Video or audio"
SOURCE_PASTE = "Paste transcript"
MEDIA_FILE_EXTENSIONS = ("mp4", "mov", "m4a", "mp3", "wav")
TRANSCRIPT_FILE_EXTENSIONS = ("txt", "md")


TRANSCRIPT_BOX_HEIGHT = 180
TRANSCRIPT_STREAM_PAINT_SECONDS = 0.1


def get_openrouter_api_key() -> str:
    """Read the OpenRouter API key from Streamlit secrets or environment."""
    try:
        secret_key = st.secrets.get("OPENROUTER_API_KEY", "")
    except Exception:
        secret_key = ""
    return str(secret_key or os.getenv("OPENROUTER_API_KEY", ""))


def _display_text(value: Any) -> str:
    """Make dynamic text safe for Streamlit's Markdown renderer and avoid LaTeX $ syntax."""
    return str(value).replace("$", "＄")


def _is_non_english(data: dict[str, Any]) -> bool:
    """Check if the current meeting contains non-English text that needs translation."""
    utterances = data.get("utterances", [])
    lanes = data.get("lanes", [])
    sample_text = " ".join([u.get("text", "") for u in utterances] + lanes)
    # Check for presence of Japanese/CJK characters
    return any(ord(char) > 0x2E80 for char in sample_text)


def _transcript_from_meeting(meeting_data: dict[str, Any]) -> str:
    """Build transcript textarea text from cached meeting utterances."""
    return "\n".join(
        f"[{u['timestamp']}] {u['speaker']}: {u['text']}"
        for u in meeting_data.get("utterances", [])
    )


def _empty_meeting_data() -> dict[str, Any]:
    """Return an empty meeting payload for an engine that has not been generated yet."""
    return {
        "utterances": [],
        "nodes": [],
        "edges": [],
        "lanes": [],
        "confirmations": [],
        "alternatives": [],
        "cache_fingerprint": "",
    }


def _engine_model_id(engine: str) -> str:
    """Return the model identity stored in a workflow cache fingerprint."""
    if engine == ENGINE_JEV:
        return JEV_MODEL
    if engine == ENGINE_GLM:
        return GLM_FLASH_LATEST_MODEL
    if engine == ENGINE_GEMINI:
        return GEMINI_FLASH_MODEL
    return DEFAULT_MODEL


def _expected_cache_fingerprint(engine: str) -> str:
    """Fingerprint the current transcript, engine/model, translation, and schema revision."""
    return workflow_cache_fingerprint(
        str(st.session_state.get("transcript_input") or ""),
        engine,
        _engine_model_id(engine),
        bool(st.session_state.get("translate_to_english")),
    )


def _engine_display_name(engine: str) -> str:
    """Return the human-readable name of the selected flowchart engine."""
    return ENGINE_LABELS.get(engine, engine)


def _engine_has_result(engine: str) -> bool:
    """Return whether the selected engine has a cached flowchart for the current input."""
    if not st.session_state.get("engine_has_result", {}).get(engine):
        return False
    meeting = st.session_state.get("meeting_by_engine", {}).get(engine) or {}
    stored = str(meeting.get("cache_fingerprint") or "")
    if not stored:
        return False
    return stored == _expected_cache_fingerprint(engine)


def _meeting_for_display(engine: str) -> dict[str, Any]:
    """Return cached meeting data only when its fingerprint matches the current input."""
    if _engine_has_result(engine):
        return st.session_state.meeting_by_engine.get(engine) or _empty_meeting_data()
    return _empty_meeting_data()


def _empty_engine_flags() -> dict[str, bool]:
    """Return a False flag for every flowchart engine."""
    return {engine: False for engine in FLOWCHART_ENGINES}


def _empty_engine_meetings() -> dict[str, dict[str, Any]]:
    """Return an empty meeting payload for every flowchart engine."""
    return {engine: _empty_meeting_data() for engine in FLOWCHART_ENGINES}


def _ensure_engine_slots() -> None:
    """Ensure every flowchart engine has a meeting cache and result flag."""
    meetings = st.session_state.setdefault("meeting_by_engine", {})
    results = st.session_state.setdefault("engine_has_result", {})
    for engine in FLOWCHART_ENGINES:
        meetings.setdefault(engine, _empty_meeting_data())
        results.setdefault(engine, False)


def _selected_flowchart_engine() -> str:
    """Return the currently selected flowchart engine from session state."""
    engine = st.session_state.get("flowchart_engine", DEFAULT_FLOWCHART_ENGINE)
    if engine in FLOWCHART_ENGINES:
        return engine
    return DEFAULT_FLOWCHART_ENGINE


def _reset_to_sample_meeting() -> None:
    """Load the English sample into every engine cache and the displayed meeting."""
    sample = get_sample_meeting_data()
    st.session_state.meeting_by_engine = {
        engine: copy.deepcopy(sample) for engine in FLOWCHART_ENGINES
    }
    st.session_state.engine_has_result = {engine: True for engine in FLOWCHART_ENGINES}
    st.session_state.engine_errors = {}
    engine = _selected_flowchart_engine()
    st.session_state.current_meeting = st.session_state.meeting_by_engine[engine]
    st.session_state.transcript_input = _transcript_from_meeting(st.session_state.current_meeting)
    _bump_transcript_box()
    translate_to_english = bool(st.session_state.get("translate_to_english"))
    for sample_engine, meeting in st.session_state.meeting_by_engine.items():
        meeting["alternatives"] = list(meeting.get("alternatives") or [])
        meeting["cache_fingerprint"] = workflow_cache_fingerprint(
            st.session_state.transcript_input,
            sample_engine,
            _engine_model_id(sample_engine),
            translate_to_english,
        )


def _sync_current_meeting_from_engine(engine: str) -> None:
    """Point the displayed meeting at the cached payload for the selected engine."""
    caches = st.session_state.get("meeting_by_engine", {})
    if engine in caches:
        st.session_state.current_meeting = caches[engine]


def _initialize_state() -> None:
    """Initialize session state variables."""
    if "flowchart_engine" not in st.session_state:
        st.session_state.flowchart_engine = DEFAULT_FLOWCHART_ENGINE
    if "translate_to_english" not in st.session_state:
        st.session_state.translate_to_english = False
    if "engine_errors" not in st.session_state:
        st.session_state.engine_errors = {}
    if "engine_has_result" not in st.session_state:
        st.session_state.engine_has_result = _empty_engine_flags()
    if "meeting_by_engine" not in st.session_state:
        st.session_state.meeting_by_engine = _empty_engine_meetings()
    _ensure_engine_slots()
    if "current_meeting" not in st.session_state:
        engine = _selected_flowchart_engine()
        st.session_state.current_meeting = st.session_state.meeting_by_engine.get(
            engine, _empty_meeting_data()
        )
    if "is_analyzing" not in st.session_state:
        st.session_state.is_analyzing = False
    if "transcript_input" not in st.session_state:
        st.session_state.transcript_input = _transcript_from_meeting(st.session_state.current_meeting)
    if "imported_transcript_id" not in st.session_state:
        st.session_state.imported_transcript_id = None
    if "transcript_box_nonce" not in st.session_state:
        st.session_state.transcript_box_nonce = 0


def _render_header() -> None:
    """Render top brand and high-level summary metrics."""
    st.title("🎯 Meeting Flow Live")
    st.caption(
        "Automatically extract and generate interactive Markdown flowcharts (Mermaid format) "
        "from meeting audio, video, or transcripts. Filters out chit-chat, assigns department swimlanes, "
        "tracks decision branches, and flags ambiguous action items in American English."
    )


def _render_summary_metrics(meeting_data: dict[str, Any]) -> None:
    """Render high-level metrics for steps, lanes, and confirmation items."""
    nodes = meeting_data.get("nodes", [])
    lanes = meeting_data.get("lanes", [])
    confirmations = meeting_data.get("confirmations", [])

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Steps", len(nodes))
    col2.metric("Identified Swimlanes", len(lanes))
    col3.metric("Action Items to Confirm", len(confirmations))


def _render_confirmation_alert(confirmations: list[dict[str, Any]]) -> None:
    """Render prominent alert for ambiguous items that need clarification."""
    if not confirmations:
        return

    for conf in confirmations:
        question = _display_text(conf.get("question", ""))
        suggested = _display_text(conf.get("suggested_role", "Unassigned"))
        confidence = conf.get("confidence", 0.0)
        st.warning(
            f"⚠️ **Confirmation Item ({conf.get('id', 1)})**: {question}\n\n"
            f"• Candidate Role: `{suggested}` (Confidence: {confidence:.0%})"
        )


def _render_utterance_card(item: dict[str, Any]) -> None:
    """Render a single conversation turn in the live transcript feed."""
    timestamp = item.get("timestamp", "00:00")
    speaker = item.get("speaker", "Speaker")
    text = _display_text(item.get("text", ""))
    is_biz = item.get("is_business", True)
    relevance = item.get("relevance_score", 0.0)
    action_type = item.get("action_type", "none")
    detected_step = item.get("detected_step")

    badge_type = "Business" if is_biz else "Chit-chat"
    badge_pct = f"{relevance:.0%}"

    with st.container():
        st.markdown(f"**[{timestamp}] {speaker}**: {text}")
        if is_biz:
            detail = f"💡 `{badge_type} {badge_pct}` · Action: `{action_type}`"
            if detected_step:
                detail += f" · Step: **{_display_text(detected_step)}**"
            st.caption(detail)
        else:
            st.caption(f"☕ `{badge_type} (Filtered)` · Not in workflow")


def _render_transcript_feed(utterances: list[dict[str, Any]]) -> None:
    """Render full scrollable timeline of meeting utterances."""
    st.subheader("🎙️ Live Transcript & Utterance Log")
    for item in utterances:
        _render_utterance_card(item)


def _flowchart_table_rows(nodes: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Build a column-oriented table so st.dataframe stays valid when nodes are empty."""
    return {
        "Step ID": [n.get("id") for n in nodes],
        "Swimlane (Role)": [n.get("lane") for n in nodes],
        "Type": [n.get("node_type") for n in nodes],
        "Action Description": [n.get("label") for n in nodes],
    }


def _mermaid_for_display(mermaid_code: str) -> str:
    """Replace '$' for Streamlit display only so labels are not treated as LaTeX."""
    return mermaid_code.replace("$", "＄")


def _render_flowchart_view(meeting_data: dict[str, Any]) -> None:
    """Render the Markdown Mermaid flowchart and raw code copy."""
    nodes = meeting_data.get("nodes", [])
    edges = meeting_data.get("edges", [])
    lanes = meeting_data.get("lanes", [])
    empty_caption = (
        "No flowchart yet. Upload a meeting, import a transcript file, "
        "paste a transcript, or load the sample."
    )
    if _engine_has_result(_selected_flowchart_engine()) and not nodes:
        empty_caption = "No supported workflow found"

    tab1, tab2, tab3 = st.tabs(["📊 Flowchart Diagram", "📋 Step Table", "📝 Markdown Source"])

    with tab1:
        alternatives = meeting_data.get("alternatives") or []
        if alternatives:
            labels = []
            for item in alternatives:
                if isinstance(item, str) and item.strip():
                    labels.append(item.strip())
                elif isinstance(item, dict):
                    label = str(item.get("label") or item.get("text") or "").strip()
                    if label:
                        labels.append(label)
            if labels:
                st.info(
                    "Alternatives not on the main path: "
                    + "; ".join(_display_text(label) for label in labels)
                )
        if not nodes:
            st.caption(empty_caption)
        else:
            mermaid_code = build_mermaid_flowchart(nodes, edges, lanes=lanes, direction="TD")
            st.caption("Markdown Mermaid Flowchart Preview:")
            st.mermaid_chart(_mermaid_for_display(mermaid_code), width="stretch")

    with tab2:
        st.caption("Flowchart nodes and assigned department swimlanes:")
        st.dataframe(_flowchart_table_rows(nodes), width="stretch", hide_index=True)

    with tab3:
        if not nodes:
            st.caption(empty_caption)
        else:
            mermaid_code = build_mermaid_flowchart(nodes, edges, lanes=lanes, direction="TD")
            st.caption("Copyable Markdown Mermaid code:")
            st.code(mermaid_code, language="mermaid")
            st.download_button(
                "Download Flowchart Markdown",
                data=mermaid_code,
                file_name="meeting_flowchart.md",
                mime="text/markdown",
            )


def _render_sidebar() -> tuple[str, str, bool, bool]:
    """Render sidebar decision controls and return VL model, engine, translate, and create-click."""
    models = get_vl_transcription_models()
    model_ids = [model["id"] for model in models]
    labels_by_id = {model["id"]: model["label"] for model in models}
    default_model = get_default_vl_model()
    default_index = model_ids.index(default_model) if default_model in model_ids else 0

    selected_model = st.sidebar.selectbox(
        "Video-to-text model",
        options=model_ids,
        index=default_index,
        format_func=lambda model_id: labels_by_id.get(model_id, model_id),
        help="Used to transcribe uploaded meeting video or audio before the selected flowchart model runs.",
    )
    if is_ling_vl_model(selected_model):
        st.sidebar.caption(
            "Ling native limits: one video per request, duration about 30s, Base64 payload up to 32 MB."
        )

    translate_to_english = st.sidebar.toggle(
        "Translate meeting to English (if source audio/transcript is non-English)",
        key="translate_to_english",
        help="Turn on if the recorded meeting is in another language (e.g. Japanese, Spanish) and needs translation to English.",
    )
    engine = st.sidebar.radio(
        "Flowchart model",
        options=list(FLOWCHART_ENGINES),
        format_func=_engine_display_name,
        key="flowchart_engine",
        help="Used to extract a Mermaid flowchart from the meeting transcript. Jev hybrid: Mercury synthesizes the graph and Jev checks supported relationships. Switching shows that model's cached flowchart if one exists.",
    )
    engine_label = _engine_display_name(engine)
    create_clicked = st.sidebar.button(
        f"Create flowchart with {engine_label}",
        type="primary",
        use_container_width=True,
        help="Runs only the currently selected flowchart model on the meeting transcript.",
    )
    if not _engine_has_result(engine):
        st.sidebar.caption(
            f"No {engine_label} flowchart yet. Click the button above to generate it."
        )
    return selected_model, engine, translate_to_english, create_clicked


def _read_uploaded_media(uploaded_file: Any) -> tuple[bytes, str]:
    """Read bytes and filename from a Streamlit uploaded media file."""
    return uploaded_file.getvalue(), uploaded_file.name


def _file_extension(filename: str) -> str:
    """Return a lowercase file extension without the leading dot."""
    return Path(str(filename or "")).suffix.lower().lstrip(".")


def _is_transcript_file(uploaded_file: Any | None) -> bool:
    """Return whether an uploaded file is a text or Markdown transcript."""
    if uploaded_file is None:
        return False
    return _file_extension(getattr(uploaded_file, "name", "")) in TRANSCRIPT_FILE_EXTENSIONS


def _decode_uploaded_text(file_bytes: bytes) -> str:
    """Decode transcript file bytes, preferring UTF-8 with an optional BOM."""
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("utf-8", errors="replace")


def _read_uploaded_transcript(uploaded_file: Any) -> str:
    """Read and normalize a TXT or Markdown transcript upload."""
    return normalize_pasted_transcript(_decode_uploaded_text(uploaded_file.getvalue()).strip())


def _imported_file_identity(uploaded_file: Any) -> tuple[str, int]:
    """Identify an imported transcript so later edits are not overwritten."""
    return (str(uploaded_file.name), int(uploaded_file.size))


def _bump_transcript_box() -> None:
    """Remount the Meeting Transcript box so a newly loaded value is shown."""
    st.session_state.transcript_box_nonce = int(st.session_state.get("transcript_box_nonce") or 0) + 1


def _apply_imported_transcript_file(uploaded_file: Any) -> None:
    """Load a newly selected transcript file into the Meeting Transcript box."""
    identity = _imported_file_identity(uploaded_file)
    if st.session_state.get("imported_transcript_id") == identity:
        return
    st.session_state.transcript_input = _read_uploaded_transcript(uploaded_file)
    st.session_state.imported_transcript_id = identity
    _bump_transcript_box()


def _analyze_with_selected_engine(
    transcript_text: str,
    api_key: str,
    translate_to_english: bool,
    engine: str,
) -> dict[str, Any]:
    """Run workflow extraction with only the currently selected flowchart engine."""
    if engine == ENGINE_JEV:
        return analyze_transcript_with_jev(
            transcript_text,
            api_key=api_key,
            translate_to_english=translate_to_english,
        )
    if engine == ENGINE_GLM:
        return analyze_transcript_with_glm(
            transcript_text,
            api_key=api_key,
            translate_to_english=translate_to_english,
        )
    if engine == ENGINE_GEMINI:
        return analyze_transcript_with_gemini(
            transcript_text,
            api_key=api_key,
            translate_to_english=translate_to_english,
        )
    return analyze_transcript_with_mercury(
        transcript_text,
        api_key=api_key,
        translate_to_english=translate_to_english,
    )


def _store_engine_analysis(
    engine: str,
    analyzed: dict[str, Any],
    translate_to_english: bool,
    transcript_text: str = "",
) -> None:
    """Cache a successful analysis for one engine and display it."""
    meeting = copy.deepcopy(st.session_state.meeting_by_engine.get(engine) or _empty_meeting_data())
    meeting["nodes"] = analyzed.get("nodes", [])
    meeting["edges"] = analyzed.get("edges", [])
    meeting["lanes"] = analyzed.get("lanes", [])
    meeting["confirmations"] = analyzed.get("confirmations", [])
    meeting["alternatives"] = analyzed.get("alternatives", [])
    meeting["validation_warnings"] = analyzed.get("validation_warnings", [])
    if analyzed.get("utterances"):
        meeting["utterances"] = analyzed["utterances"]
    displayed_transcript = transcript_text
    if translate_to_english and analyzed.get("translated_transcript"):
        displayed_transcript = str(analyzed["translated_transcript"])
        st.session_state.transcript_input = displayed_transcript
    meeting["cache_fingerprint"] = workflow_cache_fingerprint(
        displayed_transcript,
        engine,
        _engine_model_id(engine),
        translate_to_english,
    )
    st.session_state.meeting_by_engine[engine] = meeting
    st.session_state.engine_has_result[engine] = True
    st.session_state.engine_errors[engine] = None
    st.session_state.current_meeting = meeting


def _parse_embedded_json(text: str) -> Any:
    """Parse the first JSON object embedded in a string, if present."""
    start = str(text).find("{")
    if start < 0:
        return None
    try:
        payload, _end = json.JSONDecoder().raw_decode(str(text)[start:])
    except ValueError:
        return None
    return payload


def _nested_provider_message(error_obj: dict[str, Any]) -> str:
    """Read a provider's inner error message from OpenRouter metadata."""
    metadata = error_obj.get("metadata")
    if not isinstance(metadata, dict):
        return ""
    raw = metadata.get("raw")
    if isinstance(raw, dict):
        return str(raw.get("message") or raw.get("reason") or "").strip()
    if isinstance(raw, str):
        parsed = _parse_embedded_json(raw)
        if isinstance(parsed, dict):
            return str(parsed.get("message") or parsed.get("reason") or "").strip()
        return raw.strip()
    return ""


def _provider_error_message(error: BaseException) -> str:
    """Return the most specific OpenRouter/provider error text available."""
    parsed = _parse_embedded_json(str(error))
    if isinstance(parsed, dict):
        error_obj = parsed.get("error")
        if isinstance(error_obj, dict):
            nested = _nested_provider_message(error_obj)
            if nested:
                return nested
            message = str(error_obj.get("message") or "").strip()
            if message and message.lower() != "provider returned error":
                return message
        elif isinstance(error_obj, str) and error_obj.strip():
            return error_obj.strip()
    return str(error).strip()


def _friendly_transcription_reason(detail: str) -> str:
    """Rewrite common provider failures into a short user-facing reason."""
    lowered = detail.lower()
    if "too large" in lowered or "must be less than" in lowered:
        return "the uploaded file is too large for this model."
    if "timed out" in lowered or "timeout" in lowered:
        return "the model took too long to return a transcript. Long videos often succeed with another model."
    if "too long" in lowered or "duration" in lowered:
        return "the uploaded file is too long for this model."
    reason = detail.strip().rstrip(".")
    if not reason:
        return "the selected model could not transcribe this file."
    return f"{reason}."


def _other_vl_model_suggestion(current_model_id: str) -> str:
    """Name other sidebar video-to-text models the user can try."""
    labels = [
        model["label"]
        for model in get_vl_transcription_models()
        if model["id"] != current_model_id
    ]
    if not labels:
        return "a different Video-to-text model in the sidebar"
    if len(labels) == 1:
        example = labels[0]
    elif len(labels) == 2:
        example = f"{labels[0]} or {labels[1]}"
    else:
        example = f"{', '.join(labels[:-1])}, or {labels[-1]}"
    return f"a different Video-to-text model in the sidebar, such as {example}"


def _format_transcription_error(error: BaseException, vl_model: str) -> str:
    """Build a short transcription failure message that suggests another model."""
    model_label = get_vl_model_label(vl_model)
    reason = _friendly_transcription_reason(_provider_error_message(error))
    suggestion = _other_vl_model_suggestion(vl_model)
    return (
        f"Transcription with {model_label} failed: {reason} "
        f"Try {suggestion}."
    )


def _set_engine_error(engine: str, message: str) -> None:
    """Store and display an extract-flow error for the selected engine."""
    text = _display_text(message)
    st.session_state.engine_errors[engine] = text
    st.error(text)


def _transcript_box_help(paste_source: bool) -> str:
    """Return the Meeting Transcript input help text."""
    if paste_source:
        return (
            "Paste a full meeting transcript, or import a TXT or Markdown file. "
            "Timestamped captions, including YouTube-style lines such as "
            "'0:2222 seconds...', are accepted. Video-to-text extraction is skipped."
        )
    return (
        "Full meeting transcript. Uploaded video or audio streams into this box as it is transcribed. "
        "A TXT or Markdown transcript file loads here directly. You can also paste a transcript. "
        "The selected flowchart model runs only after this text is complete."
    )


def _show_transcript_box(
    slot: Any,
    text: str,
    *,
    disabled: bool,
    paste_source: bool = False,
    collapse_label: bool = True,
) -> str:
    """Render the Meeting Transcript text area into a placeholder slot."""
    return slot.text_area(
        "Meeting Transcript",
        value=text,
        height=TRANSCRIPT_BOX_HEIGHT,
        disabled=disabled,
        label_visibility="collapsed" if collapse_label else "visible",
        placeholder=(
            "Paste a meeting transcript here. YouTube-style timestamps are supported."
            if paste_source
            else ""
        ),
        help=_transcript_box_help(paste_source),
        key=f"transcript_box_{st.session_state.get('transcript_box_nonce', 0)}",
    )


def _show_streaming_transcript(slot: Any, text: str) -> None:
    """Show in-progress transcript text in the Meeting Transcript box."""
    with slot.container(height=TRANSCRIPT_BOX_HEIGHT, border=True):
        st.text(text or " ")


def _stream_transcript_into_box(slot: Any, chunks: Any) -> str:
    """Paint incoming transcript chunks into the Meeting Transcript box."""
    parts: list[str] = []
    last_paint = 0.0
    _show_streaming_transcript(
        slot,
        "Waiting for the video-to-text model to start returning the transcript...",
    )
    try:
        for chunk in chunks:
            if not chunk:
                continue
            parts.append(str(chunk))
            now = time.monotonic()
            if now - last_paint >= TRANSCRIPT_STREAM_PAINT_SECONDS:
                _show_streaming_transcript(slot, "".join(parts))
                last_paint = now
    except Exception:
        partial = "".join(parts).strip()
        if partial:
            st.session_state.transcript_input = partial
            _bump_transcript_box()
            _show_transcript_box(slot, partial, disabled=False)
        raise
    text = "".join(parts).strip()
    st.session_state.transcript_input = text
    _bump_transcript_box()
    _show_transcript_box(slot, text, disabled=False)
    return text


def _gather_meeting_transcript(
    transcript_text: str,
    api_key: str,
    uploaded_file: Any | None,
    vl_model: str,
    transcript_slot: Any | None = None,
) -> str:
    """Return the complete meeting transcript, transcribing media first when needed."""
    if _is_transcript_file(uploaded_file):
        gathered = _read_uploaded_transcript(uploaded_file)
        st.session_state.transcript_input = gathered
        return gathered
    if uploaded_file is not None:
        file_bytes, filename = _read_uploaded_media(uploaded_file)
        if transcript_slot is not None:
            gathered = _stream_transcript_into_box(
                transcript_slot,
                iter_media_transcript_chunks(
                    file_bytes,
                    filename,
                    api_key=api_key,
                    model=vl_model,
                ),
            )
        else:
            gathered = str(
                transcribe_media_with_openrouter(
                    file_bytes,
                    filename,
                    api_key=api_key,
                    model=vl_model,
                )
                or ""
            ).strip()
            st.session_state.transcript_input = gathered
        return gathered
    return normalize_pasted_transcript(str(transcript_text or "").strip())


def _handle_run_analysis(
    transcript_text: str,
    api_key: str,
    translate_to_english: bool,
    engine: str,
    uploaded_file: Any | None = None,
    vl_model: str = "",
    transcript_slot: Any | None = None,
) -> None:
    """Gather the full transcript first, then extract a flowchart with the selected engine."""
    if not api_key:
        st.error("Please configure your OpenRouter API key in .streamlit/secrets.toml or environment variables.")
        return

    engine_label = _engine_display_name(engine)
    selected_vl_model = vl_model or get_default_vl_model()
    vl_label = get_vl_model_label(selected_vl_model)
    gather_label = (
        f"Transcribing meeting with {vl_label}..."
        if uploaded_file is not None and not _is_transcript_file(uploaded_file)
        else "Gathering meeting transcript..."
    )

    with st.status(gather_label, expanded=True) as status:
        try:
            analysis_text = _gather_meeting_transcript(
                transcript_text,
                api_key=api_key,
                uploaded_file=uploaded_file,
                vl_model=selected_vl_model,
                transcript_slot=transcript_slot,
            )
        except Exception as error:
            status.update(label="Transcription failed", state="error")
            _set_engine_error(engine, _format_transcription_error(error, selected_vl_model))
            return

        if not analysis_text:
            status.update(label="No transcript to analyze", state="error")
            _set_engine_error(
                engine,
                f"Gather a complete meeting transcript before querying {engine_label}. "
                "Upload a video or audio file, import a TXT or Markdown transcript, "
                "or choose Paste transcript and paste the text, then extract the workflow.",
            )
            return

        extract_label = (
            f"Translating meeting to English and extracting workflow with {engine_label}..."
            if translate_to_english
            else f"Extracting workflow with {engine_label}..."
        )
        status.update(label=extract_label, state="running")
        try:
            analyzed = _analyze_with_selected_engine(
                analysis_text,
                api_key=api_key,
                translate_to_english=translate_to_english,
                engine=engine,
            )
            validate_workflow_graph(analyzed)
            _store_engine_analysis(
                engine,
                analyzed,
                translate_to_english,
                transcript_text=analysis_text,
            )
            if analyzed["nodes"]:
                status.update(
                    label=f"{engine_label} flowchart ready",
                    state="complete",
                    expanded=False,
                )
                success_msg = (
                    f"Successfully translated to American English and updated the {engine_label} workflow diagram!"
                    if translate_to_english
                    else f"Workflow diagram generated with {engine_label}."
                )
                st.success(success_msg)
            else:
                status.update(
                    label="No supported workflow found",
                    state="complete",
                    expanded=False,
                )
                st.info("The transcript did not establish an executable workflow.")
        except Exception as error:
            status.update(label=f"{engine_label} analysis failed", state="error")
            _set_engine_error(engine, f"{engine_label} analysis error occurred: {error}")


def _handle_add_speech_utterance(new_utterance: str, speaker_name: str) -> None:
    """Append newly typed/dictated utterance and update mock state."""
    if not new_utterance.strip():
        return

    utterances = st.session_state.current_meeting.get("utterances", [])
    new_id = len(utterances) + 1
    new_item = {
        "id": new_id,
        "timestamp": "LIVE",
        "speaker": speaker_name or "Participant",
        "text": new_utterance,
        "is_business": True,
        "relevance_score": 0.90,
        "action_type": "add",
        "detected_step": new_utterance,
        "detected_role": "Sales Representative",
    }
    utterances.append(new_item)
    st.session_state.current_meeting["utterances"] = utterances


def _render_meeting_source_selector() -> str:
    """Render the video-or-paste source control and return the selected source."""
    selected = st.segmented_control(
        "Meeting source",
        options=[SOURCE_UPLOAD, SOURCE_PASTE],
        default=SOURCE_UPLOAD,
        key="meeting_source",
        help="Paste or import an existing transcript to skip video-to-text extraction.",
    )
    return selected or SOURCE_UPLOAD


def _render_transcript_input(paste_source: bool) -> tuple[str, Any]:
    """Render the meeting transcript box and persist pasted text across reruns."""
    if paste_source:
        st.caption(
            "Paste or import an existing transcript. Extraction uses this text and skips video-to-text."
        )
        transcript_slot = st.empty()
        transcript = _show_transcript_box(
            transcript_slot,
            st.session_state.transcript_input,
            disabled=False,
            paste_source=True,
            collapse_label=False,
        )
    else:
        with st.expander("Meeting Transcript", expanded=True):
            transcript_slot = st.empty()
            transcript = _show_transcript_box(
                transcript_slot,
                st.session_state.transcript_input,
                disabled=False,
                paste_source=False,
            )
    st.session_state.transcript_input = transcript
    return transcript, transcript_slot


def _render_meeting_file_uploader(paste_source: bool) -> Any:
    """Render the meeting file uploader and load TXT/MD transcripts into the transcript box."""
    if paste_source:
        uploaded = st.file_uploader(
            "Import transcript file (TXT, MD)",
            type=list(TRANSCRIPT_FILE_EXTENSIONS),
            help="Load a meeting transcript from a text or Markdown file. Video-to-text extraction is skipped.",
            key="transcript_import_uploader",
        )
        if _is_transcript_file(uploaded):
            _apply_imported_transcript_file(uploaded)
        return None
    uploaded = st.file_uploader(
        "Upload Meeting Video, Audio, or Transcript (MP4, MOV, MP3, WAV, TXT, MD)",
        type=[*MEDIA_FILE_EXTENSIONS, *TRANSCRIPT_FILE_EXTENSIONS],
        help="Upload a video or audio file to transcribe, or a TXT/MD transcript file to skip video-to-text.",
        key="meeting_file_uploader",
    )
    if _is_transcript_file(uploaded):
        _apply_imported_transcript_file(uploaded)
        return None
    if uploaded is not None:
        with st.expander("Meeting video preview"):
            st.video(uploaded)
    return uploaded


def _render_input_controls(engine: str, translate_to_english: bool) -> tuple[str, bool, Any, Any, bool]:
    """Render meeting source, sample loader, extract button, and transcript input."""
    paste_source = _render_meeting_source_selector() == SOURCE_PASTE
    uploaded_video = _render_meeting_file_uploader(paste_source)

    engine_label = _engine_display_name(engine)
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        load_sample = st.button("Load sample transcript and flowchart", use_container_width=True)
    with col_btn2:
        button_label = (
            f"🚀 Translate & Extract Workflow ({engine_label})"
            if translate_to_english
            else f"🚀 Extract Workflow Diagram ({engine_label})"
        )
        run_llm = st.button(button_label, type="primary", use_container_width=True)

    if load_sample:
        _reset_to_sample_meeting()

    transcript, transcript_slot = _render_transcript_input(paste_source)
    return transcript, run_llm, uploaded_video, transcript_slot, paste_source


def main() -> None:
    """Main application loop."""
    st.set_page_config(
        page_title="Meeting Flow Live",
        page_icon="🎯",
        layout="wide",
    )
    _initialize_state()
    vl_model, engine, translate_to_english, create_clicked = _render_sidebar()
    _sync_current_meeting_from_engine(engine)
    _render_header()

    meeting_data = _meeting_for_display(engine)
    _render_summary_metrics(meeting_data)

    left_col, right_col = st.columns([1, 2], gap="medium")

    with left_col:
        transcript_text, run_llm_clicked, uploaded_file, transcript_slot, paste_source = (
            _render_input_controls(
                engine,
                translate_to_english,
            )
        )
        if create_clicked or run_llm_clicked:
            api_key = get_openrouter_api_key()
            media_for_transcript = None
            if not paste_source:
                media_for_transcript = uploaded_file if run_llm_clicked else None
                if media_for_transcript is None and not str(transcript_text or "").strip():
                    media_for_transcript = uploaded_file
            _handle_run_analysis(
                transcript_text,
                api_key,
                translate_to_english,
                engine,
                uploaded_file=media_for_transcript,
                vl_model=vl_model,
                transcript_slot=transcript_slot,
            )

        meeting_data = _meeting_for_display(engine)
        _render_confirmation_alert(meeting_data.get("confirmations", []))
        _render_transcript_feed(meeting_data.get("utterances", []))

    with right_col:
        engine_error = st.session_state.engine_errors.get(engine)
        if engine_error:
            st.error(engine_error)
        elif not _engine_has_result(engine):
            st.caption(
                f"No {_engine_display_name(engine)} flowchart yet. "
                f"Click Create flowchart with {_engine_display_name(engine)} in the sidebar to generate it."
            )
        _render_flowchart_view(_meeting_for_display(engine))

    # Chat / utterance input docked at bottom
    user_speech = st.chat_input("Enter utterance to simulate speech (e.g., Manager approves lease contract, then routes to Legal)")
    if user_speech:
        _handle_add_speech_utterance(user_speech, "Participant")


if __name__ == "__main__":
    main()
