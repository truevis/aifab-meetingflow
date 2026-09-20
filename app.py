"""Streamlit interface for Meeting Flow Live: Meeting Video/Audio to Markdown Flowchart."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import streamlit as st

from utils.flowchart_generator import build_mermaid_flowchart
from utils.meeting_analyzer import (
    analyze_transcript_with_jev,
    analyze_transcript_with_mercury,
    get_default_vl_model,
    get_sample_meeting_data,
    get_vl_model_label,
    get_vl_transcription_models,
    is_ling_vl_model,
    transcribe_media_with_openrouter,
)

ENGINE_MERCURY = "mercury"
ENGINE_JEV = "jev"


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


def _selected_engine(use_jev: bool) -> str:
    """Return the flowchart cache key for the sidebar toggle."""
    return ENGINE_JEV if use_jev else ENGINE_MERCURY


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
    }


def _engine_display_name(use_jev: bool) -> str:
    """Return the human-readable name of the selected flowchart engine."""
    return "Jev" if use_jev else "Mercury 2.5"


def _engine_has_result(engine: str) -> bool:
    """Return whether the selected engine already has a cached flowchart."""
    return bool(st.session_state.get("engine_has_result", {}).get(engine))


def _reset_to_sample_meeting() -> None:
    """Load the English sample into both engine caches and the displayed meeting."""
    st.session_state.meeting_by_engine = {
        ENGINE_MERCURY: copy.deepcopy(get_sample_meeting_data()),
        ENGINE_JEV: copy.deepcopy(get_sample_meeting_data()),
    }
    st.session_state.engine_has_result = {ENGINE_MERCURY: True, ENGINE_JEV: True}
    st.session_state.engine_errors = {}
    engine = _selected_engine(bool(st.session_state.get("flowchart_use_jev", False)))
    st.session_state.current_meeting = st.session_state.meeting_by_engine[engine]
    st.session_state.transcript_input = _transcript_from_meeting(st.session_state.current_meeting)


def _sync_current_meeting_from_engine(use_jev: bool) -> None:
    """Point the displayed meeting at the cached payload for the selected engine."""
    engine = _selected_engine(use_jev)
    caches = st.session_state.get("meeting_by_engine", {})
    if engine in caches:
        st.session_state.current_meeting = caches[engine]


def _initialize_state() -> None:
    """Initialize session state variables."""
    if "flowchart_use_jev" not in st.session_state:
        st.session_state.flowchart_use_jev = False
    if "translate_to_english" not in st.session_state:
        st.session_state.translate_to_english = False
    if "engine_errors" not in st.session_state:
        st.session_state.engine_errors = {}
    if "engine_has_result" not in st.session_state:
        st.session_state.engine_has_result = {ENGINE_MERCURY: False, ENGINE_JEV: False}
    if "meeting_by_engine" not in st.session_state:
        st.session_state.meeting_by_engine = {
            ENGINE_MERCURY: _empty_meeting_data(),
            ENGINE_JEV: _empty_meeting_data(),
        }
    if "current_meeting" not in st.session_state:
        engine = _selected_engine(bool(st.session_state.get("flowchart_use_jev", False)))
        st.session_state.current_meeting = st.session_state.meeting_by_engine.get(
            engine, _empty_meeting_data()
        )
    if "is_analyzing" not in st.session_state:
        st.session_state.is_analyzing = False
    if "transcript_input" not in st.session_state:
        st.session_state.transcript_input = _transcript_from_meeting(st.session_state.current_meeting)


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
    empty_caption = "No flowchart yet. Upload a meeting or load the sample."

    tab1, tab2, tab3 = st.tabs(["📊 Flowchart Diagram", "📋 Step Table", "📝 Markdown Source"])

    with tab1:
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


def _render_sidebar() -> tuple[str, bool, bool, bool]:
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
    use_jev = st.sidebar.toggle(
        "Use Jev instead of Mercury for flowchart",
        key="flowchart_use_jev",
        help="Off = Mercury 2.5. On = Jev. Switching shows that engine's cached flowchart if one exists.",
    )
    engine_label = _engine_display_name(use_jev)
    create_clicked = st.sidebar.button(
        f"Create flowchart with {engine_label}",
        type="primary",
        use_container_width=True,
        help="Runs only the currently selected flowchart model on the meeting transcript.",
    )
    if not _engine_has_result(_selected_engine(use_jev)):
        st.sidebar.caption(
            f"No {engine_label} flowchart yet. Click the button above to generate it."
        )
    return selected_model, use_jev, translate_to_english, create_clicked


def _read_uploaded_media(uploaded_file: Any) -> tuple[bytes, str]:
    """Read bytes and filename from a Streamlit uploaded media file."""
    return uploaded_file.getvalue(), uploaded_file.name


def _analyze_with_selected_engine(
    transcript_text: str,
    api_key: str,
    translate_to_english: bool,
    use_jev: bool,
) -> dict[str, Any]:
    """Run workflow extraction with only the currently selected flowchart engine."""
    if use_jev:
        return analyze_transcript_with_jev(
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
) -> None:
    """Cache a successful analysis for one engine and display it."""
    meeting = copy.deepcopy(st.session_state.meeting_by_engine.get(engine) or _empty_meeting_data())
    meeting["nodes"] = analyzed.get("nodes", [])
    meeting["edges"] = analyzed.get("edges", [])
    meeting["lanes"] = analyzed.get("lanes", [])
    meeting["confirmations"] = analyzed.get("confirmations", [])
    if analyzed.get("utterances"):
        meeting["utterances"] = analyzed["utterances"]
    st.session_state.meeting_by_engine[engine] = meeting
    st.session_state.engine_has_result[engine] = True
    st.session_state.engine_errors[engine] = None
    st.session_state.current_meeting = meeting
    if translate_to_english and analyzed.get("translated_transcript"):
        st.session_state.transcript_input = analyzed["translated_transcript"]


def _handle_run_analysis(
    transcript_text: str,
    api_key: str,
    translate_to_english: bool,
    use_jev: bool,
    uploaded_file: Any | None = None,
    vl_model: str = "",
) -> None:
    """Transcribe if needed, then extract a flowchart with only the selected engine."""
    if not api_key:
        st.error("Please configure your OpenRouter API key in .streamlit/secrets.toml or environment variables.")
        return

    engine = _selected_engine(use_jev)
    engine_label = _engine_display_name(use_jev)
    selected_vl_model = vl_model or get_default_vl_model()
    vl_label = get_vl_model_label(selected_vl_model)
    if uploaded_file is not None:
        spinner_msg = f"Transcribing with {vl_label}, then extracting workflow with {engine_label}..."
    elif translate_to_english:
        spinner_msg = f"Translating meeting to English and extracting workflow diagram with {engine_label}..."
    else:
        spinner_msg = f"Extracting workflow diagram with {engine_label}..."

    with st.spinner(spinner_msg):
        try:
            analysis_text = transcript_text
            if uploaded_file is not None:
                file_bytes, filename = _read_uploaded_media(uploaded_file)
                analysis_text = transcribe_media_with_openrouter(
                    file_bytes,
                    filename,
                    api_key=api_key,
                    model=selected_vl_model,
                )
                st.session_state.transcript_input = analysis_text

            analyzed = _analyze_with_selected_engine(
                analysis_text,
                api_key=api_key,
                translate_to_english=translate_to_english,
                use_jev=use_jev,
            )
            if "nodes" in analyzed and "edges" in analyzed:
                _store_engine_analysis(engine, analyzed, translate_to_english)
                success_msg = (
                    f"Successfully translated to American English and updated the {engine_label} workflow diagram!"
                    if translate_to_english
                    else f"Workflow diagram successfully generated with {engine_label}!"
                )
                st.success(success_msg)
            else:
                st.session_state.engine_errors[engine] = "The analysis response format was invalid."
                st.error(st.session_state.engine_errors[engine])
        except Exception as error:
            message = _display_text(f"{engine_label} analysis error occurred: {error}")
            st.session_state.engine_errors[engine] = message
            st.error(message)


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


def _render_input_controls(use_jev: bool, translate_to_english: bool) -> tuple[str, bool, Any]:
    """Render meeting video upload, sample scenario loader, extract button, and transcript input."""
    uploaded_video = st.file_uploader(
        "Upload Meeting Video or Audio (MP4, MOV, MP3, WAV)",
        type=["mp4", "mov", "m4a", "mp3", "wav"],
        help="Upload a video or audio file to generate the workflow diagram.",
    )

    if uploaded_video is not None:
        with st.expander("Meeting video preview"):
            st.video(uploaded_video)

    engine_label = _engine_display_name(use_jev)
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

    with st.expander("Meeting Transcript"):
        transcript = st.text_area(
            "Meeting Transcript",
            value=st.session_state.transcript_input,
            height=180,
            label_visibility="collapsed",
            help="Transcript text from the meeting. If 'Translate meeting to English' is enabled in the sidebar, it will be translated.",
        )

    return transcript, run_llm, uploaded_video


def main() -> None:
    """Main application loop."""
    st.set_page_config(
        page_title="Meeting Flow Live",
        page_icon="🎯",
        layout="wide",
    )
    _initialize_state()
    vl_model, use_jev, translate_to_english, create_clicked = _render_sidebar()
    _sync_current_meeting_from_engine(use_jev)
    _render_header()

    meeting_data = st.session_state.current_meeting
    _render_summary_metrics(meeting_data)

    left_col, right_col = st.columns([1, 2], gap="medium")

    with left_col:
        transcript_text, run_llm_clicked, uploaded_file = _render_input_controls(
            use_jev,
            translate_to_english,
        )
        if create_clicked or run_llm_clicked:
            api_key = get_openrouter_api_key()
            _handle_run_analysis(
                transcript_text,
                api_key,
                translate_to_english,
                use_jev,
                uploaded_file=uploaded_file if run_llm_clicked else None,
                vl_model=vl_model,
            )

        meeting_data = st.session_state.current_meeting
        _render_confirmation_alert(meeting_data.get("confirmations", []))
        _render_transcript_feed(meeting_data.get("utterances", []))

    with right_col:
        engine_error = st.session_state.engine_errors.get(_selected_engine(use_jev))
        if engine_error:
            st.error(engine_error)
        elif not _engine_has_result(_selected_engine(use_jev)):
            st.caption(
                f"No {_engine_display_name(use_jev)} flowchart yet. "
                f"Click Create flowchart with {_engine_display_name(use_jev)} in the sidebar to generate it."
            )
        _render_flowchart_view(st.session_state.current_meeting)

    # Chat / utterance input docked at bottom
    user_speech = st.chat_input("Enter utterance to simulate speech (e.g., Manager approves lease contract, then routes to Legal)")
    if user_speech:
        _handle_add_speech_utterance(user_speech, "Participant")


if __name__ == "__main__":
    main()
