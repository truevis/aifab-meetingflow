"""Meeting analysis module for Speech-to-Text, translation, and workflow extraction using OpenRouter."""

from __future__ import annotations

import base64
import json
import os
import re
from typing import Any

import requests

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
DEFAULT_MODEL = "inception/mercury-2.5"
JEV_MODEL = "~typesafe/jev-latest"
DEFAULT_VL_MODEL = "inclusionai/ling-3.0-flash-vl:free"
TRANSCRIPTION_TIMEOUT_SECONDS = 300
MERCURY_TIMEOUT_SECONDS = 60
JEV_TIMEOUT_SECONDS = 120
JEV_BUSINESS_THRESHOLD = 0.5
JEV_UTTERANCE_BATCH_SIZE = 10
JEV_UTTERANCE_TEXT_CHARS = 400
JEV_STEP_LABEL_CHARS = 48
JEV_FALLBACK_ROLES = {"Unassigned", "Other", "Facilitator"}

_TIMESTAMPED_UTTERANCE_RE = re.compile(
    r"^\[(?P<timestamp>[^\]]+)\]\s*(?:(?P<speaker>[^:]{1,80}):\s*)?(?P<text>.*)$"
)
_SPEAKER_UTTERANCE_RE = re.compile(
    r"^(?P<speaker>[A-Za-z][A-Za-z0-9 ./\-]{0,40}):\s*(?P<text>.+)$"
)
_SECTION_HEADING_RE = re.compile(
    r"Section\s+(\d+)\.\s+([^?.]+?)(?:\s*[?.]|$)",
    re.IGNORECASE,
)
_ANSWERED_STATUS_RE = re.compile(
    r"\?\s*(yes|no|current|cancelled|on schedule|approved)\b",
    re.IGNORECASE,
)
_PROCESS_BRANCH_RE = re.compile(
    r"\b(if |unless |otherwise|rejected|approved vs|passed\?|failed)\b",
    re.IGNORECASE,
)

# Sample scenario replicating the demonstration from the X video (@masa_okamura108) in American English
SAMPLE_REAL_ESTATE_TRANSCRIPT = [
    {
        "id": 1,
        "timestamp": "00:03",
        "speaker": "Facilitator",
        "text": "Alright everyone, let's get our meeting started.",
        "is_business": False,
        "relevance_score": 0.10,
        "action_type": "none",
        "detected_step": None,
        "detected_role": None,
    },
    {
        "id": 2,
        "timestamp": "00:08",
        "speaker": "Sales Rep",
        "text": "Today we want to map out the entire workflow from tenant inquiry to lease handover.",
        "is_business": True,
        "relevance_score": 0.62,
        "action_type": "add",
        "detected_step": "Start",
        "detected_role": "Sales Representative",
    },
    {
        "id": 3,
        "timestamp": "00:18",
        "speaker": "Sales Rep",
        "text": "First, the Sales Representative receives an inquiry from a prospective tenant via phone or web portal.",
        "is_business": True,
        "relevance_score": 0.97,
        "action_type": "add",
        "detected_step": "Receive tenant inquiry",
        "detected_role": "Sales Representative",
    },
    {
        "id": 4,
        "timestamp": "00:30",
        "speaker": "Sales Rep",
        "text": "Next, the Sales Representative qualifies their criteria and proposes available properties.",
        "is_business": True,
        "relevance_score": 0.97,
        "action_type": "add",
        "detected_step": "Qualify requirements / Propose properties",
        "detected_role": "Sales Representative",
    },
    {
        "id": 5,
        "timestamp": "00:41",
        "speaker": "Colleague",
        "text": "By the way, did anyone try that new coffee shop across the street?",
        "is_business": False,
        "relevance_score": 0.02,
        "action_type": "none",
        "detected_step": None,
        "detected_role": None,
    },
    {
        "id": 6,
        "timestamp": "00:46",
        "speaker": "Sales Rep",
        "text": "Sorry about that! If they like a property, the Sales Representative conducts an on-site property tour.",
        "is_business": True,
        "relevance_score": 0.97,
        "action_type": "add",
        "detected_step": "Conduct property tour",
        "detected_role": "Sales Representative",
    },
    {
        "id": 7,
        "timestamp": "00:54",
        "speaker": "Sales Rep",
        "text": "After the showing, the prospective tenant submits a rental application.",
        "is_business": True,
        "relevance_score": 0.96,
        "action_type": "add",
        "detected_step": "Submit rental application",
        "detected_role": "Applicant",
    },
    {
        "id": 8,
        "timestamp": "01:02",
        "speaker": "Sales Rep",
        "text": "Upon receiving the application, the Underwriting / Guarantor conducts credit screening.",
        "is_business": True,
        "relevance_score": 0.97,
        "action_type": "add",
        "detected_step": "Conduct credit screening",
        "detected_role": "Underwriting",
    },
    {
        "id": 9,
        "timestamp": "01:10",
        "speaker": "Sales Rep",
        "text": "If the application fails credit screening, the Sales Representative proposes alternative listings.",
        "is_business": True,
        "relevance_score": 0.97,
        "action_type": "branch",
        "detected_step": "Screening Passed? (Rejected -> Propose alternative listings)",
        "detected_role": "Underwriting / Sales Representative",
    },
    {
        "id": 10,
        "timestamp": "01:21",
        "speaker": "Sales Rep",
        "text": "If approved, our Contracts Officer prepares the lease and reviews disclosure agreements.",
        "is_business": True,
        "relevance_score": 0.97,
        "action_type": "add",
        "detected_step": "Explain disclosures / Execute lease contract",
        "detected_role": "Contracts",
    },
    {
        "id": 11,
        "timestamp": "01:34",
        "speaker": "Sales Rep",
        "text": "Wait, hold on — before signing, someone has to get final approval from the property owner.",
        "is_business": True,
        "relevance_score": 0.89,
        "action_type": "unclear",
        "detected_step": "Obtain landlord approval (Unassigned role)",
        "detected_role": "Unassigned",
    },
    {
        "id": 12,
        "timestamp": "01:44",
        "speaker": "Sales Rep",
        "text": "Once executed, Finance verifies the initial deposit payment.",
        "is_business": True,
        "relevance_score": 0.81,
        "action_type": "add",
        "detected_step": "Verify initial payment",
        "detected_role": "Finance",
    },
    {
        "id": 13,
        "timestamp": "01:50",
        "speaker": "Sales Rep",
        "text": "When payment is confirmed, Property Management hands over the keys, completing onboarding.",
        "is_business": True,
        "relevance_score": 0.95,
        "action_type": "add",
        "detected_step": "Hand over keys / Complete onboarding",
        "detected_role": "Property Management",
    },
    {
        "id": 14,
        "timestamp": "01:58",
        "speaker": "Sales Rep",
        "text": "Oh, one correction: mandatory disclosures must be delivered by a Licensed Broker, not Contracts.",
        "is_business": True,
        "relevance_score": 0.97,
        "action_type": "modify",
        "detected_step": "Reassign statutory disclosure to Licensed Broker",
        "detected_role": "Licensed Broker",
    },
]

SAMPLE_REAL_ESTATE_NODES = [
    {"id": "N1", "lane": "Sales Representative", "label": "Start", "node_type": "start"},
    {"id": "N2", "lane": "Sales Representative", "label": "Receive tenant inquiry", "node_type": "action"},
    {"id": "N3", "lane": "Sales Representative", "label": "Qualify criteria", "node_type": "action"},
    {"id": "N4", "lane": "Sales Representative", "label": "Propose properties", "node_type": "action"},
    {"id": "N5", "lane": "Sales Representative", "label": "Conduct property tour", "node_type": "action"},
    {"id": "N6", "lane": "Applicant", "label": "Submit rental application", "node_type": "action"},
    {"id": "N7", "lane": "Underwriting", "label": "Conduct credit screening", "node_type": "action"},
    {"id": "N8", "lane": "Underwriting", "label": "Screening Passed?", "node_type": "decision"},
    {"id": "N9", "lane": "Sales Representative", "label": "Propose alternative listings", "node_type": "action"},
    {"id": "N10", "lane": "Unassigned", "label": "Obtain landlord approval", "node_type": "action"},
    {"id": "N11", "lane": "Licensed Broker", "label": "Review statutory disclosures", "node_type": "action"},
    {"id": "N12", "lane": "Contracts", "label": "Execute lease agreement", "node_type": "action"},
    {"id": "N13", "lane": "Finance", "label": "Verify initial deposit payment", "node_type": "action"},
    {"id": "N14", "lane": "Property Management", "label": "Hand over keys", "node_type": "action"},
    {"id": "N15", "lane": "Property Management", "label": "Complete Onboarding", "node_type": "end"},
]

SAMPLE_REAL_ESTATE_EDGES = [
    {"source": "N1", "target": "N2", "label": None},
    {"source": "N2", "target": "N3", "label": None},
    {"source": "N3", "target": "N4", "label": None},
    {"source": "N4", "target": "N5", "label": None},
    {"source": "N5", "target": "N6", "label": None},
    {"source": "N6", "target": "N7", "label": None},
    {"source": "N7", "target": "N8", "label": None},
    {"source": "N8", "target": "N9", "label": "Rejected"},
    {"source": "N9", "target": "N4", "label": None},
    {"source": "N8", "target": "N10", "label": "Approved"},
    {"source": "N10", "target": "N11", "label": None},
    {"source": "N11", "target": "N12", "label": None},
    {"source": "N12", "target": "N13", "label": None},
    {"source": "N13", "target": "N14", "label": None},
    {"source": "N14", "target": "N15", "label": None},
]

SAMPLE_REAL_ESTATE_CONFIRMATIONS = [
    {
        "id": 1,
        "question": "Who is responsible for obtaining landlord approval before signing?",
        "suggested_role": "Sales Representative or Contracts",
        "confidence": 0.65,
    }
]

SAMPLE_REAL_ESTATE_LANES = [
    "Sales Representative",
    "Applicant",
    "Underwriting",
    "Unassigned",
    "Licensed Broker",
    "Contracts",
    "Finance",
    "Property Management",
]


def get_sample_meeting_data() -> dict[str, Any]:
    """Return pre-extracted demonstration data based on the X demo video in American English."""
    return {
        "utterances": SAMPLE_REAL_ESTATE_TRANSCRIPT,
        "nodes": SAMPLE_REAL_ESTATE_NODES,
        "edges": SAMPLE_REAL_ESTATE_EDGES,
        "lanes": SAMPLE_REAL_ESTATE_LANES,
        "confirmations": SAMPLE_REAL_ESTATE_CONFIRMATIONS,
    }


def _workflow_system_prompt(translate_to_english: bool) -> str:
    """Build the shared workflow-extraction system prompt for OpenRouter chat completions."""
    if translate_to_english:
        translation_instruction = (
            "CRITICAL TRANSLATION REQUIREMENT:\n"
            "The user requested translation to American English. If the input transcript is in any language "
            "other than English, you MUST translate everything into natural American English. All lanes, "
            "node labels, branch conditions, confirmation questions, and utterance transcripts MUST be completely translated into American English.\n\n"
        )
    else:
        translation_instruction = (
            "LANGUAGE PREFERENCE:\n"
            "Process the meeting in English (assume English meeting by default). Keep all generated labels, lanes, and questions in clean American English.\n\n"
        )

    return (
        "You are an expert business workflow engineer and analyst.\n"
        "Your task is to analyze the provided meeting transcript to extract a business workflow diagram with swimlanes, "
        "as well as a structured utterance log.\n\n"
        f"{translation_instruction}"
        "ANALYSIS RULES:\n"
        "1. Identify chit-chat vs business process talk. Ignore chit-chat from the workflow nodes.\n"
        "2. Extract lanes (roles/departments), nodes (id, lane, label, node_type: start|action|decision|end), "
        "and directed edges (source, target, label).\n"
        "3. Detect any unclear or ambiguous items (e.g., missing responsibilities, unassigned roles) as confirmation questions.\n"
        "4. Output an utterance log with timestamp, speaker, text, "
        "is_business (boolean), relevance_score (0.0 - 1.0), action_type (add|modify|branch|unclear|none), detected_step, and detected_role.\n"
        "5. Output strictly valid JSON matching this schema:\n"
        "{\n"
        '  "translated_transcript": "full transcript text",\n'
        '  "utterances": [\n'
        '     {"id": 1, "timestamp": "00:00", "speaker": "Speaker", "text": "Utterance text", "is_business": true, "relevance_score": 0.95, "action_type": "add", "detected_step": "Step name", "detected_role": "Role"}\n'
        "  ],\n"
        '  "lanes": ["Role 1", "Role 2", ...],\n'
        '  "nodes": [\n'
        '     {"id": "N1", "lane": "Role 1", "label": "Step description", "node_type": "start|action|decision|end"}\n'
        "  ],\n"
        '  "edges": [\n'
        '     {"source": "N1", "target": "N2", "label": "optional branch condition"}\n'
        "  ],\n"
        '  "confirmations": [\n'
        '     {"id": 1, "question": "Question text", "suggested_role": "Role", "confidence": 0.7}\n'
        "  ]\n"
        "}"
    )


def _raise_for_openrouter(response: requests.Response) -> None:
    """Raise an HTTP error that includes the OpenRouter response body."""
    if response.ok:
        return
    detail = (response.text or "").strip() or response.reason
    raise requests.HTTPError(
        f"{response.status_code} Client Error: {response.reason} for url: {response.url} — {detail}",
        response=response,
    )


def _analyze_transcript_with_openrouter(
    transcript_text: str,
    api_key: str,
    model: str,
    translate_to_english: bool,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Call OpenRouter chat completions to extract a workflow JSON payload."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/aifab-meetingflow",
        "X-Title": "MeetingFlowLive",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _workflow_system_prompt(translate_to_english)},
            {"role": "user", "content": transcript_text},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
    }
    response = requests.post(
        OPENROUTER_API_URL,
        headers=headers,
        json=payload,
        timeout=timeout_seconds,
    )
    _raise_for_openrouter(response)
    result_data = response.json()
    content = result_data["choices"][0]["message"]["content"]
    return json.loads(content)


def analyze_transcript_with_mercury(
    transcript_text: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    translate_to_english: bool = False,
) -> dict[str, Any]:
    """Call OpenRouter (defaulting to Inception Mercury 2.5) to analyze transcript text and extract workflow."""
    return _analyze_transcript_with_openrouter(
        transcript_text,
        api_key=api_key,
        model=model,
        translate_to_english=translate_to_english,
        timeout_seconds=MERCURY_TIMEOUT_SECONDS,
    )


def _clip_text(text: str, max_chars: int) -> str:
    """Trim text to a maximum character length."""
    value = str(text).strip()
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 1] + "…"


def _clip_at_word(text: str, max_chars: int) -> str:
    """Trim text at a word boundary without adding an ellipsis."""
    value = " ".join(str(text).split())
    if len(value) <= max_chars:
        return value
    clipped = value[:max_chars].rsplit(" ", 1)[0].rstrip(".,;:-")
    return clipped or value[:max_chars]


def _compact_step_label(text: str, max_chars: int = JEV_STEP_LABEL_CHARS) -> str:
    """Turn a transcript utterance into a short flowchart node label."""
    cleaned = " ".join(str(text).split())
    section = _SECTION_HEADING_RE.search(cleaned)
    if section:
        heading = f"Section {section.group(1)}. {section.group(2).strip(' .,-')}"
        return _clip_at_word(heading, max_chars)
    first = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)[0]
    first = re.sub(r"\s+(Yes|No|Cancelled)\b.*", "", first, flags=re.IGNORECASE).strip(" .")
    return _clip_at_word(first or cleaned, max_chars)


def _is_answered_status_check(text: str) -> bool:
    """Return True when a question is already answered in the same utterance."""
    return bool(_ANSWERED_STATUS_RE.search(text))


def _is_process_branch(text: str, action_type: str) -> bool:
    """Return True when the utterance is a real workflow fork, not a status Q&A."""
    if _is_answered_status_check(text):
        return False
    if _PROCESS_BRANCH_RE.search(text):
        return True
    return action_type == "branch" and "?" in text and " if " in f" {text.lower()} "


def _resolve_jev_lane(role_label: str, speaker: str) -> str:
    """Prefer a named process role; otherwise keep the meeting speaker as the lane."""
    speaker_name = str(speaker).strip() or "Participant"
    if not role_label or role_label in JEV_FALLBACK_ROLES:
        return speaker_name
    return role_label


def _role_key(name: str) -> str:
    """Normalize a role or speaker name into a Jev choice id."""
    key = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
    return key or "other"


def _pretty_role(role_id: str) -> str:
    """Return a display label for a swimlane role id."""
    labels = {
        "sales_representative": "Sales Representative",
        "applicant": "Applicant",
        "underwriting": "Underwriting",
        "contracts": "Contracts",
        "licensed_broker": "Licensed Broker",
        "finance": "Finance",
        "property_management": "Property Management",
        "manager": "Manager",
        "project_manager": "Project Manager",
        "procurement": "Procurement",
        "superintendent": "Superintendent",
        "architect": "Architect",
        "subcontractor": "Subcontractor",
        "facilitator": "Facilitator",
        "unassigned": "Unassigned",
        "other": "Other",
    }
    if role_id in labels:
        return labels[role_id]
    return role_id.replace("_", " ").title() or "Unassigned"


def _jev_generic_role_criteria() -> dict[str, str]:
    """Return closed-set role options Jev can assign to a process step."""
    return {
        "sales_representative": "Sales or leasing representative who handles inquiries and tours",
        "applicant": "Customer, tenant, applicant, or requester in the process",
        "underwriting": "Credit screening, underwriting, or risk review",
        "contracts": "Contracts, legal, or lease execution",
        "licensed_broker": "Licensed broker responsible for statutory disclosures",
        "finance": "Finance, payment, or deposit verification",
        "property_management": "Property management, operations, or key handover",
        "manager": "Manager or approver who signs off",
        "project_manager": "Project manager running the status meeting or overall job",
        "procurement": "Procurement, purchasing, POs, vendors, or buyout",
        "superintendent": "Field superintendent or site supervisor",
        "architect": "Architect, designer, or drawing author",
        "subcontractor": "Trade subcontractor performing the work",
        "facilitator": "Meeting facilitator who is not a process owner",
        "unassigned": "No responsible role is named",
        "other": "A different role than the listed options",
    }


def _jev_role_criteria(utterances: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, str]]:
    """Build Jev role choices from generic lanes plus speakers in the transcript."""
    generic = _jev_generic_role_criteria()
    criteria = dict(generic)
    labels = {key: _pretty_role(key) for key in criteria}
    for utterance in utterances:
        speaker = str(utterance.get("speaker") or "Participant").strip() or "Participant"
        key = _role_key(speaker)
        if key in generic:
            continue
        criteria[key] = f"The meeting speaker labeled {speaker}"
        labels[key] = speaker
    return criteria, labels


def _parse_transcript_utterances(transcript_text: str) -> list[dict[str, Any]]:
    """Split a transcript into timestamped speaker utterances."""
    lines = [line.strip() for line in str(transcript_text).splitlines() if line.strip()]
    if not lines and str(transcript_text).strip():
        lines = [str(transcript_text).strip()]
    utterances: list[dict[str, Any]] = []
    for line in lines:
        timestamp = "00:00"
        speaker = "Participant"
        text = line
        stamped = _TIMESTAMPED_UTTERANCE_RE.match(line)
        if stamped:
            timestamp = (stamped.group("timestamp") or timestamp).strip() or timestamp
            speaker = (stamped.group("speaker") or speaker).strip() or speaker
            text = (stamped.group("text") or "").strip()
        else:
            spoken = _SPEAKER_UTTERANCE_RE.match(line)
            if spoken:
                speaker = spoken.group("speaker").strip() or speaker
                text = spoken.group("text").strip()
        if not text:
            continue
        utterances.append(
            {
                "id": len(utterances) + 1,
                "timestamp": timestamp,
                "speaker": speaker,
                "text": text,
            }
        )
    return utterances


def _jev_action_criteria() -> dict[str, str]:
    """Return workflow action choices for one utterance."""
    return {
        "add": "Adds or reviews a process step, agenda section, or status item",
        "modify": "Corrects or reassigns an earlier process step",
        "branch": "Creates two different next steps, such as approve vs reject. A status question that is answered in the same turn is not a branch",
        "unclear": "Mentions a step but the owner or next action is missing",
        "none": "No workflow step, including greetings and chit-chat",
    }


def _build_jev_utterance_questions(
    utterances: list[dict[str, Any]],
    role_criteria: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Build Jev questions that classify a batch of transcript utterances."""
    questions: dict[str, dict[str, Any]] = {}
    action_criteria = _jev_action_criteria()
    for utterance in utterances:
        uid = f"u{utterance['id']}"
        questions[f"{uid}_business"] = {
            "type": "noul",
            "instructions": (
                f'For the utterance with id "{uid}": does this utterance describe a '
                "business process step rather than greeting or chit-chat?"
            ),
            "criteria": {
                "true": "Names a process step, role, approval, application, payment, or handover",
                "false": "Greeting, joke, small talk, or no process content",
            },
        }
        questions[f"{uid}_action"] = {
            "type": "choice",
            "instructions": (
                f'For the utterance with id "{uid}": what workflow action does this utterance represent?'
            ),
            "criteria": action_criteria,
        }
        questions[f"{uid}_role"] = {
            "type": "choice",
            "instructions": (
                f'For the utterance with id "{uid}": which role owns the process step being described?'
            ),
            "criteria": role_criteria,
        }
    return questions


def _jev_state(utterances: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the Decisions API state payload for a meeting transcript."""
    return {
        "description": "A meeting transcript split into numbered utterances.",
        "utterances": [
            {
                "id": f"u{utterance['id']}",
                "timestamp": utterance["timestamp"],
                "speaker": utterance["speaker"],
                "text": _clip_text(utterance["text"], JEV_UTTERANCE_TEXT_CHARS),
            }
            for utterance in utterances
        ],
    }


def _jev_headers(api_key: str) -> dict[str, str]:
    """Return OpenRouter headers for a Jev Decisions request."""
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/aifab-meetingflow",
        "X-Title": "MeetingFlowLive",
        "X-OpenRouter-Title": "MeetingFlowLive",
    }


def _ask_jev_decisions(
    api_key: str,
    model: str,
    state: dict[str, Any],
    questions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Submit one OpenRouter Decisions request and return the answers object."""
    payload = {"model": model, "state": state, "questions": questions}
    response = requests.post(
        OPENROUTER_DECISIONS_URL,
        headers=_jev_headers(api_key),
        json=payload,
        timeout=JEV_TIMEOUT_SECONDS,
    )
    _raise_for_openrouter(response)
    result_data = response.json()
    answers = result_data.get("answers")
    if not isinstance(answers, dict):
        raise RuntimeError("Jev returned no decision answers")
    return answers


def _choice_value(answer: Any, default: str) -> tuple[str, float]:
    """Read a Decisions choice and its confidence."""
    if not isinstance(answer, dict):
        return default, 0.0
    choice = str(answer.get("choice") or default)
    confidence = float(answer.get("confidence", 0.0) or 0.0)
    return choice, confidence


def _noul_value(answer: Any) -> float:
    """Read a Decisions yes/no probability."""
    if not isinstance(answer, dict):
        return 0.0
    return float(answer.get("noul", 0.0) or 0.0)


def _classify_utterances_with_jev(
    utterances: list[dict[str, Any]],
    api_key: str,
    model: str,
) -> list[dict[str, Any]]:
    """Classify each utterance with Jev and return labeled utterances."""
    role_criteria, role_labels = _jev_role_criteria(utterances)
    state = _jev_state(utterances)
    answers: dict[str, Any] = {}
    for start in range(0, len(utterances), JEV_UTTERANCE_BATCH_SIZE):
        batch = utterances[start : start + JEV_UTTERANCE_BATCH_SIZE]
        batch_answers = _ask_jev_decisions(
            api_key,
            model,
            state,
            _build_jev_utterance_questions(batch, role_criteria),
        )
        answers.update(batch_answers)

    classified: list[dict[str, Any]] = []
    for utterance in utterances:
        uid = f"u{utterance['id']}"
        relevance = _noul_value(answers.get(f"{uid}_business"))
        is_business = relevance >= JEV_BUSINESS_THRESHOLD
        action_type = _choice_value(answers.get(f"{uid}_action"), "none")[0]
        if action_type not in _jev_action_criteria():
            action_type = "none"
        if not is_business:
            action_type = "none"
        elif action_type == "branch" and _is_answered_status_check(utterance["text"]):
            action_type = "add"
        role_id, role_confidence = _choice_value(answers.get(f"{uid}_role"), "unassigned")
        role_label = _resolve_jev_lane(
            role_labels.get(role_id, _pretty_role(role_id)),
            str(utterance.get("speaker") or "Participant"),
        )
        step_label = _compact_step_label(utterance["text"])
        classified.append(
            {
                **utterance,
                "is_business": is_business,
                "relevance_score": round(relevance, 2),
                "action_type": action_type,
                "detected_step": step_label if is_business and action_type != "none" else None,
                "detected_role": role_label if is_business and action_type != "none" else None,
                "role_confidence": role_confidence,
            }
        )
    return classified


def _jev_node_type(label: str, action_type: str) -> str:
    """Map an utterance onto a flowchart node type."""
    if _is_process_branch(label, action_type):
        return "decision"
    return "action"


def _workflow_from_jev_utterances(utterances: list[dict[str, Any]]) -> dict[str, Any]:
    """Assemble lanes, nodes, edges, and confirmations from classified utterances."""
    step_utterances = [
        utterance
        for utterance in utterances
        if utterance.get("is_business") and utterance.get("action_type") != "none"
    ]
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    lanes: list[str] = []
    confirmations: list[dict[str, Any]] = []
    if not step_utterances:
        return {
            "nodes": nodes,
            "edges": edges,
            "lanes": lanes,
            "confirmations": confirmations,
        }

    def _add_lane(lane: str) -> None:
        if lane not in lanes:
            lanes.append(lane)

    first_lane = str(step_utterances[0].get("detected_role") or "Unassigned")
    last_lane = str(step_utterances[-1].get("detected_role") or first_lane)
    _add_lane(first_lane)
    nodes.append({"id": "N1", "lane": first_lane, "label": "Start", "node_type": "start"})
    previous_id = "N1"
    for index, utterance in enumerate(step_utterances):
        node_id = f"N{index + 2}"
        lane = str(utterance.get("detected_role") or "Unassigned")
        _add_lane(lane)
        label = str(utterance.get("detected_step") or _compact_step_label(utterance["text"]))
        nodes.append(
            {
                "id": node_id,
                "lane": lane,
                "label": label,
                "node_type": _jev_node_type(f"{label} {utterance.get('text', '')}", str(utterance.get("action_type") or "add")),
            }
        )
        edges.append({"source": previous_id, "target": node_id, "label": None})
        previous_id = node_id
        if utterance.get("action_type") == "unclear":
            confirmations.append(
                {
                    "id": len(confirmations) + 1,
                    "question": f"Who is responsible for '{label}'?",
                    "suggested_role": lane,
                    "confidence": round(float(utterance.get("role_confidence", 0.65) or 0.65), 2),
                }
            )
    end_id = f"N{len(step_utterances) + 2}"
    _add_lane(last_lane)
    nodes.append({"id": end_id, "lane": last_lane, "label": "Complete", "node_type": "end"})
    edges.append({"source": previous_id, "target": end_id, "label": None})
    return {
        "nodes": nodes,
        "edges": edges,
        "lanes": lanes,
        "confirmations": confirmations,
    }


def analyze_transcript_with_jev(
    transcript_text: str,
    api_key: str,
    model: str = JEV_MODEL,
    translate_to_english: bool = False,
) -> dict[str, Any]:
    """Classify a transcript with Jev Decisions and assemble a workflow payload.

    Jev does not generate text, so translate_to_english cannot rewrite labels.
    """
    if not str(transcript_text).strip():
        raise ValueError("Meeting transcript is empty.")
    parsed = _parse_transcript_utterances(transcript_text)
    if not parsed:
        raise ValueError("Could not parse any utterances from the meeting transcript.")
    classified = _classify_utterances_with_jev(parsed, api_key, model)
    workflow = _workflow_from_jev_utterances(classified)
    display_utterances = [
        {key: value for key, value in utterance.items() if key != "role_confidence"}
        for utterance in classified
    ]
    return {
        "translated_transcript": transcript_text,
        "utterances": display_utterances,
        **workflow,
    }


def get_vl_transcription_models() -> list[dict[str, str]]:
    """Return the OpenRouter VL models available for video/audio transcription."""
    return [
        {"id": "inclusionai/ling-3.0-flash-vl:free", "label": "Ling 3.0 Flash VL Free"},
        {"id": "z-ai/glm-5.3-flash", "label": "GLM 5.3 Flash"},
        {"id": "qwen/qwen3.8-flash", "label": "Qwen 3.8 Flash"},
        {"id": "google/gemini-3-flash-preview", "label": "Gemini 3 Flash Preview"},
        {"id": "google/gemini-3.6-flash", "label": "Gemini 3.6 Flash"},
    ]


def get_default_vl_model() -> str:
    """Return the default OpenRouter VL model ID for transcription."""
    return DEFAULT_VL_MODEL


def get_vl_model_label(model_id: str) -> str:
    """Return the human-readable label for a VL model ID."""
    for model in get_vl_transcription_models():
        if model["id"] == model_id:
            return model["label"]
    return model_id


def is_ling_vl_model(model_id: str) -> bool:
    """Return True when the Ling 3.0 Flash VL Free model is selected."""
    return model_id == DEFAULT_VL_MODEL


def _get_file_extension(filename: str) -> str:
    """Return the lowercase file extension without the leading dot."""
    if "." not in filename:
        return ""
    return filename.rsplit(".", 1)[-1].lower()


def _is_audio_filename(filename: str) -> bool:
    """Return True when the filename is a supported audio type."""
    return _get_file_extension(filename) in {"mp3", "wav", "m4a"}


def _get_media_mime_type(filename: str) -> str:
    """Map a meeting media filename to a MIME type."""
    mime_by_ext = {
        "mp4": "video/mp4",
        "mov": "video/quicktime",
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "m4a": "audio/mp4",
    }
    extension = _get_file_extension(filename)
    if _is_audio_filename(filename):
        return mime_by_ext.get(extension, "audio/mpeg")
    return mime_by_ext.get(extension, "video/mp4")


def _encode_media_base64(file_bytes: bytes) -> str:
    """Encode raw media bytes as a Base64 string."""
    return base64.b64encode(file_bytes).decode("utf-8")


def _is_gemini_vl_model(model_id: str) -> bool:
    """Return True when the selected VL model should use Gemini agentic video processing."""
    return model_id in {
        "google/gemini-3-flash-preview",
        "google/gemini-3.6-flash",
    }


def _build_transcription_prompt() -> str:
    """Return the user prompt for a timestamped speaker transcript in the original language."""
    return (
        "Transcribe this meeting into a timestamped speaker transcript in the original language. "
        "Use the format [MM:SS] Speaker: text on each line. "
        "Identify speakers when possible. Do not translate. "
        "Output only the transcript."
    )


def _build_transcription_user_content(
    file_bytes: bytes,
    filename: str,
    model: str,
) -> list[dict[str, Any]]:
    """Build the multimodal OpenRouter user content for video or audio transcription."""
    encoded = _encode_media_base64(file_bytes)
    if _is_audio_filename(filename):
        media_part: dict[str, Any] = {
            "type": "input_audio",
            "input_audio": {
                "data": encoded,
                "format": _get_file_extension(filename) or "mp3",
            },
        }
    else:
        mime_type = _get_media_mime_type(filename)
        media_part = {
            "type": "video_url",
            "video_url": {"url": f"data:{mime_type};base64,{encoded}"},
        }
        if _is_gemini_vl_model(model):
            media_part["processing"] = "agentic"
    return [
        media_part,
        {"type": "text", "text": _build_transcription_prompt()},
    ]


def transcribe_media_with_openrouter(
    file_bytes: bytes,
    filename: str,
    api_key: str,
    model: str = DEFAULT_VL_MODEL,
) -> str:
    """Call OpenRouter VL to transcribe uploaded meeting video or audio."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/aifab-meetingflow",
        "X-Title": "MeetingFlowLive",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": _build_transcription_user_content(file_bytes, filename, model),
            }
        ],
        "temperature": 0.1,
    }
    response = requests.post(
        OPENROUTER_API_URL,
        headers=headers,
        json=payload,
        timeout=TRANSCRIPTION_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    result_data = response.json()
    content = result_data["choices"][0]["message"]["content"]
    if isinstance(content, list):
        text_parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        return "\n".join(part for part in text_parts if part).strip()
    return str(content).strip()

