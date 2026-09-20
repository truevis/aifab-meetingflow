"""Tax-document classification through OpenRouter's Decisions API."""

from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

import requests


OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
OPENROUTER_HTTP_REFERER = "https://aifab-classify-foms.streamlit.app"
OPENROUTER_TITLE = "aifab-classify-foms"
MODEL = "~typesafe/jev-latest"
SECOND_PASS_MODEL = "inception/mercury-2.5"
SECOND_PASS_THRESHOLD = 0.9
CONFIDENCE_THRESHOLD = 0.95
SECOND_OPINION_SCHEMA = {
    "type": "object",
    "properties": {
        "form_type": {
            "type": "string",
            "description": "IRS form number/title, e.g. Form 1099 Composite",
        },
        "document_kind": {
            "type": "string",
            "description": "Short kind label for the document",
        },
        "included_forms": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Component forms if composite; empty otherwise",
        },
        "rationale": {
            "type": "string",
            "description": "Brief reason for the identification",
        },
    },
    "required": ["form_type", "document_kind", "included_forms", "rationale"],
    "additionalProperties": False,
}
# Jev's request budget is ~32k tokens for the whole payload (state + questions).
# Keep the submitted state well under it so long PDFs still classify.
MAX_STATE_CHARS = 48_000
MIN_STATE_CHARS = 4_000
MAX_OVERFLOW_SHRINKS = 6
CRITERIA_PATH = Path(__file__).resolve().parent.parent / "data" / "criteria.json"
NOT_IN_LIST = "not_in_this_list"
RARE_PARENTS = ("form-5471", "form-8865", "form-8933", "form-1118", "form-5713")

FALLBACK_FORM_CRITERIA = {
    "form-1040": "Form 1040 U.S. Individual Income Tax Return",
    "form-1040-schedule-a": "Form 1040 Schedule A Itemized Deductions",
    "form-1040-schedule-b": "Form 1040 Schedule B Interest and Ordinary Dividends",
    "form-1040-schedule-c": "Form 1040 Schedule C Profit or Loss From Business",
    "form-1040-schedule-d": "Form 1040 Schedule D Capital Gains and Losses",
    "form-1040-schedule-e": "Form 1040 Schedule E Supplemental Income and Loss",
    "form-w-2": "Form W-2 Wage and Tax Statement",
    "form-1099-int": "Form 1099-INT Interest Income",
    "form-1099-div": "Form 1099-DIV Dividends and Distributions",
    "form-1099-misc": "Form 1099-MISC Miscellaneous Income",
    "form-1099-nec": "Form 1099-NEC Nonemployee Compensation",
    "form-1099-b": "Form 1099-B Proceeds From Broker and Barter Exchange Transactions",
    "form-1099-r": "Form 1099-R Distributions From Pensions, Annuities, Retirement",
    "form-1098": "Form 1098 Mortgage Interest Statement",
    "form-1098-t": "Form 1098-T Tuition Statement",
    "form-1095-a": "Form 1095-A Health Insurance Marketplace Statement",
    "form-1095-c": "Form 1095-C Employer-Provided Health Insurance Offer and Coverage",
    "other": "A different tax form or a document that cannot be identified from this list",
}

KIND_CRITERIA = {
    "form_page": {
        "what": "A document containing at least one fillable entry (numbered lines, boxes, name/TIN fields), even if it also carries instruction text",
        "examples": [
            "1 Wages, tips, other compensation",
            "Name(s) shown on return  Your social security number",
            "Line 7 Subtract line 6 from line 5",
        ],
    },
    "instructions": {
        "what": "Only instruction prose explaining who must file, definitions, or boxes, with no fillable entry anywhere in the document",
        "examples": ["Instructions for Recipient", "General Instructions", "Purpose of Form"],
    },
    "blank": {"what": "A blank document or one with little or no extracted text"},
    "cover_sheet": {"what": "A cover or notice document, filing warning, or transmittal sheet"},
    "state_tax_form": {"what": "A state or local tax form or schedule issued by a state agency, not the IRS"},
    "broker_or_bank_statement": {
        "what": "A brokerage, bank, payroll, or custodian statement, account summary, or transaction detail"
    },
    "letter_or_other": {
        "what": "Any other page, such as a letter, invoice, filing instruction, e-file authorization, or preparer worksheet"
    },
}

FAMILY_DESCRIPTIONS = {
    "form-5471": "Form 5471 or any of its Schedules E, G-1, H, I-1, J, M, O, P, Q, R",
    "form-8865": "Form 8865 or its Schedules G, H, K-1, K-2, K-3, O, P",
    "form-8933": "Form 8933 or its Schedules A-F",
    "form-1118": "Form 1118 or its Schedules I, J, K, L",
    "form-5713": "Form 5713 or its Schedules A, B, C",
}

FORM_INSTRUCTIONS = (
    "Which IRS form or form family is this document from? Read the form number and title throughout "
    "the document. An instructions document belongs to the form it instructs. If the form is not among "
    "the options, choose not_in_this_list."
)

# SSN: 123-45-6789, 123 45 6789, 123.45.6789 (not EIN 12-3456789).
_SSN_RE = re.compile(r"(?<!\d)\d{3}[\s.\-\u2013\u2014]+\d{2}[\s.\-\u2013\u2014]+\d{4}(?!\d)")
_MONTH_NAME = (
    r"(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
)
_BIRTHDATE_RES = (
    re.compile(
        r"\b(?:0?[1-9]|1[0-2])\s*[-/.]\s*(?:0?[1-9]|[12]\d|3[01])\s*[-/.]\s*(?:19|20)\d{2}\b"
    ),
    re.compile(r"\b(?:19|20)\d{2}[-/](?:0?[1-9]|1[0-2])[-/](?:0?[1-9]|[12]\d|3[01])\b"),
    re.compile(r"\b(?:0?[1-9]|1[0-2])[-/.](?:0?[1-9]|[12]\d|3[01])[-/.]\d{2}\b"),
    re.compile(
        rf"\b{_MONTH_NAME}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+(?:19|20)\d{{2}}\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH_NAME}\.?,?\s+(?:19|20)\d{{2}}\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b{_MONTH_NAME}[-/]\d{{1,2}}[-/](?:19|20)\d{{2}}\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b\d{{1,2}}[-/]{_MONTH_NAME}[-/](?:19|20)\d{{2}}\b",
        re.IGNORECASE,
    ),
)


def _sanitize_openrouter_text(text: str) -> str:
    """Remove Social Security numbers and birthdates before sending text to OpenRouter."""
    sanitized = _SSN_RE.sub("[SSN]", text)
    for pattern in _BIRTHDATE_RES:
        sanitized = pattern.sub("[DATE]", sanitized)
    return sanitized


def _sanitize_openrouter_value(value: Any) -> Any:
    """Sanitize string values anywhere in an OpenRouter payload."""
    if isinstance(value, str):
        return _sanitize_openrouter_text(value)
    if isinstance(value, dict):
        return {key: _sanitize_openrouter_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_openrouter_value(item) for item in value]
    return value


def _fallback_criteria() -> dict[str, dict[str, Any]]:
    """Return the original app's compact criteria in the upstream data shape."""
    return {
        form_id: {"id": form_id, "label": description, "title": "", "pageCount": 1}
        for form_id, description in FALLBACK_FORM_CRITERIA.items()
    }


@lru_cache(maxsize=1)
def load_form_criteria() -> dict[str, dict[str, Any]]:
    """Load bundled form criteria, with the original compact list as fallback."""
    try:
        with CRITERIA_PATH.open(encoding="utf-8") as handle:
            criteria = json.load(handle)
        if not isinstance(criteria, dict) or len(criteria) < 200:
            raise ValueError("Form criteria are incomplete")
        return criteria
    except (OSError, ValueError):
        return _fallback_criteria()


def _criterion_for(entry: dict[str, Any], *, compact: bool = False) -> dict[str, Any]:
    """Convert one upstream form entry to a Decisions API criterion."""
    label = entry.get("label", entry.get("id", "Unknown form"))
    title = entry.get("title", "")
    criterion: dict[str, Any] = {"what": f"{label} — {title}" if title else label}
    if compact:
        return criterion
    if entry.get("boxes"):
        criterion["examples"] = entry["boxes"]
    if entry.get("notFor"):
        criterion["not_for"] = f"Not {', '.join(entry['notFor'])}"
    return criterion


def _family_of(form_id: str, criteria: dict[str, dict[str, Any]]) -> str:
    """Return the first-stage family for a form identifier."""
    parent = criteria[form_id].get("parent")
    return parent if parent in RARE_PARENTS else form_id


def _first_list_criteria(criteria: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build the first-stage form choices used by the reference classifier."""
    choices: dict[str, dict[str, Any]] = {}
    for form_id in sorted(criteria):
        if _family_of(form_id, criteria) != form_id:
            continue
        choices[form_id] = (
            {"what": FAMILY_DESCRIPTIONS[form_id]}
            if form_id in RARE_PARENTS
            else _criterion_for(criteria[form_id], compact=True)
        )
    choices[NOT_IN_LIST] = {"what": "The document belongs to a form or schedule not listed above"}
    return choices


def _family_criteria(parent: str, criteria: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build second-stage choices for a rare form family."""
    choices = {
        form_id: _criterion_for(criteria[form_id])
        for form_id in sorted(criteria)
        if form_id == parent or criteria[form_id].get("parent") == parent
    }
    choices[NOT_IN_LIST] = {"what": "The document belongs to a form not listed above"}
    return choices


def _lines_of(text: str) -> list[str]:
    """Normalize extracted PDF text into non-empty lines."""
    return [line.rstrip() for line in text.splitlines() if line.strip()]


def truncate_state(text: str, max_chars: int = MAX_STATE_CHARS) -> str:
    """Fit extracted PDF text into Jev's request budget by truncating the bottom.

    Form numbers and titles tend to appear on the first pages, so the head is
    kept intact and everything past the cap is cut from the bottom.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _is_blank(lines: list[str]) -> bool:
    """Return whether a page is blank enough to classify without an API call."""
    page_text = " ".join(lines).strip()
    return len(page_text) < 60 or "intentionally left blank" in page_text.lower() or "left blank intentionally" in page_text.lower()


def build_page_state(text: str, body_chars: int = 2500) -> dict[str, str]:
    """Split a page into the header/body/footer state used by the reference app."""
    lines = _lines_of(text)
    has_footer = len(lines) > 18
    header = lines[:12]
    footer = lines[-6:] if has_footer else []
    body = lines[12:-6] if has_footer else lines[12:]
    return {
        "header": "\n".join(header),
        "body": "\n".join(body)[:body_chars],
        "footer": "\n".join(footer),
    }


def build_classification_request(
    text: str,
    criteria: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build one OpenRouter request from the complete extracted PDF text."""
    form_criteria = criteria or load_form_criteria()
    return {
        "model": MODEL,
        "state": _sanitize_openrouter_text(text),
        "questions": {
            "kind": {
                "type": "choice",
                "instructions": "What kind of document is this overall?",
                "criteria": KIND_CRITERIA,
            },
            "form": {
                "type": "choice",
                "instructions": FORM_INSTRUCTIONS,
                "criteria": _first_list_criteria(form_criteria),
            },
        },
    }


def _usage_input_tokens(response_data: dict[str, Any]) -> int:
    """Read token usage across the documented OpenRouter naming variants."""
    usage = response_data.get("usage") or {}
    return int(usage.get("inputTokens", usage.get("input_tokens", 0)) or 0)


def _state_too_long(detail: str) -> bool:
    """Return whether a 400 detail indicates the submitted state overflowed Jev."""
    return "max_tokens_exceeded" in detail or "context_length_exceeded" in detail


def _slim_questions(questions: dict[str, dict[str, Any]]) -> bool:
    """Drop bulky examples and exclusions from choice criteria."""
    changed = False
    for question in questions.values():
        criteria = question.get("criteria")
        if not isinstance(criteria, dict):
            continue
        for criterion in criteria.values():
            if not isinstance(criterion, dict):
                continue
            for key in ("examples", "not_for"):
                if key in criterion:
                    del criterion[key]
                    changed = True
    return changed


def _shrink_overflow_payload(payload: dict[str, Any]) -> bool:
    """Reduce a Decisions payload after max_tokens. Return whether anything changed."""
    questions = payload.get("questions")
    slimed = _slim_questions(questions) if isinstance(questions, dict) else False
    state = payload.get("state")
    if not isinstance(state, str) or len(state) <= MIN_STATE_CHARS:
        return slimed
    next_chars = max(MIN_STATE_CHARS, len(state) // 2)
    if next_chars >= len(state):
        return slimed
    payload["state"] = truncate_state(state, max_chars=next_chars)
    return True


def _response_detail(response: requests.Response) -> str:
    """Return the response body, or a placeholder when OpenRouter sent nothing."""
    return response.text.strip() or "(empty response body)"


def _is_spotty_internet_error(error: BaseException, status_code: int | None = None) -> bool:
    """Return whether the failure could come from a dropped or flaky connection."""
    if status_code in (408, 429) or (status_code is not None and status_code >= 500):
        return True
    return isinstance(
        error,
        (
            requests.Timeout,
            requests.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.SSLError,
            requests.exceptions.JSONDecodeError,
        ),
    )


def _print_openrouter_error(
    error: BaseException,
    *,
    attempt: int,
    retries: int,
    retrying: bool,
    detail: str = "",
) -> None:
    """Print the full OpenRouter failure so the terminal is not just 'None'."""
    action = "waiting 1s then retrying" if retrying else "giving up"
    print(
        f"[ERROR] OpenRouter request failed "
        f"(attempt {attempt + 1}/{retries}, {action}): {error!r}",
        file=sys.stderr,
    )
    if detail:
        print(f"[ERROR] OpenRouter response: {detail}", file=sys.stderr)
    traceback.print_exception(type(error), error, error.__traceback__, file=sys.stderr)


def _ask_openrouter(
    api_key: str,
    state: Any,
    questions: dict[str, dict[str, Any]],
    session: requests.Session,
    retries: int = 3,
) -> tuple[dict[str, Any], int]:
    """Submit a Decisions request and retry flaky-network failures after 1s."""
    payload = _sanitize_openrouter_value(
        {"model": MODEL, "state": state, "questions": questions}
    )
    last_error: Exception | None = None
    overflow_shrinks = 0
    attempt = 0

    while attempt < retries:
        try:
            response = session.post(
                OPENROUTER_DECISIONS_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": OPENROUTER_HTTP_REFERER,
                    "X-OpenRouter-Title": OPENROUTER_TITLE,
                },
                json=payload,
                timeout=90,
            )
        except Exception as error:
            last_error = error
            retrying = attempt < retries - 1 and _is_spotty_internet_error(error)
            _print_openrouter_error(error, attempt=attempt, retries=retries, retrying=retrying)
            if retrying:
                time.sleep(1)
                attempt += 1
                continue
            break

        if response.status_code == 402:
            raise RuntimeError("OpenRouter payment or credit limit reached (402)")

        if not response.ok:
            detail = _response_detail(response)
            last_error = RuntimeError(f"OpenRouter returned {response.status_code}: {detail}")
            if response.status_code == 400 and _state_too_long(detail):
                previous_chars = len(payload["state"]) if isinstance(payload.get("state"), str) else 0
                if overflow_shrinks < MAX_OVERFLOW_SHRINKS and _shrink_overflow_payload(payload):
                    overflow_shrinks += 1
                    next_chars = len(payload["state"]) if isinstance(payload.get("state"), str) else 0
                    print(
                        f"[WARN] OpenRouter max_tokens_exceeded; "
                        f"shrunk state {previous_chars} → {next_chars} chars and retrying",
                        file=sys.stderr,
                    )
                    continue
                _print_openrouter_error(
                    last_error, attempt=attempt, retries=retries, retrying=False, detail=detail
                )
                break

            retrying = attempt < retries - 1 and _is_spotty_internet_error(
                last_error, response.status_code
            )
            _print_openrouter_error(
                last_error, attempt=attempt, retries=retries, retrying=retrying, detail=detail
            )
            if retrying:
                time.sleep(1)
                attempt += 1
                continue
            break

        try:
            response_data = response.json()
        except ValueError as error:
            last_error = error
            retrying = attempt < retries - 1 and _is_spotty_internet_error(error)
            _print_openrouter_error(
                error, attempt=attempt, retries=retries, retrying=retrying, detail=_response_detail(response)
            )
            if retrying:
                time.sleep(1)
                attempt += 1
                continue
            break

        answers = response_data.get("answers")
        if not isinstance(answers, dict):
            last_error = RuntimeError("OpenRouter returned no decision answers")
            retrying = attempt < retries - 1
            _print_openrouter_error(
                last_error,
                attempt=attempt,
                retries=retries,
                retrying=retrying,
                detail=_response_detail(response),
            )
            if retrying:
                time.sleep(1)
                attempt += 1
                continue
            break
        return answers, _usage_input_tokens(response_data)

    if last_error is None:
        raise RuntimeError("OpenRouter request failed with no error detail")
    raise RuntimeError(f"OpenRouter request failed: {last_error}") from last_error


def _best_choice(answer: dict[str, Any], excluded: str = NOT_IN_LIST) -> tuple[str, float]:
    """Return the highest-probability non-fallback choice."""
    probabilities = answer.get("probabilities") or {}
    candidates = {key: float(value) for key, value in probabilities.items() if key != excluded}
    if candidates:
        return max(candidates.items(), key=lambda item: item[1])
    choice = str(answer.get("choice", "unknown"))
    return choice, float(answer.get("confidence", 0.0) or 0.0)


def _error_result(message: str, jev_text: str = "") -> dict[str, Any]:
    """Build a consistent failed classification result."""
    print(f"[ERROR] {message}", file=sys.stderr)
    return {
        "form_type": "unknown",
        "document_kind": "unknown",
        "confidence": 0.0,
        "form_confidence": 0.0,
        "kind_confidence": 0.0,
        "gated": False,
        "calls": 0,
        "input_tokens": 0,
        "probabilities": {},
        "jev_text": jev_text,
        "error": message,
        "second_opinion": None,
    }


def _empty_second_opinion(*, error: str | None = None) -> dict[str, Any]:
    """Build a second-opinion object with optional error and empty fields."""
    return {
        "form_type": "",
        "document_kind": "",
        "included_forms": [],
        "rationale": "",
        "model": SECOND_PASS_MODEL,
        "error": error,
    }


def _chat_content_text(content: Any) -> str:
    """Normalize chat message content into a single string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
            else:
                text = getattr(item, "text", None)
                if text:
                    parts.append(str(text))
        return "".join(parts)
    return "" if content is None else str(content)


def _mercury_response_text(response: Any) -> str:
    """Read the first chat completion message from an OpenRouter SDK response."""
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices")
    if not choices:
        raise RuntimeError("Mercury returned no choices")
    first = choices[0]
    message = getattr(first, "message", None)
    if message is None and isinstance(first, dict):
        message = first.get("message")
    content = getattr(message, "content", None) if message is not None else None
    if content is None and isinstance(message, dict):
        content = message.get("content")
    text = _chat_content_text(content).strip()
    if not text:
        raise RuntimeError("Mercury returned empty content")
    return text


def _parse_second_opinion(raw: str) -> dict[str, Any]:
    """Parse Mercury JSON into the attached second-opinion object."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Mercury JSON was not an object")
    included = data.get("included_forms") or []
    if not isinstance(included, list):
        included = [str(included)]
    return {
        "form_type": str(data.get("form_type") or ""),
        "document_kind": str(data.get("document_kind") or ""),
        "included_forms": [str(item) for item in included],
        "rationale": str(data.get("rationale") or ""),
        "model": SECOND_PASS_MODEL,
        "error": None,
    }


def _ask_mercury_second_opinion(api_key: str, jev_text: str) -> dict[str, Any]:
    """Request a structured Mercury 2.5 second opinion for low Jev confidence."""
    from openrouter import OpenRouter

    prompt = (
        "Identify the tax form from the following text. Fill the JSON schema with the IRS "
        "form number/title, a short document kind, component forms if this is a composite "
        "(otherwise an empty list), and a brief rationale.\n\n"
        f"{_sanitize_openrouter_text(jev_text)}"
    )
    with OpenRouter(
        api_key=api_key,
        http_referer=OPENROUTER_HTTP_REFERER,
        x_open_router_title=OPENROUTER_TITLE,
    ) as client:
        response = client.chat.send(
            model=SECOND_PASS_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            provider={"only": ["Inception"], "require_parameters": True},
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "tax_form_second_opinion",
                    "strict": True,
                    "schema": SECOND_OPINION_SCHEMA,
                },
            },
        )
    return _parse_second_opinion(_mercury_response_text(response))


def _attach_second_opinion(
    result: dict[str, Any],
    api_key: str,
    status_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run Mercury when Jev form confidence is below the second-pass threshold."""
    form_confidence = float(result.get("form_confidence", 0.0) or 0.0)
    jev_text = result.get("jev_text") or ""
    if form_confidence >= SECOND_PASS_THRESHOLD or not jev_text:
        result["second_opinion"] = None
        return result
    if status_callback:
        status_callback("Jev form confidence below 90%, requesting Mercury 2.5")
    try:
        result["second_opinion"] = _ask_mercury_second_opinion(api_key, jev_text)
    except Exception as error:
        print(f"[ERROR] Mercury second opinion failed: {error!r}", file=sys.stderr)
        traceback.print_exception(type(error), error, error.__traceback__, file=sys.stderr)
        result["second_opinion"] = _empty_second_opinion(error=str(error))
    return result


def classify_tax_document(
    text: str,
    api_key: str | None = None,
    criteria: dict[str, dict[str, Any]] | None = None,
    gate: float = CONFIDENCE_THRESHOLD,
    session: requests.Session | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Classify the combined text extracted from one PDF."""
    resolved_api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
    if not resolved_api_key:
        return _error_result("OPENROUTER_API_KEY is not set")

    lines = _lines_of(text)
    if _is_blank(lines):
        return {
            "form_type": "blank",
            "document_kind": "blank",
            "confidence": 1.0,
            "form_confidence": 1.0,
            "kind_confidence": 1.0,
            "gated": True,
            "calls": 0,
            "input_tokens": 0,
            "probabilities": {"form": {"blank": 1.0}, "kind": {"blank": 1.0}},
            "jev_text": "",
            "error": None,
            "second_opinion": None,
        }

    form_criteria = criteria or load_form_criteria()
    http_session = session or requests.Session()
    state = truncate_state(_sanitize_openrouter_text(text))

    try:
        first_request = build_classification_request(text, form_criteria)
        answers, input_tokens = _ask_openrouter(
            resolved_api_key,
            state,
            first_request["questions"],
            http_session,
        )
        if "form" not in answers or "kind" not in answers:
            raise RuntimeError("OpenRouter response is missing form or kind answers")

        form_type, form_probability = _best_choice(answers["form"])
        step_confidences = [form_probability]
        calls = 1
        probabilities: dict[str, Any] = {
            "form": answers["form"].get("probabilities", {}),
            "kind": answers["kind"].get("probabilities", {}),
        }

        if form_type in RARE_PARENTS:
            sub_answers, sub_tokens = _ask_openrouter(
                resolved_api_key,
                state,
                {
                    "sub": {
                        "type": "choice",
                        "instructions": (
                            "Which specific form or schedule is this document from? Read the form number "
                            "throughout the text; instructions belong to the form they instruct."
                        ),
                        "criteria": _family_criteria(form_type, form_criteria),
                    }
                },
                http_session,
            )
            if "sub" not in sub_answers:
                raise RuntimeError("OpenRouter response is missing the form-family answer")
            form_type, sub_probability = _best_choice(sub_answers["sub"])
            step_confidences.append(sub_probability)
            probabilities["sub"] = sub_answers["sub"].get("probabilities", {})
            calls += 1
            input_tokens += sub_tokens

        form_confidence = min(step_confidences)
        kind_answer = answers["kind"]
        result = {
            "form_type": form_type,
            "document_kind": kind_answer.get("choice", "unknown"),
            "confidence": form_confidence,
            "form_confidence": form_confidence,
            "kind_confidence": float(kind_answer.get("confidence", 0.0) or 0.0),
            "gated": form_confidence >= gate,
            "calls": calls,
            "input_tokens": input_tokens,
            "probabilities": probabilities,
            "jev_text": state,
            "error": None,
        }
    except (RuntimeError, requests.RequestException, ValueError) as error:
        result = _error_result(str(error), jev_text=state)
    return _attach_second_opinion(result, resolved_api_key, status_callback)
