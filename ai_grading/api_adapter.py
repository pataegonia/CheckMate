from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


LOGGER = logging.getLogger(__name__)

IDEMPOTENT_4XX_CODES = {400, 404, 409}


def from_be_invoke_event(event: dict[str, Any]) -> dict[str, Any]:
    """Adapt the BE→Lambda invoke payload into a GradingRequest-compatible dict.

    BE payload shape (agreed):
        {
          "submission": {"submissionId": 10, "image": {"bucket": "...", "key": "..."}},
          "assignment": {"assignmentId": 1, "answer": {...}},
          "questions": [{"questionId", "orderNum", "type", "expectedAnswer",
                         "maxScore", "gradingCriteria"}],
          "callback": {"url": "..."}
        }
    """
    submission = event.get("submission") or {}
    assignment = event.get("assignment") or {}
    image = submission.get("image") or {}
    questions_in = event.get("questions") or []

    return {
        "jobType": event.get("jobType", "GRADE"),
        "submissionId": submission.get("submissionId"),
        "assignmentId": assignment.get("assignmentId"),
        "studentId": submission.get("studentId", "unknown"),
        "s3": {
            "bucket": image.get("bucket"),
            "key": image.get("key"),
            "contentType": image.get("contentType", "image/jpeg"),
        } if image.get("bucket") and image.get("key") else None,
        "questions": [_question_to_internal(q) for q in questions_in],
    }


def _question_to_internal(question: dict[str, Any]) -> dict[str, Any]:
    required = ("questionId", "orderNum", "expectedAnswer", "maxScore")
    missing = [key for key in required if key not in question or question[key] is None]
    if missing:
        raise ValueError(f"BE question payload is missing required fields: {', '.join(missing)}")
    return {
        "questionId": question["questionId"],
        "questionNumber": question["orderNum"],
        "type": question.get("type", "short_answer"),
        "questionContent": question.get("questionContent", ""),
        "answer": question["expectedAnswer"],
        "maxScore": question["maxScore"],
        "rubric": question.get("gradingCriteria", question.get("rubric", "")),
        "imageCrop": question.get("imageCrop"),
    }


def to_be_grade_payload(output: dict[str, Any]) -> dict[str, Any]:
    """Convert our internal GradingOutput dict into the BE callback body."""
    status = output.get("status", "FAILED")
    if status != "DONE":
        return to_be_fail_payload(output.get("failReason") or "AI 채점 실패")

    questions = []
    for item in output.get("questions", []):
        questions.append(
            {
                "questionId": _coerce_long(item.get("questionId")),
                "score": _round_score(item.get("score", 0)),
                "reason": _compose_reason(item),
                "imageUrl": item.get("imageUrl"),
            }
        )
    return {
        "status": "DONE",
        "totalScore": _round_score(output.get("totalScore", 0)),
        "gradedAt": _iso_now(),
        "questions": questions,
    }


def to_be_fail_payload(fail_reason: str) -> dict[str, Any]:
    return {
        "status": "FAILED",
        "failReason": fail_reason or "AI 채점 실패",
        "gradedAt": _iso_now(),
    }


def post_callback(
    *,
    url: str,
    payload: dict[str, Any],
    internal_token: str,
    timeout_seconds: float = 30.0,
) -> tuple[int, str]:
    """POST result to BE. Returns (status_code, raw_body).

    Treats 4xx in IDEMPOTENT_4XX_CODES as 'do not retry' — caller should
    log and consider the job done. Raises on 5xx / network errors so the
    Lambda runtime retries.
    """
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "X-Internal-Token": internal_token,
        "Accept": "application/json",
    }
    request = Request(url, data=body, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return response.status, response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        response_headers = dict(exc.headers) if exc.headers else {}
        LOGGER.error(
            "BE callback %s returned HTTP %s. body=%r headers=%r",
            url, exc.code, detail, response_headers,
        )
        if exc.code in IDEMPOTENT_4XX_CODES:
            return exc.code, detail
        raise RuntimeError(f"BE callback {url} failed: HTTP {exc.code} {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"BE callback {url} unreachable: {exc.reason}") from exc


def _compose_reason(item: dict[str, Any]) -> str:
    """Combine reason + feedbackForStudent into a single BE field.

    Matches the BE-agreed format:
        [채점 결과 한 줄 요약]
        [감점 사유 (필요 시)]
        [학생 피드백 / 풀이 힌트 (선택)]
    """
    parts: list[str] = []
    reason = (item.get("reason") or "").strip()
    if reason:
        parts.append(reason)

    feedback = (item.get("feedbackForStudent") or "").strip()
    if feedback and feedback != reason:
        parts.append(feedback)

    mistake = (item.get("mistakeType") or "").strip()
    if mistake:
        parts.append(f"오류 유형: {mistake}")

    return "\n".join(parts) if parts else "채점 근거가 비어 있습니다."


def _coerce_long(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"questionId must be coercible to Long, got {value!r}") from exc


def _round_score(value: Any) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"score must be numeric, got {value!r}") from exc


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0, tzinfo=None).isoformat()
