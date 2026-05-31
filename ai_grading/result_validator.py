from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .schemas import (
    GradeSummary,
    GradingOutput,
    GradingRequest,
    GradingStatus,
    QuestionGradeResult,
    QuestionInput,
    QuestionRegradeResult,
    QuestionResultLabel,
    RegradeOutput,
)


DEFAULT_CONFIDENCE_THRESHOLD = 0.75


def validate_grade_question(
    question: QuestionInput,
    raw: dict[str, Any],
    *,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> QuestionGradeResult:
    max_score = question.max_score
    score = _coerce_int(raw.get("score", 0), default=0)
    score = _clamp(score, 0, max_score)
    result = _coerce_result(raw, score=score, max_score=max_score)
    confidence = _clamp_float(raw.get("confidence", 0.0), 0.0, 1.0)
    needs_manual_review = bool(raw.get("needsManualReview", raw.get("needs_manual_review", False)))
    if confidence < confidence_threshold:
        needs_manual_review = True

    return QuestionGradeResult(
        questionId=question.question_id,
        result=result,
        score=score,
        maxScore=max_score,
        confidence=confidence,
        reason=str(raw.get("reason") or "AI 채점 근거가 비어 있습니다."),
        feedbackForStudent=str(
            raw.get("feedbackForStudent")
            or raw.get("feedback_for_student")
            or "풀이를 다시 확인해 주세요."
        ),
        detectedAnswer=str(raw.get("detectedAnswer") or raw.get("detected_answer") or ""),
        mistakeType=raw.get("mistakeType", raw.get("mistake_type")),
        needsManualReview=needs_manual_review,
    )


def validate_regrade_question(
    question: QuestionInput,
    raw: dict[str, Any],
    *,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> QuestionRegradeResult:
    max_score = question.max_score
    score = _coerce_int(raw.get("aiRegradedScore", raw.get("ai_regraded_score", 0)), default=0)
    score = _clamp(score, 0, max_score)
    confidence = _clamp_float(raw.get("confidence", 0.0), 0.0, 1.0)
    needs_manual_review = bool(raw.get("needsManualReview", raw.get("needs_manual_review", False)))
    if confidence < confidence_threshold:
        needs_manual_review = True

    return QuestionRegradeResult(
        questionId=question.question_id,
        aiRegradedScore=score,
        maxScore=max_score,
        reason=str(raw.get("reason") or "AI 재채점 근거가 비어 있습니다."),
        confidence=confidence,
        needsManualReview=needs_manual_review,
    )


def build_grade_output(
    request: GradingRequest,
    question_results: Iterable[QuestionGradeResult],
) -> GradingOutput:
    questions = list(question_results)
    total_score = sum(item.score for item in questions)
    max_score = sum(item.max_score for item in questions)
    summary = GradeSummary(
        correct=sum(1 for item in questions if item.result == QuestionResultLabel.CORRECT),
        partial=sum(1 for item in questions if item.result == QuestionResultLabel.PARTIAL),
        wrong=sum(1 for item in questions if item.result == QuestionResultLabel.WRONG),
    )
    return GradingOutput(
        submissionId=request.submission_id,
        assignmentId=request.assignment_id,
        status=GradingStatus.DONE,
        totalScore=total_score,
        maxScore=max_score,
        correctRate=round(total_score / max_score, 4) if max_score else 0.0,
        summary=summary,
        questions=questions,
    )


def build_regrade_output(
    request: GradingRequest,
    question_results: Iterable[QuestionRegradeResult],
) -> RegradeOutput:
    return RegradeOutput(
        submissionId=request.submission_id,
        assignmentId=request.assignment_id,
        status=GradingStatus.DONE,
        questions=list(question_results),
    )


def build_failed_grade_output(request: GradingRequest, reason: str) -> GradingOutput:
    return GradingOutput(
        submissionId=request.submission_id,
        assignmentId=request.assignment_id,
        status=GradingStatus.FAILED,
        totalScore=0,
        maxScore=sum(question.max_score for question in request.questions),
        correctRate=0.0,
        summary=GradeSummary(),
        questions=[],
        failReason=reason,
    )


def build_failed_regrade_output(request: GradingRequest, reason: str) -> RegradeOutput:
    return RegradeOutput(
        submissionId=request.submission_id,
        assignmentId=request.assignment_id,
        status=GradingStatus.FAILED,
        questions=[],
        failReason=reason,
    )


def _coerce_result(raw: dict[str, Any], *, score: int, max_score: int) -> QuestionResultLabel:
    value = str(raw.get("result", "")).upper()
    if value in QuestionResultLabel.__members__:
        result = QuestionResultLabel[value]
    elif raw.get("isCorrect") is True or raw.get("is_correct") is True:
        result = QuestionResultLabel.CORRECT
    elif raw.get("isCorrect") is False or raw.get("is_correct") is False:
        result = QuestionResultLabel.WRONG
    elif max_score > 0 and score >= max_score:
        result = QuestionResultLabel.CORRECT
    elif score > 0:
        result = QuestionResultLabel.PARTIAL
    else:
        result = QuestionResultLabel.WRONG

    if max_score > 0 and score >= max_score:
        return QuestionResultLabel.CORRECT
    if score <= 0:
        return QuestionResultLabel.WRONG
    if result in {QuestionResultLabel.CORRECT, QuestionResultLabel.WRONG}:
        return QuestionResultLabel.PARTIAL
    return result


def _coerce_int(value: Any, *, default: int) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def _clamp_float(value: Any, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = low
    return max(low, min(number, high))
