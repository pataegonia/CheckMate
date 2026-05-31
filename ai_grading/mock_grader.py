from __future__ import annotations

from .result_validator import build_grade_output, build_regrade_output
from .schemas import (
    GradingRequest,
    QuestionGradeResult,
    QuestionRegradeResult,
    QuestionResultLabel,
)


class MockGrader:
    def grade(self, request: GradingRequest):
        questions = [
            QuestionGradeResult(
                questionId=question.question_id,
                result=QuestionResultLabel.CORRECT,
                score=question.max_score,
                maxScore=question.max_score,
                confidence=0.95,
                reason="Mock mode에서 생성된 채점 결과입니다.",
                feedbackForStudent="풀이와 최종 답이 정답 기준과 일치하는 것으로 처리되었습니다.",
                detectedAnswer=question.answer,
                mistakeType=None,
                needsManualReview=False,
            )
            for question in request.questions
        ]
        return build_grade_output(request, questions)

    def regrade(self, request: GradingRequest):
        previous_by_question_id = _previous_by_question_id(request.previous_result)
        questions = []
        for question in request.questions:
            previous = previous_by_question_id.get(str(question.question_id), {})
            previous_score = previous.get("score", previous.get("aiRegradedScore", question.max_score))
            questions.append(
                QuestionRegradeResult(
                    questionId=question.question_id,
                    aiRegradedScore=max(0, min(int(previous_score), question.max_score)),
                    maxScore=question.max_score,
                    reason="Mock mode에서 이전 점수를 유지하는 재채점 결과입니다.",
                    confidence=0.95,
                    needsManualReview=False,
                )
            )
        return build_regrade_output(request, questions)


def _previous_by_question_id(previous_result):
    if isinstance(previous_result, dict):
        items = previous_result.get("questions", [])
    elif isinstance(previous_result, list):
        items = previous_result
    else:
        items = []
    return {str(item.get("questionId")): item for item in items if isinstance(item, dict)}
