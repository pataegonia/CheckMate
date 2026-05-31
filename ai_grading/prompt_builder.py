from __future__ import annotations

import json
from typing import Any

from .schemas import GradingRequest, QuestionInput


SYSTEM_PROMPT = """
당신은 수학 서술형 풀이를 채점하는 AI 채점 보조자입니다.
학생 풀이 이미지를 읽고 문항별 풀이 과정을 분석하세요.
교사가 제공한 정답과 채점 기준을 최우선으로 따르세요.
최종 답만 보지 말고 풀이 과정의 논리적 타당성을 평가하세요.
부분점수가 가능하면 채점 기준에 따라 부분점수를 부여하세요.
확실하지 않은 경우 needsManualReview=true로 표시하세요.
모든 점수는 0 이상 maxScore 이하의 정수여야 합니다.
출력은 반드시 JSON만 반환하세요.
Markdown, 설명 문장, 코드블록을 출력하지 마세요.
""".strip()


def build_grade_prompt(
    request: GradingRequest,
    question: QuestionInput,
    *,
    crop_metadata: dict[str, Any] | None = None,
) -> str:
    metadata = crop_metadata or {}
    return f"""
아래 한 문항만 채점하세요. 첨부 이미지는 전체 페이지가 아니라 해당 문항 crop입니다.

과제명: {request.assignment_title or ""}
학생명: {request.student_name or ""}
submissionId: {request.submission_id}
assignmentId: {request.assignment_id}

문항 정보:
- questionId: {question.question_id}
- questionNumber: {question.question_number}
- questionType: {question.question_type}
- questionContent: {question.question_content}
- teacherAnswer: {question.answer}
- maxScore: {question.max_score}
- rubric: {question.rubric or "별도 채점 기준 없음"}

내부 OCR/문항 추출 참고 정보:
{json.dumps(metadata, ensure_ascii=False, indent=2)}

반드시 아래 JSON 객체 하나만 반환하세요.
{{
  "questionId": {json.dumps(question.question_id, ensure_ascii=False)},
  "result": "CORRECT | PARTIAL | WRONG",
  "score": 0,
  "maxScore": {question.max_score},
  "confidence": 0.0,
  "reason": "교사용 채점 근거를 한국어로 간단히 작성",
  "feedbackForStudent": "학생에게 보여줄 피드백을 한국어로 작성",
  "detectedAnswer": "이미지에서 읽은 학생 최종 답",
  "mistakeType": null,
  "needsManualReview": false
}}

규칙:
- result는 CORRECT, PARTIAL, WRONG 중 하나만 사용하세요.
- score는 0 이상 {question.max_score} 이하의 정수만 사용하세요.
- 풀이 과정이 일부 타당하면 rubric에 따라 PARTIAL을 줄 수 있습니다.
- 이미지가 흐리거나 풀이 판독이 불확실하면 confidence를 낮추고 needsManualReview=true로 두세요.
- maxScore는 반드시 {question.max_score}로 반환하세요.
""".strip()


def build_regrade_prompt(
    request: GradingRequest,
    question: QuestionInput,
    *,
    previous_question_result: dict[str, Any] | None,
    crop_metadata: dict[str, Any] | None = None,
) -> str:
    metadata = crop_metadata or {}
    return f"""
아래 한 문항의 이전 AI 채점 결과를 재검토하세요. 첨부 이미지는 해당 문항 crop입니다.
최종 확정은 교사가 하므로, 이 모듈은 재검토 점수와 사유만 제안합니다.

재채점 요청 사유:
{request.request_reason or "별도 사유 없음"}

문항 정보:
- questionId: {question.question_id}
- questionNumber: {question.question_number}
- questionType: {question.question_type}
- questionContent: {question.question_content}
- teacherAnswer: {question.answer}
- maxScore: {question.max_score}
- rubric: {question.rubric or "별도 채점 기준 없음"}

이전 채점 결과:
{json.dumps(previous_question_result or {}, ensure_ascii=False, indent=2)}

내부 OCR/문항 추출 참고 정보:
{json.dumps(metadata, ensure_ascii=False, indent=2)}

반드시 아래 JSON 객체 하나만 반환하세요.
{{
  "questionId": {json.dumps(question.question_id, ensure_ascii=False)},
  "aiRegradedScore": 0,
  "maxScore": {question.max_score},
  "reason": "이전 채점이 타당한지, 점수를 유지/수정하는 이유를 한국어로 작성",
  "confidence": 0.0,
  "needsManualReview": false
}}

규칙:
- aiRegradedScore는 0 이상 {question.max_score} 이하의 정수만 사용하세요.
- 이전 점수가 타당하면 그대로 유지하세요.
- 판단이 불확실하거나 교사 확인이 필요하면 needsManualReview=true로 두세요.
""".strip()
