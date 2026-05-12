from __future__ import annotations

import io
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol


DEFAULT_CONFIDENCE_THRESHOLD = 0.75


def normalize_answer(value: Any) -> str:
    text = str(value or "").strip().lower()
    replacements = {
        "−": "-",
        "―": "-",
        "–": "-",
        "—": "-",
        "﹣": "-",
        "－": "-",
        "＜": "<",
        "〈": "<",
        "‹": "<",
        "≤": "<=",
        "≦": "<=",
        "＞": ">",
        "〉": ">",
        "›": ">",
        "≥": ">=",
        "≧": ">=",
        "＝": "=",
        "×": "x",
        "χ": "x",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = re.sub(r"7\s*[∠l]", "x<", text)
    text = re.sub(r"7\s*>", "x>", text)
    text = re.sub(r"\s+", "", text)
    text = text.strip(".,;:()[]{}")
    return text


def _json_safe(value: Any) -> Any:
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@dataclass
class QuestionInput:
    question_id: str
    problem_num: int
    expected_answer: str
    student_answer: str
    ocr_confidence: float
    ocr_source: str
    is_correct_by_rule: bool
    question_type: str = "short_answer"
    grading_criteria: str = ""
    candidates: list[dict[str, Any]] = field(default_factory=list)
    image_bytes: bytes | None = field(default=None, repr=False)
    image_format: str = "png"
    answer_image_bytes: bytes | None = field(default=None, repr=False)
    answer_image_format: str = "png"


@dataclass
class QuestionGrade:
    question_id: str
    problem_num: int
    question_type: str
    is_correct: bool
    score: float
    max_score: float
    reason: str
    confidence: float
    source: str
    needs_review: bool
    extracted_answer: str = ""
    deduction: str = ""


@dataclass
class SubmissionPayload:
    submission_id: str
    assignment_id: str
    student_id: str
    created_at: str
    questions: list[QuestionInput]


@dataclass
class SubmissionGrade:
    submission_id: str
    assignment_id: str
    student_id: str
    total_score: float
    max_score: float
    questions: list[QuestionGrade]


class AIGrader(Protocol):
    def grade_submission(self, payload: SubmissionPayload) -> SubmissionGrade:
        ...


def current_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def needs_ai_review(
    question: QuestionInput,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> bool:
    if question.question_type == "descriptive":
        return True
    if question.question_type == "short_answer" and not question.is_correct_by_rule:
        return True
    if not question.student_answer:
        return True
    if question.ocr_source in {"none", "v1-fallback"}:
        return True
    if question.ocr_confidence < 0:
        return True
    if question.ocr_confidence < confidence_threshold:
        return True
    return False


def _encode_png_bytes(image: Any) -> bytes | None:
    if image is None:
        return None
    try:
        from PIL import Image

        if isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        elif hasattr(image, "shape"):
            pil_image = Image.fromarray(image).convert("RGB")
        else:
            return None

        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


def build_submission_payload(
    results: list[dict[str, Any]],
    *,
    submission_id: str = "local-demo-submission",
    assignment_id: str = "local-demo-assignment",
    student_id: str = "local-demo-student",
    grading_criteria_by_problem: dict[int, str] | None = None,
    question_specs: dict[int, dict[str, Any]] | None = None,
) -> SubmissionPayload:
    grading_criteria_by_problem = grading_criteria_by_problem or {}
    question_specs = question_specs or {}
    questions: list[QuestionInput] = []

    for result in results:
        problem_num = int(result["problem_num"])
        spec = question_specs.get(problem_num, {})
        student_answer = result.get("student_answer_override") or result.get("student_answer", "")
        correct_answer = str(result.get("correct_answer", spec.get("answer", "")))
        question_id = str(result.get("question_id") or problem_num)
        question_type = str(result.get("question_type") or spec.get("type", "short_answer"))
        grading_criteria = str(
            result.get("grading_criteria")
            or grading_criteria_by_problem.get(problem_num, "")
            or spec.get("rubric", "")
        )

        questions.append(
            QuestionInput(
                question_id=question_id,
                problem_num=problem_num,
                expected_answer=correct_answer,
                student_answer=str(student_answer or ""),
                ocr_confidence=float(result.get("confidence", -1.0)),
                ocr_source=str(result.get("source", "none")),
                is_correct_by_rule=(
                    normalize_answer(student_answer) == normalize_answer(correct_answer)
                ),
                question_type=question_type,
                grading_criteria=grading_criteria,
                candidates=_json_safe(result.get("candidates", [])),
                image_bytes=_encode_png_bytes(result.get("crop_rgb")),
                image_format="png",
                answer_image_bytes=_encode_png_bytes(result.get("answer_crop_rgb")),
                answer_image_format="png",
            )
        )

    return SubmissionPayload(
        submission_id=submission_id,
        assignment_id=assignment_id,
        student_id=student_id,
        created_at=current_timestamp(),
        questions=questions,
    )


class MockAIGrader:
    def __init__(
        self,
        *,
        max_score_per_question: float = 1.0,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        self.max_score_per_question = max_score_per_question
        self.confidence_threshold = confidence_threshold

    def grade_question(self, question: QuestionInput) -> QuestionGrade:
        review = needs_ai_review(question, self.confidence_threshold)
        is_correct = question.is_correct_by_rule
        score = self.max_score_per_question if is_correct else 0.0
        confidence = question.ocr_confidence if question.ocr_confidence >= 0 else 0.0

        if question.question_type == "descriptive":
            reason = (
                "서술형 문항입니다. 실제 Bedrock/Claude 연동 시 풀이 과정과 "
                "채점 기준을 함께 검토해야 합니다."
            )
            source = "mock-ai-review"
        elif review:
            reason = (
                "OCR 신뢰도가 낮거나 답안 후보가 불확실하여 AI 재검토 대상으로 표시했습니다."
            )
            source = "mock-ai-review"
        elif is_correct:
            reason = "OCR로 추출한 답안이 저장된 정답과 일치합니다."
            source = "rule"
        else:
            reason = "OCR로 추출한 답안이 저장된 정답과 일치하지 않습니다."
            source = "rule"

        return QuestionGrade(
            question_id=question.question_id,
            problem_num=question.problem_num,
            question_type=question.question_type,
            is_correct=is_correct,
            score=score,
            max_score=self.max_score_per_question,
            reason=reason,
            confidence=confidence,
            source=source,
            needs_review=review,
            extracted_answer=question.student_answer,
            deduction="" if is_correct else "정답 불일치",
        )

    def grade_submission(self, payload: SubmissionPayload) -> SubmissionGrade:
        grades = [self.grade_question(question) for question in payload.questions]
        return SubmissionGrade(
            submission_id=payload.submission_id,
            assignment_id=payload.assignment_id,
            student_id=payload.student_id,
            total_score=sum(grade.score for grade in grades),
            max_score=sum(grade.max_score for grade in grades),
            questions=grades,
        )


def _coerce_question_grade(
    question: QuestionInput,
    raw: dict[str, Any],
    *,
    source: str,
    max_score_per_question: float,
) -> QuestionGrade:
    is_correct = bool(raw.get("isCorrect", raw.get("is_correct", False)))
    score = float(raw.get("score", max_score_per_question if is_correct else 0.0))
    max_score = float(raw.get("maxScore", raw.get("max_score", max_score_per_question)))
    confidence = float(raw.get("confidence", 0.0))

    return QuestionGrade(
        question_id=str(raw.get("questionId", raw.get("question_id", question.question_id))),
        problem_num=int(raw.get("problemNum", raw.get("problem_num", question.problem_num))),
        question_type=str(raw.get("type", raw.get("questionType", question.question_type))),
        is_correct=is_correct,
        score=max(0.0, min(score, max_score)),
        max_score=max_score,
        reason=str(raw.get("reason", "AI 채점 결과입니다.")),
        confidence=max(0.0, min(confidence, 1.0)),
        source=source,
        needs_review=bool(raw.get("needsReview", raw.get("needs_review", False))),
        extracted_answer=str(
            raw.get(
                "extractedAnswer",
                raw.get("extracted_answer", question.student_answer),
            )
        ),
        deduction=str(raw.get("deduction", "")),
    )


class BedrockConverseGrader:
    """Bedrock Converse API adapter.

    Credentials are resolved by boto3, so keep them outside the repository
    using `aws configure`, environment variables, or an AWS role.
    """

    def __init__(
        self,
        *,
        model_id: str | None = None,
        region_name: str | None = None,
        profile_name: str | None = None,
        max_score_per_question: float = 1.0,
        max_tokens: int = 700,
        temperature: float = 0.0,
    ) -> None:
        self.model_id = model_id or os.getenv("BEDROCK_MODEL_ID")
        self.region_name = (
            region_name
            or os.getenv("AWS_REGION")
            or os.getenv("AWS_DEFAULT_REGION")
            or "us-east-1"
        )
        self.profile_name = profile_name or os.getenv("AWS_PROFILE")
        self.max_score_per_question = max_score_per_question
        self.max_tokens = max_tokens
        self.temperature = temperature

        if not self.model_id:
            raise ValueError("Set BEDROCK_MODEL_ID or pass model_id to BedrockConverseGrader.")

    def _client(self):
        import boto3

        if self.profile_name:
            session = boto3.Session(profile_name=self.profile_name, region_name=self.region_name)
        else:
            session = boto3.Session(region_name=self.region_name)
        return session.client("bedrock-runtime")

    def grade_question(self, question: QuestionInput) -> QuestionGrade:
        prompt = build_bedrock_grading_prompt(question)
        content = [{"text": prompt}]
        if question.image_bytes:
            content.insert(
                0,
                {
                    "image": {
                        "format": question.image_format,
                        "source": {"bytes": question.image_bytes},
                    }
                },
            )
        if question.answer_image_bytes:
            content.insert(
                1 if question.image_bytes else 0,
                {
                    "image": {
                        "format": question.answer_image_format,
                        "source": {"bytes": question.answer_image_bytes},
                    }
                },
            )
        response = self._client().converse(
            modelId=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
            inferenceConfig={
                "maxTokens": self.max_tokens,
                "temperature": self.temperature,
            },
        )

        blocks = response.get("output", {}).get("message", {}).get("content", [])
        text = "\n".join(block.get("text", "") for block in blocks if "text" in block).strip()
        try:
            raw_grade = parse_ai_json_response(text)
        except ValueError as e:
            return QuestionGrade(
                question_id=question.question_id,
                problem_num=question.problem_num,
                question_type=question.question_type,
                is_correct=False,
                score=0.0,
                max_score=self.max_score_per_question,
                reason=f"AI 응답을 JSON으로 해석하지 못했습니다: {e}",
                confidence=0.0,
                source="bedrock-parse-error",
                needs_review=True,
                extracted_answer=question.student_answer,
                deduction="AI 응답 파싱 실패",
            )
        return _coerce_question_grade(
            question,
            raw_grade,
            source="bedrock",
            max_score_per_question=self.max_score_per_question,
        )

    def grade_submission(self, payload: SubmissionPayload) -> SubmissionGrade:
        grades = [self.grade_question(question) for question in payload.questions]
        return SubmissionGrade(
            submission_id=payload.submission_id,
            assignment_id=payload.assignment_id,
            student_id=payload.student_id,
            total_score=sum(grade.score for grade in grades),
            max_score=sum(grade.max_score for grade in grades),
            questions=grades,
        )


class HybridAIGrader:
    """Use local rule grading for easy questions and Bedrock for selected questions."""

    def __init__(
        self,
        *,
        remote_grader: BedrockConverseGrader | None = None,
        local_grader: MockAIGrader | None = None,
        mode: str = "review_needed",
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        self.remote_grader = remote_grader or BedrockConverseGrader()
        self.local_grader = local_grader or MockAIGrader(
            confidence_threshold=confidence_threshold
        )
        self.mode = mode
        self.confidence_threshold = confidence_threshold

    def should_use_remote(self, question: QuestionInput) -> bool:
        if self.mode == "all":
            return True
        if self.mode == "short_descriptive":
            return (
                question.question_type in {"short_answer", "descriptive"}
                or needs_ai_review(question, self.confidence_threshold)
            )
        if self.mode == "review_incorrect":
            return question.question_type == "descriptive" or not question.is_correct_by_rule
        return needs_ai_review(question, self.confidence_threshold)

    def grade_submission(self, payload: SubmissionPayload) -> SubmissionGrade:
        grades: list[QuestionGrade] = []
        for question in payload.questions:
            if self.should_use_remote(question):
                grades.append(self.remote_grader.grade_question(question))
            else:
                grades.append(self.local_grader.grade_question(question))

        return SubmissionGrade(
            submission_id=payload.submission_id,
            assignment_id=payload.assignment_id,
            student_id=payload.student_id,
            total_score=sum(grade.score for grade in grades),
            max_score=sum(grade.max_score for grade in grades),
            questions=grades,
        )


def build_bedrock_grading_prompt(question: QuestionInput) -> str:
    if question.question_type == "multiple_choice":
        task = (
            "This is a multiple-choice question. Inspect the image if provided and identify "
            "the selected option from circles, check marks, or written choice numbers. "
            "Grade by comparing the selected option with the expected answer."
        )
    elif question.question_type == "descriptive":
        task = (
            "This is a descriptive math question. Inspect the full solution process in the "
            "image if provided. Grade the reasoning, intermediate steps, final answer, and "
            "deductions according to the grading criteria. Do not grade only by the final answer."
        )
    else:
        task = (
            "This is a short-answer math question. The image may contain several lines of "
            "scratch work or intermediate calculations. Find the final answer first, then "
            "compare it with the expected answer. Ignore intermediate values unless they are "
            "clearly marked as the final answer."
        )

    return f"""
You are an AI math grading assistant for CheckMate.
Grade the student's answer using the expected answer, problem type, image, OCR output, and grading criteria.
Return exactly one valid JSON object. Do not include markdown, comments, explanations, or code fences.
The first character of your response must be `{{` and the last character must be `}}`.

JSON schema:
{{
  "questionId": "{question.question_id}",
  "problemNum": {question.problem_num},
  "type": "{question.question_type}",
  "extractedAnswer": "the final answer found from OCR text or image",
  "isCorrect": true,
  "score": 1.0,
  "maxScore": 1.0,
  "reason": "short grading reason in Korean",
  "deduction": "deduction reason in Korean, empty if none",
  "confidence": 0.0,
  "needsReview": false
}}

Task: {task}
Problem type: {question.question_type}
Expected answer: {question.expected_answer}
Student OCR answer: {question.student_answer}
OCR confidence: {question.ocr_confidence}
OCR source: {question.ocr_source}
Grading criteria: {question.grading_criteria or "No extra grading criteria."}
OCR candidates: {json.dumps(question.candidates, ensure_ascii=False)}

Rules:
- Write Korean in `reason` and `deduction`.
- If two images are provided, the first is the full question crop and the second is an enlarged answer-focused crop.
- If the image and OCR disagree, prefer the image and explain briefly.
- If the final answer is unclear, set `needsReview` to true and lower confidence.
- For short-answer questions, extract only the final answer, not every intermediate calculation.
- For descriptive questions, award partial credit according to the grading criteria.
""".strip()


def dump_json(data: Any) -> str:
    if hasattr(data, "__dataclass_fields__"):
        data = asdict(data)
    return json.dumps(_json_safe(data), ensure_ascii=False, indent=2)


def submission_grade_to_api_dict(grade: SubmissionGrade) -> dict[str, Any]:
    return {
        "submissionId": grade.submission_id,
        "assignmentId": grade.assignment_id,
        "studentId": grade.student_id,
        "totalScore": grade.total_score,
        "maxScore": grade.max_score,
        "questions": [
            {
                "questionId": question.question_id,
                "problemNum": question.problem_num,
                "type": question.question_type,
                "extractedAnswer": question.extracted_answer,
                "isCorrect": question.is_correct,
                "score": question.score,
                "maxScore": question.max_score,
                "reason": question.reason,
                "confidence": question.confidence,
                "source": question.source,
                "needsReview": question.needs_review,
                "deduction": question.deduction,
            }
            for question in grade.questions
        ],
    }


def dump_api_json(grade: SubmissionGrade) -> str:
    return json.dumps(submission_grade_to_api_dict(grade), ensure_ascii=False, indent=2)


def parse_ai_json_response(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if not text:
        raise ValueError("empty Bedrock response")
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            snippet = text[:160].replace("\n", " ")
            raise ValueError(f"no JSON object found in response: {snippet}") from None
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError as e:
            snippet = text[start : min(end + 1, start + 160)].replace("\n", " ")
            raise ValueError(f"invalid JSON object: {snippet}") from e
