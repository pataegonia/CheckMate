from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class JobType(str, Enum):
    GRADE = "GRADE"
    REGRADE = "REGRADE"


class GradingStatus(str, Enum):
    DONE = "DONE"
    FAILED = "FAILED"


class QuestionResultLabel(str, Enum):
    CORRECT = "CORRECT"
    PARTIAL = "PARTIAL"
    WRONG = "WRONG"


class S3ImageInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    bucket: str
    key: str
    content_type: str = Field(default="image/jpeg", alias="contentType")


class ImageCropHint(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    x1: int = Field(ge=0)
    y1: int = Field(ge=0)
    x2: int = Field(gt=0)
    y2: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_bounds(self) -> "ImageCropHint":
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("imageCrop must satisfy x2 > x1 and y2 > y1")
        return self


class QuestionInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    question_id: str | int = Field(alias="questionId")
    question_number: int | None = Field(default=None, alias="questionNumber")
    question_type: str = Field(default="short_answer", alias="type")
    question_content: str = Field(default="", alias="questionContent")
    answer: str = ""
    max_score: int = Field(alias="maxScore", ge=0)
    rubric: str = ""
    image_crop: ImageCropHint | None = Field(default=None, alias="imageCrop")

    @model_validator(mode="after")
    def default_question_number(self) -> "QuestionInput":
        if self.question_number is None:
            try:
                self.question_number = int(self.question_id)
            except (TypeError, ValueError):
                self.question_number = None
        return self

    @property
    def question_id_str(self) -> str:
        return str(self.question_id)


class GradingRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    job_type: JobType = Field(alias="jobType")
    submission_id: str | int = Field(alias="submissionId")
    assignment_id: str | int = Field(alias="assignmentId")
    student_id: str = Field(default="unknown", alias="studentId")
    student_name: str | None = Field(default=None, alias="studentName")
    assignment_title: str | None = Field(default=None, alias="assignmentTitle")
    s3: S3ImageInput | None = None
    questions: list[QuestionInput] = Field(min_length=1)
    previous_result: dict[str, Any] | list[Any] | None = Field(default=None, alias="previousResult")
    request_reason: str | None = Field(default=None, alias="requestReason")

    @field_validator("job_type", mode="before")
    @classmethod
    def normalize_job_type(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.upper()
        return value


class QuestionGradeResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    question_id: str | int = Field(alias="questionId")
    result: QuestionResultLabel
    score: int = Field(ge=0)
    max_score: int = Field(alias="maxScore", ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    feedback_for_student: str = Field(alias="feedbackForStudent")
    detected_answer: str = Field(default="", alias="detectedAnswer")
    mistake_type: str | None = Field(default=None, alias="mistakeType")
    needs_manual_review: bool = Field(default=False, alias="needsManualReview")
    image_url: str | None = Field(default=None, alias="imageUrl")


class GradeSummary(BaseModel):
    correct: int = 0
    partial: int = 0
    wrong: int = 0


class GradingOutput(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    submission_id: str | int = Field(alias="submissionId")
    assignment_id: str | int = Field(alias="assignmentId")
    status: GradingStatus
    total_score: int = Field(alias="totalScore", ge=0)
    max_score: int = Field(alias="maxScore", ge=0)
    correct_rate: float = Field(alias="correctRate", ge=0.0, le=1.0)
    summary: GradeSummary
    questions: list[QuestionGradeResult]
    fail_reason: str | None = Field(default=None, alias="failReason")


class QuestionRegradeResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    question_id: str | int = Field(alias="questionId")
    ai_regraded_score: int = Field(alias="aiRegradedScore", ge=0)
    max_score: int = Field(alias="maxScore", ge=0)
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    needs_manual_review: bool = Field(default=False, alias="needsManualReview")


class RegradeOutput(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    submission_id: str | int = Field(alias="submissionId")
    assignment_id: str | int = Field(alias="assignmentId")
    status: GradingStatus
    questions: list[QuestionRegradeResult]
    fail_reason: str | None = Field(default=None, alias="failReason")


OutputDict = dict[str, Any]
BedrockImageFormat = Literal["png", "jpeg", "gif", "webp"]
