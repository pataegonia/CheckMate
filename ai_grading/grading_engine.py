from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .bedrock_client import (
    BedrockClaudeClient,
    BedrockImage,
    S3ImageLoader,
    encode_image_for_bedrock,
    pil_image_from_bytes,
)
from .mock_grader import MockGrader
from .prompt_builder import SYSTEM_PROMPT, build_grade_prompt, build_regrade_prompt
from .response_parser import ResponseParseError, parse_json_response
from .result_validator import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    build_failed_grade_output,
    build_failed_regrade_output,
    build_grade_output,
    build_regrade_output,
    validate_grade_question,
    validate_regrade_question,
)
from .schemas import GradingRequest, JobType, QuestionInput, OutputDict


@dataclass
class QuestionCrop:
    images: list[BedrockImage]
    metadata: dict[str, Any]
    upload_png: BedrockImage | None = None


def grade_submission(payload: dict[str, Any] | GradingRequest) -> OutputDict:
    request = payload if isinstance(payload, GradingRequest) else GradingRequest.model_validate(payload)
    mode = os.getenv("AI_GRADING_MODE", "mock").lower()

    if mode == "mock":
        result = _run_mock(request)
    elif mode == "bedrock":
        result = _run_bedrock(request)
    else:
        failed = _failed_output(request, f"Unsupported AI_GRADING_MODE: {mode}")
        return failed

    return _dump_output(result)


def _run_mock(request: GradingRequest):
    grader = MockGrader()
    if request.job_type == JobType.REGRADE:
        return grader.regrade(request)
    return grader.grade(request)


def _run_bedrock(request: GradingRequest):
    if request.s3 is None:
        return _failed_model(request, "s3 input is required when AI_GRADING_MODE=bedrock")

    try:
        s3_gateway = S3ImageLoader()
        page_bytes = s3_gateway.load(request.s3)
        page_image = pil_image_from_bytes(page_bytes)
        crops = _build_question_crops(page_image, request)
        client = BedrockClaudeClient()

        if request.job_type == JobType.REGRADE:
            results = []
            previous_by_id = _previous_by_question_id(request.previous_result)
            for question in request.questions:
                crop = _require_crop(question, crops)
                prompt = build_regrade_prompt(
                    request,
                    question,
                    previous_question_result=previous_by_id.get(str(question.question_id)),
                    crop_metadata=crop.metadata,
                )
                raw_text = client.invoke_multimodal(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=prompt,
                    images=crop.images,
                )
                raw_json = parse_json_response(raw_text)
                results.append(
                    validate_regrade_question(
                        question,
                        raw_json,
                        confidence_threshold=_confidence_threshold(),
                    )
                )
            return build_regrade_output(request, results)

        results = []
        for question in request.questions:
            crop = _require_crop(question, crops)
            prompt = build_grade_prompt(request, question, crop_metadata=crop.metadata)
            raw_text = client.invoke_multimodal(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt,
                images=crop.images,
            )
            raw_json = parse_json_response(raw_text)
            graded = validate_grade_question(
                question,
                raw_json,
                confidence_threshold=_confidence_threshold(),
            )
            image_url = _maybe_upload_crop(
                s3_gateway, request, question, crop
            )
            if image_url:
                graded = graded.model_copy(update={"image_url": image_url})
            results.append(graded)
        return build_grade_output(request, results)
    except ResponseParseError as exc:
        return _failed_model(request, str(exc))
    except Exception as exc:
        return _failed_model(request, f"{type(exc).__name__}: {exc}")


def _build_question_crops(page_image, request: GradingRequest) -> dict[str, QuestionCrop]:
    manual_crops = _build_manual_crops(page_image, request)
    missing = [question for question in request.questions if question.question_id_str not in manual_crops]
    if not missing:
        return manual_crops

    detected_crops = _build_pipeline_crops(page_image, request)
    return {**detected_crops, **manual_crops}


def _build_manual_crops(page_image, request: GradingRequest) -> dict[str, QuestionCrop]:
    crops: dict[str, QuestionCrop] = {}
    for question in request.questions:
        if question.image_crop is None:
            continue
        box = (
            question.image_crop.x1,
            question.image_crop.y1,
            question.image_crop.x2,
            question.image_crop.y2,
        )
        cropped = page_image.crop(box)
        crops[question.question_id_str] = QuestionCrop(
            images=[encode_image_for_bedrock(cropped)],
            metadata={
                "cropSource": "imageCrop",
                "bbox": list(box),
                "questionNumber": question.question_number,
            },
            upload_png=encode_image_for_bedrock(cropped, image_format="png"),
        )
    return crops


def _build_pipeline_crops(page_image, request: GradingRequest) -> dict[str, QuestionCrop]:
    import numpy as np

    from pipeline import grade_page

    answer_dict = {
        question.question_number: question.answer
        for question in request.questions
        if question.question_number is not None
    }
    if not answer_dict:
        raise ValueError("questionNumber is required for internal question cropping")

    page_rgb = np.array(page_image.convert("RGB"))
    crop_results = grade_page(page_rgb, answer_dict, return_debug=False)
    by_problem = {int(item["problem_num"]): item for item in crop_results}
    crops: dict[str, QuestionCrop] = {}

    for question in request.questions:
        if question.question_number is None:
            continue
        item = by_problem.get(int(question.question_number))
        if item is None:
            continue
        images = [encode_image_for_bedrock(item["crop_rgb"])]
        if item.get("answer_crop_rgb") is not None:
            images.append(encode_image_for_bedrock(item["answer_crop_rgb"]))
        crops[question.question_id_str] = QuestionCrop(
            images=images,
            metadata={
                "cropSource": "pipeline.grade_page",
                "questionNumber": question.question_number,
                "bbox": list(item.get("bbox", [])),
                "ocrStudentAnswer": item.get("student_answer", ""),
                "ocrConfidence": item.get("confidence", 0.0),
                "ocrSource": item.get("source", ""),
                "ocrCandidates": item.get("candidates", []),
                "hasAnswerFocusCrop": item.get("answer_crop_rgb") is not None,
            },
            upload_png=encode_image_for_bedrock(item["crop_rgb"], image_format="png"),
        )
    return crops


def _require_crop(question: QuestionInput, crops: dict[str, QuestionCrop]) -> QuestionCrop:
    crop = crops.get(question.question_id_str)
    if crop is None:
        raise ValueError(
            f"Could not crop questionId={question.question_id} "
            f"(questionNumber={question.question_number})"
        )
    return crop


def _previous_by_question_id(previous_result):
    if isinstance(previous_result, dict):
        items = previous_result.get("questions", [])
    elif isinstance(previous_result, list):
        items = previous_result
    else:
        items = []
    return {str(item.get("questionId")): item for item in items if isinstance(item, dict)}


def _confidence_threshold() -> float:
    try:
        return float(os.getenv("AI_GRADING_CONFIDENCE_THRESHOLD", str(DEFAULT_CONFIDENCE_THRESHOLD)))
    except ValueError:
        return DEFAULT_CONFIDENCE_THRESHOLD


def _failed_model(request: GradingRequest, reason: str):
    if request.job_type == JobType.REGRADE:
        return build_failed_regrade_output(request, reason)
    return build_failed_grade_output(request, reason)


def _failed_output(request: GradingRequest, reason: str) -> OutputDict:
    return _dump_output(_failed_model(request, reason))


def _dump_output(result) -> OutputDict:
    data = result.model_dump(by_alias=True, mode="json")
    if data.get("failReason") is None:
        data.pop("failReason", None)
    return data


def _maybe_upload_crop(
    s3_gateway: S3ImageLoader,
    request: GradingRequest,
    question: QuestionInput,
    crop: QuestionCrop,
) -> str | None:
    bucket = os.getenv("CHECKMATE_CROPS_BUCKET") or (
        request.s3.bucket if request.s3 is not None else None
    )
    if not bucket or crop.upload_png is None:
        return None
    prefix = os.getenv("CHECKMATE_CROPS_PREFIX", "graded-crops").strip("/")
    key = f"{prefix}/{request.assignment_id}/{request.submission_id}/{question.question_id_str}.png"
    s3_gateway.put_bytes(
        bucket=bucket,
        key=key,
        data=crop.upload_png.data,
        content_type="image/png",
    )
    expires_in = int(os.getenv("CHECKMATE_CROPS_URL_TTL_SECONDS", str(7 * 24 * 3600)))
    return s3_gateway.presign_get(bucket=bucket, key=key, expires_in=expires_in)
