from __future__ import annotations

import io
import os
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from ai_grading.api_adapter import (
    from_be_invoke_event,
    post_callback,
    to_be_grade_payload,
)
from ai_grading.bedrock_client import BedrockImage
from ai_grading.grading_engine import QuestionCrop, _maybe_upload_crop
from ai_grading.lambda_handler import handler
from ai_grading.schemas import GradingRequest
from ai_grading.worker import HttpResultSink, WorkerJob


EVENT = {
    "submission": {
        "submissionId": 10,
        "image": {"bucket": "checkmate-bucket", "key": "submissions/a.jpg"},
    },
    "assignment": {"assignmentId": 1},
    "questions": [
        {
            "questionId": 21,
            "orderNum": 814,
            "type": "descriptive",
            "expectedAnswer": "2",
            "maxScore": 10,
            "gradingCriteria": "Show the reasoning.",
        }
    ],
    "callback": {"url": "https://be.example/api/internal/submissions/10/result"},
}


class AdapterTests(unittest.TestCase):
    def test_be_event_maps_to_internal_request(self) -> None:
        request = GradingRequest.model_validate(from_be_invoke_event(EVENT))

        self.assertEqual(request.submission_id, 10)
        self.assertEqual(request.s3.bucket, "checkmate-bucket")
        self.assertEqual(request.questions[0].question_id, 21)
        self.assertEqual(request.questions[0].question_number, 814)
        self.assertEqual(request.questions[0].question_type, "descriptive")
        self.assertEqual(request.questions[0].answer, "2")

    def test_result_payload_combines_feedback_and_preserves_image_url(self) -> None:
        result = to_be_grade_payload(
            {
                "status": "DONE",
                "totalScore": 6.7,
                "questions": [
                    {
                        "questionId": "21",
                        "score": 6.7,
                        "reason": "Partial credit.",
                        "feedbackForStudent": "Review the final sign.",
                        "imageUrl": "https://signed.example/crop.png",
                    }
                ],
            }
        )

        self.assertEqual(result["totalScore"], 7)
        self.assertEqual(result["questions"][0]["questionId"], 21)
        self.assertEqual(
            result["questions"][0]["reason"],
            "Partial credit.\nReview the final sign.",
        )
        self.assertEqual(
            result["questions"][0]["imageUrl"],
            "https://signed.example/crop.png",
        )

    def test_image_crop_is_forwarded_to_internal_request(self) -> None:
        event = {**EVENT, "questions": [{**EVENT["questions"][0]}]}
        event["questions"][0]["imageCrop"] = {"x1": 10, "y1": 20, "x2": 110, "y2": 220}

        request = GradingRequest.model_validate(from_be_invoke_event(event))

        crop = request.questions[0].image_crop
        self.assertIsNotNone(crop)
        self.assertEqual((crop.x1, crop.y1, crop.x2, crop.y2), (10, 20, 110, 220))

    def test_be_event_rejects_missing_order_number(self) -> None:
        event = {**EVENT, "questions": [{**EVENT["questions"][0]}]}
        event["questions"][0].pop("orderNum")

        with self.assertRaises(ValueError):
            from_be_invoke_event(event)

    def test_callback_treats_duplicate_as_terminal_but_not_bad_token(self) -> None:
        duplicate = HTTPError(
            "https://be.example/result",
            409,
            "Conflict",
            {},
            io.BytesIO(b"already complete"),
        )
        with patch("ai_grading.api_adapter.urlopen", side_effect=duplicate):
            status, _ = post_callback(
                url="https://be.example/result",
                payload={"status": "FAILED"},
                internal_token="secret",
            )
        self.assertEqual(status, 409)

        forbidden = HTTPError(
            "https://be.example/result",
            403,
            "Forbidden",
            {},
            io.BytesIO(b"invalid token"),
        )
        with patch("ai_grading.api_adapter.urlopen", side_effect=forbidden):
            with self.assertRaises(RuntimeError):
                post_callback(
                    url="https://be.example/result",
                    payload={"status": "FAILED"},
                    internal_token="bad-secret",
                )


class CropUploadTests(unittest.TestCase):
    def test_crop_uses_submission_bucket_by_default(self) -> None:
        request = GradingRequest.model_validate(from_be_invoke_event(EVENT))
        crop = QuestionCrop(
            images=[],
            metadata={},
            upload_png=BedrockImage(data=b"png", format="png"),
        )
        gateway = Mock()
        gateway.presign_get.return_value = "https://signed.example/crop.png"

        with patch.dict(os.environ, {}, clear=True):
            url = _maybe_upload_crop(gateway, request, request.questions[0], crop)

        self.assertEqual(url, "https://signed.example/crop.png")
        gateway.put_bytes.assert_called_once_with(
            bucket="checkmate-bucket",
            key="graded-crops/1/10/21.png",
            data=b"png",
            content_type="image/png",
        )


class WorkerSinkTests(unittest.TestCase):
    def test_worker_adds_internal_header_and_resolves_submission_endpoint(self) -> None:
        sink = HttpResultSink(
            endpoint="https://be.example/api/internal/submissions/{submissionId}/result",
            internal_token="secret",
        )
        job = WorkerJob(payload={"submissionId": 10})

        with patch("ai_grading.worker._request_json") as request_json:
            sink.submit(job, {"submissionId": 10, "status": "DONE"})

        request_json.assert_called_once()
        kwargs = request_json.call_args.kwargs
        self.assertEqual(
            request_json.call_args.args[0],
            "https://be.example/api/internal/submissions/10/result",
        )
        self.assertEqual(kwargs["extra_headers"], {"X-Internal-Token": "secret"})

    def test_worker_only_swallows_agreed_terminal_4xx(self) -> None:
        sink = HttpResultSink(endpoint="https://be.example/result")
        job = WorkerJob(payload={"submissionId": 10})

        with patch(
            "ai_grading.worker._request_json",
            side_effect=RuntimeError("HTTP 409 from endpoint: done"),
        ):
            sink.submit(job, {"submissionId": 10, "status": "DONE"})

        with patch(
            "ai_grading.worker._request_json",
            side_effect=RuntimeError("HTTP 403 from endpoint: token"),
        ):
            with self.assertRaises(RuntimeError):
                sink.submit(job, {"submissionId": 10, "status": "DONE"})


class LambdaHandlerTests(unittest.TestCase):
    def test_handler_posts_be_payload_with_internal_token(self) -> None:
        output = {
            "status": "DONE",
            "totalScore": 10,
            "questions": [
                {
                    "questionId": 21,
                    "score": 10,
                    "reason": "Correct.",
                    "feedbackForStudent": "",
                    "imageUrl": None,
                }
            ],
        }
        with (
            patch.dict(os.environ, {"APP_INTERNAL_TOKEN": "secret"}, clear=True),
            patch("ai_grading.lambda_handler.grade_submission", return_value=output),
            patch(
                "ai_grading.lambda_handler.post_callback",
                return_value=(200, '{"code":"S108"}'),
            ) as callback,
        ):
            result = handler(EVENT)

        self.assertEqual(result["status"], "DONE")
        kwargs = callback.call_args.kwargs
        self.assertEqual(kwargs["url"], EVENT["callback"]["url"])
        self.assertEqual(kwargs["internal_token"], "secret")
        self.assertEqual(kwargs["payload"]["questions"][0]["questionId"], 21)


if __name__ == "__main__":
    unittest.main()
