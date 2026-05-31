from __future__ import annotations

import argparse
import json
import logging
import os
import re
import signal
import sys
import time
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import ValidationError

from .grading_engine import grade_submission


LOGGER = logging.getLogger(__name__)


@dataclass
class WorkerJob:
    payload: dict[str, Any]
    source_id: str | None = None
    ack_token: str | None = None
    raw_message: dict[str, Any] | None = None


class JobSource(Protocol):
    def receive(self) -> WorkerJob | None:
        ...

    def ack(self, job: WorkerJob) -> None:
        ...

    def fail(self, job: WorkerJob, exc: Exception) -> None:
        ...


class ResultSink(Protocol):
    def submit(self, job: WorkerJob, result: dict[str, Any]) -> None:
        ...


class SqsJobSource:
    def __init__(
        self,
        *,
        queue_url: str,
        region_name: str | None = None,
        profile_name: str | None = None,
        wait_time_seconds: int = 20,
        visibility_timeout: int | None = None,
    ) -> None:
        self.queue_url = queue_url
        self.region_name = (
            region_name
            or os.getenv("AWS_REGION")
            or os.getenv("AWS_DEFAULT_REGION")
            or "us-east-1"
        )
        self.profile_name = profile_name or os.getenv("AWS_PROFILE")
        self.wait_time_seconds = wait_time_seconds
        self.visibility_timeout = visibility_timeout
        self._sqs_client = None

    def _client(self):
        if self._sqs_client is None:
            import boto3

            if self.profile_name:
                session = boto3.Session(
                    profile_name=self.profile_name,
                    region_name=self.region_name,
                )
            else:
                session = boto3.Session(region_name=self.region_name)
            self._sqs_client = session.client("sqs")
        return self._sqs_client

    def receive(self) -> WorkerJob | None:
        params: dict[str, Any] = {
            "QueueUrl": self.queue_url,
            "MaxNumberOfMessages": 1,
            "WaitTimeSeconds": self.wait_time_seconds,
            "MessageAttributeNames": ["All"],
            "AttributeNames": ["All"],
        }
        if self.visibility_timeout is not None:
            params["VisibilityTimeout"] = self.visibility_timeout

        response = self._client().receive_message(**params)
        messages = response.get("Messages", [])
        if not messages:
            return None

        message = messages[0]
        payload = decode_job_payload(message.get("Body", ""))
        return WorkerJob(
            payload=payload,
            source_id=message.get("MessageId"),
            ack_token=message.get("ReceiptHandle"),
            raw_message=message,
        )

    def ack(self, job: WorkerJob) -> None:
        if job.ack_token:
            self._client().delete_message(
                QueueUrl=self.queue_url,
                ReceiptHandle=job.ack_token,
            )

    def fail(self, job: WorkerJob, exc: Exception) -> None:
        LOGGER.exception("Job failed before it could be acknowledged: %s", job.source_id)


class HttpPollingJobSource:
    def __init__(
        self,
        *,
        endpoint: str,
        method: str = "GET",
        timeout_seconds: float = 15.0,
        bearer_token: str | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.method = method.upper()
        self.timeout_seconds = timeout_seconds
        self.bearer_token = bearer_token

    def receive(self) -> WorkerJob | None:
        status, data = _request_json(
            self.endpoint,
            method=self.method,
            timeout_seconds=self.timeout_seconds,
            bearer_token=self.bearer_token,
            allow_empty=True,
        )
        if status in {204, 404} or data is None:
            return None

        payload = extract_payload(data)
        return WorkerJob(
            payload=payload,
            source_id=str(data.get("jobId") or data.get("gradingJobId") or ""),
            raw_message=data,
        )

    def ack(self, job: WorkerJob) -> None:
        return None

    def fail(self, job: WorkerJob, exc: Exception) -> None:
        LOGGER.exception("HTTP-polled job failed before result submission: %s", job.source_id)


class StdinJobSource:
    def __init__(self, *, input_path: str | None = None) -> None:
        self.input_path = input_path
        self._consumed = False

    def receive(self) -> WorkerJob | None:
        if self._consumed:
            return None
        self._consumed = True

        if self.input_path:
            with open(self.input_path, "r", encoding="utf-8") as file:
                data = json.load(file)
        else:
            data = json.load(sys.stdin)
        return WorkerJob(payload=extract_payload(data), source_id="stdin")

    def ack(self, job: WorkerJob) -> None:
        return None

    def fail(self, job: WorkerJob, exc: Exception) -> None:
        LOGGER.exception("stdin job failed")


class HttpResultSink:
    IDEMPOTENT_4XX = {400, 404, 409}

    def __init__(
        self,
        *,
        endpoint: str,
        timeout_seconds: float = 30.0,
        bearer_token: str | None = None,
        internal_token: str | None = None,
        envelope: bool = False,
    ) -> None:
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self.bearer_token = bearer_token
        self.internal_token = internal_token
        self.envelope = envelope

    def submit(self, job: WorkerJob, result: dict[str, Any]) -> None:
        body = build_result_envelope(job, result) if self.envelope else result
        url = self._resolve_endpoint(job, result)
        extra_headers = {"X-Internal-Token": self.internal_token} if self.internal_token else None
        try:
            _request_json(
                url,
                method="POST",
                body=body,
                timeout_seconds=self.timeout_seconds,
                bearer_token=self.bearer_token,
                allow_empty=False,
                extra_headers=extra_headers,
            )
        except RuntimeError as exc:
            code = _http_status_from_runtime_error(exc)
            if code in self.IDEMPOTENT_4XX:
                LOGGER.warning("Result endpoint returned non-retryable %s; treating as done: %s", code, exc)
                return
            raise

    def _resolve_endpoint(self, job: WorkerJob, result: dict[str, Any]) -> str:
        if "{submissionId}" not in self.endpoint:
            return self.endpoint
        submission_id = result.get("submissionId") or job.payload.get("submissionId")
        if submission_id is None:
            raise ValueError("Result endpoint contains {submissionId} but payload has none")
        return self.endpoint.replace("{submissionId}", str(submission_id))


class StdoutResultSink:
    def __init__(self, *, pretty: bool = True) -> None:
        self.pretty = pretty

    def submit(self, job: WorkerJob, result: dict[str, Any]) -> None:
        if self.pretty:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))


def run_worker(
    *,
    source: JobSource,
    sink: ResultSink,
    poll_interval_seconds: float = 5.0,
    once: bool = False,
) -> int:
    stop = StopFlag()
    stop.install()

    LOGGER.info("AI grading worker started")
    while not stop.requested:
        try:
            job = source.receive()
        except Exception:
            LOGGER.exception("Failed to receive job; skipping")
            if once:
                return 1
            time.sleep(poll_interval_seconds)
            continue
        if job is None:
            if once:
                return 0
            time.sleep(poll_interval_seconds)
            continue

        ok = process_job(job, source=source, sink=sink)
        if once:
            return 0 if ok else 1

    LOGGER.info("AI grading worker stopped")
    return 0


def process_job(job: WorkerJob, *, source: JobSource, sink: ResultSink) -> bool:
    try:
        LOGGER.info("Processing grading job: %s", job.source_id or job.payload.get("submissionId"))
        result = safe_grade_submission(job.payload)
        sink.submit(job, result)
        source.ack(job)
        LOGGER.info(
            "Submitted grading result: submissionId=%s status=%s",
            result.get("submissionId"),
            result.get("status"),
        )
        return True
    except Exception as exc:
        source.fail(job, exc)
        return False


def safe_grade_submission(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return grade_submission(payload)
    except ValidationError as exc:
        return build_failed_result(payload, f"Invalid grading payload: {exc}")
    except Exception as exc:
        return build_failed_result(payload, f"{type(exc).__name__}: {exc}")


def build_failed_result(payload: dict[str, Any], fail_reason: str) -> dict[str, Any]:
    questions = payload.get("questions") if isinstance(payload.get("questions"), list) else []
    max_score = sum(_coerce_int(question.get("maxScore"), 0) for question in questions if isinstance(question, dict))
    base: dict[str, Any] = {
        "submissionId": payload.get("submissionId"),
        "assignmentId": payload.get("assignmentId"),
        "status": "FAILED",
        "questions": [],
        "failReason": fail_reason,
    }
    if str(payload.get("jobType", "GRADE")).upper() != "REGRADE":
        base.update(
            {
                "totalScore": 0,
                "maxScore": max_score,
                "correctRate": 0.0,
                "summary": {"correct": 0, "partial": 0, "wrong": 0},
            }
        )
    return base


def decode_job_payload(body: str) -> dict[str, Any]:
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError("SQS message body must be JSON") from exc
    return extract_payload(data)


def extract_payload(data: Any) -> dict[str, Any]:
    if isinstance(data, str):
        return decode_job_payload(data)

    if not isinstance(data, dict):
        raise ValueError("grading job payload must be a JSON object")

    # SNS-to-SQS fanout wraps the actual JSON message in the Message field.
    message = data.get("Message")
    if isinstance(message, str):
        try:
            return extract_payload(json.loads(message))
        except json.JSONDecodeError:
            pass

    for key in ("payload", "job", "gradingJob"):
        nested = data.get(key)
        if isinstance(nested, dict):
            return nested

    return data


def build_result_envelope(job: WorkerJob, result: dict[str, Any]) -> dict[str, Any]:
    payload = job.payload
    return {
        "jobId": payload.get("jobId") or payload.get("gradingJobId") or job.source_id,
        "jobType": payload.get("jobType"),
        "submissionId": result.get("submissionId") or payload.get("submissionId"),
        "assignmentId": result.get("assignmentId") or payload.get("assignmentId"),
        "studentId": payload.get("studentId"),
        "status": result.get("status"),
        "result": result,
    }


def build_source(args: argparse.Namespace) -> JobSource:
    source_type = args.source or _default_source_type()
    if source_type == "sqs":
        queue_url = args.queue_url or os.getenv("AI_GRADING_SQS_QUEUE_URL")
        if not queue_url:
            raise ValueError("AI_GRADING_SQS_QUEUE_URL or --queue-url is required for SQS source")
        return SqsJobSource(
            queue_url=queue_url,
            wait_time_seconds=args.sqs_wait_seconds,
            visibility_timeout=args.sqs_visibility_timeout,
        )

    if source_type == "http":
        endpoint = args.poll_endpoint or os.getenv("AI_GRADING_POLL_ENDPOINT")
        if not endpoint:
            raise ValueError("AI_GRADING_POLL_ENDPOINT or --poll-endpoint is required for HTTP source")
        return HttpPollingJobSource(
            endpoint=endpoint,
            method=args.poll_method,
            timeout_seconds=args.http_timeout_seconds,
            bearer_token=os.getenv("AI_GRADING_API_TOKEN"),
        )

    if source_type == "stdin":
        return StdinJobSource(input_path=args.input)

    raise ValueError(f"Unsupported worker source: {source_type}")


def build_sink(args: argparse.Namespace) -> ResultSink:
    endpoint = args.result_endpoint or os.getenv("AI_GRADING_RESULT_ENDPOINT")
    if endpoint:
        return HttpResultSink(
            endpoint=endpoint,
            timeout_seconds=args.http_timeout_seconds,
            bearer_token=os.getenv("AI_GRADING_API_TOKEN"),
            internal_token=os.getenv("APP_INTERNAL_TOKEN"),
            envelope=args.result_envelope,
        )
    return StdoutResultSink(pretty=not args.compact_json)


def _default_source_type() -> str:
    configured = os.getenv("AI_GRADING_WORKER_SOURCE")
    if configured:
        return configured.lower()
    if os.getenv("AI_GRADING_SQS_QUEUE_URL"):
        return "sqs"
    if os.getenv("AI_GRADING_POLL_ENDPOINT"):
        return "http"
    return "stdin"


def _request_json(
    endpoint: str,
    *,
    method: str,
    body: dict[str, Any] | None = None,
    timeout_seconds: float,
    bearer_token: str | None,
    allow_empty: bool = False,
    extra_headers: dict[str, str] | None = None,
) -> tuple[int, Any]:
    headers = {"Accept": "application/json"}
    data = None
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    if extra_headers:
        headers.update(extra_headers)

    request = Request(endpoint, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            status = response.status
            raw = response.read()
    except HTTPError as exc:
        if allow_empty and exc.code in {204, 404}:
            return exc.code, None
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {endpoint}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach {endpoint}: {exc.reason}") from exc

    if not raw:
        if allow_empty:
            return status, None
        raise RuntimeError(f"Empty response from {endpoint}")

    try:
        return status, json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        snippet = raw[:200].decode("utf-8", errors="replace")
        raise RuntimeError(f"Response from {endpoint} was not JSON: {snippet}") from exc


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _http_status_from_runtime_error(exc: RuntimeError) -> int | None:
    match = re.match(r"HTTP (\d{3})\b", str(exc))
    return int(match.group(1)) if match else None


class StopFlag:
    def __init__(self) -> None:
        self.requested = False

    def install(self) -> None:
        try:
            signal.signal(signal.SIGINT, self._handle)
            signal.signal(signal.SIGTERM, self._handle)
        except ValueError:
            return None

    def _handle(self, signum, frame) -> None:
        self.requested = True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CheckMate AI grading worker")
    parser.add_argument("--source", choices=("sqs", "http", "stdin"))
    parser.add_argument("--once", action="store_true", help="Process at most one job and exit")
    parser.add_argument("--input", help="Read one grading payload from a JSON file")
    parser.add_argument("--poll-interval-seconds", type=float, default=float(os.getenv("AI_GRADING_POLL_INTERVAL", "5")))
    parser.add_argument("--queue-url", help="SQS queue URL. Defaults to AI_GRADING_SQS_QUEUE_URL")
    parser.add_argument("--sqs-wait-seconds", type=int, default=int(os.getenv("AI_GRADING_SQS_WAIT_SECONDS", "20")))
    parser.add_argument("--sqs-visibility-timeout", type=int, default=_optional_int("AI_GRADING_SQS_VISIBILITY_TIMEOUT"))
    parser.add_argument("--poll-endpoint", help="Spring endpoint that returns one pending grading job")
    parser.add_argument("--poll-method", default=os.getenv("AI_GRADING_POLL_METHOD", "GET"))
    parser.add_argument("--result-endpoint", help="Spring endpoint that accepts a grading result")
    parser.add_argument("--result-envelope", action="store_true", default=_env_bool("AI_GRADING_RESULT_ENVELOPE"))
    parser.add_argument("--http-timeout-seconds", type=float, default=float(os.getenv("AI_GRADING_HTTP_TIMEOUT", "30")))
    parser.add_argument("--compact-json", action="store_true", help="Write compact JSON when using stdout sink")
    parser.add_argument("--log-level", default=os.getenv("AI_GRADING_LOG_LEVEL", "INFO"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    try:
        source = build_source(args)
        sink = build_sink(args)
        return run_worker(
            source=source,
            sink=sink,
            poll_interval_seconds=args.poll_interval_seconds,
            once=args.once,
        )
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 2


def _optional_int(name: str) -> int | None:
    value = os.getenv(name)
    if not value:
        return None
    return int(value)


def _env_bool(name: str) -> bool:
    return os.getenv(name, "").lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    raise SystemExit(main())
