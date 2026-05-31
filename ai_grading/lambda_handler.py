from __future__ import annotations

import json
import logging
import os
from typing import Any

from .api_adapter import (
    from_be_invoke_event,
    post_callback,
    to_be_fail_payload,
    to_be_grade_payload,
)
from .grading_engine import grade_submission


LOGGER = logging.getLogger()
if not LOGGER.handlers:
    logging.basicConfig(level=os.getenv("AI_GRADING_LOG_LEVEL", "INFO").upper())


def handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    """Lambda entrypoint invoked by BE.

    Expected event shape — see api_adapter.from_be_invoke_event.
    BE-agreed contract:
      - callback URL comes from event["callback"]["url"]
      - APP_INTERNAL_TOKEN env var holds the shared secret for the X-Internal-Token header
    """
    callback_url = (event.get("callback") or {}).get("url")
    if not callback_url:
        raise ValueError("event.callback.url is required")

    internal_token = os.environ.get("APP_INTERNAL_TOKEN")
    if not internal_token:
        raise RuntimeError("APP_INTERNAL_TOKEN env var must be set")

    submission_id = (event.get("submission") or {}).get("submissionId")
    LOGGER.info("Grading start: submissionId=%s", submission_id)

    try:
        internal_payload = from_be_invoke_event(event)
        output = grade_submission(internal_payload)
        be_payload = to_be_grade_payload(output)
    except Exception as exc:
        LOGGER.exception("Grading failed before callback: %s", exc)
        be_payload = to_be_fail_payload(f"{type(exc).__name__}: {exc}")

    status, body = post_callback(
        url=callback_url,
        payload=be_payload,
        internal_token=internal_token,
    )
    LOGGER.info(
        "Callback returned: submissionId=%s status=%s http=%s",
        submission_id,
        be_payload.get("status"),
        status,
    )
    return {
        "submissionId": submission_id,
        "status": be_payload.get("status"),
        "callbackHttpStatus": status,
        "callbackBody": body,
    }


if __name__ == "__main__":
    import sys

    raw = sys.stdin.read()
    event_in = json.loads(raw) if raw else {}
    result = handler(event_in)
    print(json.dumps(result, ensure_ascii=False, indent=2))
