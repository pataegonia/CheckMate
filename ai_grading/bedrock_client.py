from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Any

from .schemas import BedrockImageFormat, S3ImageInput


@dataclass(frozen=True)
class BedrockImage:
    data: bytes
    format: BedrockImageFormat = "jpeg"


class S3ImageLoader:
    def __init__(
        self,
        *,
        region_name: str | None = None,
        profile_name: str | None = None,
    ) -> None:
        self.region_name = (
            region_name
            or os.getenv("AWS_REGION")
            or os.getenv("AWS_DEFAULT_REGION")
            or "us-east-1"
        )
        self.profile_name = profile_name or os.getenv("AWS_PROFILE")

    def _client(self) -> Any:
        import boto3

        if self.profile_name:
            session = boto3.Session(profile_name=self.profile_name, region_name=self.region_name)
        else:
            session = boto3.Session(region_name=self.region_name)
        return session.client("s3")

    def load(self, s3: S3ImageInput) -> bytes:
        response = self._client().get_object(Bucket=s3.bucket, Key=s3.key)
        return response["Body"].read()

    def put_bytes(
        self,
        *,
        bucket: str,
        key: str,
        data: bytes,
        content_type: str = "image/png",
    ) -> None:
        self._client().put_object(
            Bucket=bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )

    def presign_get(
        self,
        *,
        bucket: str,
        key: str,
        expires_in: int = 7 * 24 * 3600,
    ) -> str:
        return self._client().generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expires_in,
        )


class BedrockClaudeClient:
    def __init__(
        self,
        *,
        model_id: str | None = None,
        region_name: str | None = None,
        profile_name: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.model_id = model_id or os.getenv("BEDROCK_MODEL_ID") or "us.anthropic.claude-sonnet-4-6"
        self.region_name = (
            region_name
            or os.getenv("AWS_REGION")
            or os.getenv("AWS_DEFAULT_REGION")
            or "us-east-1"
        )
        self.profile_name = profile_name or os.getenv("AWS_PROFILE")
        self.max_tokens = max_tokens or int(os.getenv("AI_GRADING_MAX_TOKENS", "1200"))
        self.temperature = temperature

    def _client(self) -> Any:
        import boto3

        if self.profile_name:
            session = boto3.Session(profile_name=self.profile_name, region_name=self.region_name)
        else:
            session = boto3.Session(region_name=self.region_name)
        return session.client("bedrock-runtime")

    def invoke_multimodal(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        images: list[BedrockImage],
    ) -> str:
        content: list[dict[str, Any]] = []
        for image in images:
            content.append(
                {
                    "image": {
                        "format": image.format,
                        "source": {"bytes": image.data},
                    }
                }
            )
        content.append({"text": user_prompt})

        response = self._client().converse(
            modelId=self.model_id,
            system=[{"text": system_prompt}],
            messages=[{"role": "user", "content": content}],
            inferenceConfig={
                "maxTokens": self.max_tokens,
                "temperature": self.temperature,
            },
        )
        blocks = response.get("output", {}).get("message", {}).get("content", [])
        return "\n".join(block.get("text", "") for block in blocks if "text" in block).strip()


def pil_image_from_bytes(image_bytes: bytes) -> Any:
    Image = _image_module()
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def encode_image_for_bedrock(
    image: Any,
    *,
    image_format: BedrockImageFormat = "jpeg",
    max_side: int | None = None,
    quality: int = 85,
) -> BedrockImage:
    pil_image = _to_pil_rgb(image)
    max_side = max_side or int(os.getenv("AI_GRADING_CROP_MAX_SIDE", "1600"))
    pil_image = _resize_to_max_side(pil_image, max_side=max_side)

    buffer = io.BytesIO()
    save_format = "JPEG" if image_format == "jpeg" else image_format.upper()
    save_kwargs: dict[str, Any] = {}
    if image_format == "jpeg":
        save_kwargs.update({"quality": quality, "optimize": True})
    pil_image.save(buffer, format=save_format, **save_kwargs)
    return BedrockImage(data=buffer.getvalue(), format=image_format)


def _to_pil_rgb(image: Any) -> Any:
    Image = _image_module()
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    try:
        import numpy as np

        if isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
    except Exception:
        pass
    raise TypeError("image must be a PIL image or numpy array")


def _resize_to_max_side(image: Any, *, max_side: int) -> Any:
    if max_side <= 0:
        return image
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image
    ratio = max_side / float(longest)
    new_size = (max(1, int(width * ratio)), max(1, int(height * ratio)))
    Image = _image_module()
    return image.resize(new_size, Image.Resampling.LANCZOS)


def _image_module() -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required for AI_GRADING_MODE=bedrock image processing"
        ) from exc
    return Image
