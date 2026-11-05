"""Resolve task image inputs through the authorized durable asset catalog."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from alphonse.agent_v2.assets import MAX_ATTACHMENT_BYTES
from alphonse.agent_v2.core.inference.models import InferenceImage

IMAGE_SUFFIXES = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp", "image/gif": ".gif"}


def task_image_inputs(task: Any, asset_store: Any) -> tuple[InferenceImage, ...]:
    """Snapshot only registered, task-attached images owned by the requester.

    Attachment metadata may identify assets but cannot supply paths or bytes.
    Missing/forbidden inputs fail explicitly rather than becoming text-only calls.
    """
    metadata = task.metadata if isinstance(task.metadata, dict) else {}
    attachments = metadata.get("attachments") or []
    ids = list(metadata.get("asset_ids") or [])
    for item in attachments:
        if not isinstance(item, dict):
            continue
        asset_id = str(item.get("asset_id") or "").strip()
        if not asset_id and str(item.get("mime_type") or "").startswith("image/"):
            raise ValueError("inference_image_attachment_unregistered")
        if asset_id:
            ids.append(asset_id)
    images = []
    for asset_id in dict.fromkeys(str(item) for item in ids if item):
        if asset_store is None or not task.user:
            raise ValueError("inference_attachment_store_unavailable")
        asset = asset_store.get(asset_id, requester_user_id=str(task.user))
        if asset is None:
            raise ValueError("inference_attachment_not_found_or_forbidden")
        if not asset.mime_type.startswith("image/"):
            continue
        if asset.mime_type not in IMAGE_SUFFIXES:
            raise ValueError(f"inference_image_type_unsupported:{asset.mime_type}")
        try:
            with Path(asset.path).open("rb") as source:
                content = source.read(MAX_ATTACHMENT_BYTES + 1)
        except OSError as exc:
            raise ValueError("inference_image_unavailable") from exc
        if not content or len(content) > MAX_ATTACHMENT_BYTES:
            raise ValueError("inference_image_size_invalid")
        if hashlib.sha256(content).hexdigest() != asset.sha256:
            raise ValueError("inference_image_integrity_failed")
        images.append(InferenceImage(asset_id=asset.asset_id, mime_type=asset.mime_type, content=content))
    return tuple(images)
