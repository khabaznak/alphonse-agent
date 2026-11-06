"""Optional, presentation-only skill resources. Never participates in tool curation."""

from __future__ import annotations

import base64
import hashlib
import json
import struct
import zlib
from pathlib import Path
from typing import Any

STATES = ("idle", "planning", "doing", "checking", "acting", "error", "disconnected")
PRESETS = {"doctor": "Family health assistant", "exercise-coach": "Exercise coach"}
MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_TOTAL_BYTES = 8 * 1024 * 1024


def leading_skill(selection: Any) -> str | None:
    """Use an existing System One judgment; ambiguity is deliberately the base mascot."""
    if not isinstance(selection, dict):
        return None
    selected = selection.get("selected_skill_ids")
    primary = selection.get("primary_skill_ids", [])
    if not isinstance(selected, list) or not isinstance(primary, list):
        return None
    selected = set(item for item in selected if isinstance(item, str))
    primary = set(item for item in primary if isinstance(item, str) and item in selected)
    candidates = primary or selected
    return next(iter(candidates)) if len(candidates) == 1 else None


def contained_image(root: Path, relative: str) -> bytes:
    path = Path(relative)
    if not relative or path.is_absolute() or ".." in path.parts or "\\" in relative:
        raise ValueError("appearance_path_invalid")
    root = root.resolve(strict=True)
    target = root / path
    if not target.resolve(strict=True).is_relative_to(root):
        raise ValueError("appearance_outside_package")
    if target.suffix.lower() != ".png" or not target.is_file() or target.stat().st_size > MAX_IMAGE_BYTES:
        raise ValueError("appearance_image_invalid")
    data = target.read_bytes()
    validate_png(data)
    return data


def validate_png(data: bytes) -> None:
    """Only bounded, static PNGs cross IPC; never SVG/HTML, URLs or animation."""
    if len(data) > MAX_IMAGE_BYTES or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("appearance_png_required")
    offset, dimensions, has_pixels = 8, None, False
    while offset + 12 <= len(data):
        size = struct.unpack(">I", data[offset:offset + 4])[0]
        kind = data[offset + 4:offset + 8]
        payload = data[offset + 8:offset + 8 + size]
        end = offset + size + 12
        if end > len(data) or zlib.crc32(kind + payload) != struct.unpack(">I", data[end - 4:end])[0]:
            raise ValueError("appearance_png_corrupt")
        if dimensions is None:
            if kind != b"IHDR" or size != 13:
                raise ValueError("appearance_png_header")
            dimensions = struct.unpack(">II", payload[:8])
            if not all(1 <= value <= 4096 for value in dimensions):
                raise ValueError("appearance_png_dimensions")
        if kind == b"acTL":
            raise ValueError("appearance_static_png_required")
        has_pixels |= kind == b"IDAT"
        if kind == b"IEND" and size == 0 and has_pixels and end == len(data):
            return
        offset = end
    raise ValueError("appearance_png_incomplete")


def resolve_appearance(record: Any, *, include_resources: bool = False) -> dict[str, Any] | None:
    """Fail only the optional appearance when a package is missing or incompatible."""
    spec = record.appearance
    if not spec:
        return None
    try:
        if not isinstance(spec, dict) or type(spec.get("version")) is not int or spec["version"] != 1:
            raise ValueError("appearance_version_unsupported")
        preset = spec.get("preset")
        if preset is not None:
            if preset not in PRESETS:
                raise ValueError("appearance_preset_unsupported")
            root = Path(__file__).parent / "mascot_assets"
            resources = {"sheet": contained_image(root, f"{preset}.png")}
            states = {state: {"resource": "sheet", "frame": [index % 4, index // 4, 4, 2]} for index, state in enumerate(STATES)}
            label = PRESETS[preset]
        else:
            paths = spec.get("states")
            if not isinstance(paths, dict) or not paths or set(paths) - set(STATES):
                raise ValueError("appearance_states_invalid")
            label = str(spec.get("label") or record.name)[:80]
            resources, states = {}, {}
            for state, relative in paths.items():
                if not isinstance(relative, str):
                    raise ValueError("appearance_path_invalid")
                if relative not in resources:
                    resources[relative] = contained_image(Path(record.directory), relative)
                states[state] = {"resource": relative}
        if sum(len(data) for data in resources.values()) > MAX_TOTAL_BYTES:
            raise ValueError("appearance_resources_too_large")
        digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode())
        for key, data in sorted(resources.items()):
            digest.update(key.encode())
            digest.update(data)
        result = {"skill_id": record.skill_id, "label": label, "revision": digest.hexdigest(), "states": states}
        if preset:
            result["preset"] = preset
        if include_resources:
            result["resources"] = {key: "data:image/png;base64," + base64.b64encode(data).decode("ascii") for key, data in resources.items()}
        return result
    except (OSError, ValueError, TypeError, struct.error):
        return None
