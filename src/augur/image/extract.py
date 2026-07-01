"""VLM-based extraction of structured facts from a still image.

The live model call (`extract_from_image`) is wired to the Anthropic Messages
API. It needs an API key in the environment (`ANTHROPIC_API_KEY`) and the
`vision` extra installed (`pip install -e ".[vision]"`). The model id is
overridable via the `model` argument or `AUGUR_VLM_MODEL`; the default is a
current vision-capable Claude and should be reviewed against live docs.

The request building, image encoding, and response-text extraction are factored
into small pure helpers, and the client is injectable, so the whole path is
unit-tested with a fake client — no key or network needed. Model output flows
through `parse_vlm_response` (strict, defensive JSON parsing) into the
`ImageExtraction` schema the rest of the system consumes.
"""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from augur.fusion.observations import Observation

# Default vision model. Overridable via the `model` arg or AUGUR_VLM_MODEL; the
# spec calls for confirming the current id against live docs rather than trusting
# this constant blindly.
DEFAULT_VLM_MODEL = "claude-sonnet-4-6"

_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


class BatteryInfo(BaseModel):
    visible: bool = False
    printed_mah: float | None = None
    cell_count: int | None = None
    dims_mm: list[float] | None = None  # [l, w, h]
    confidence: float = 0.0


class ImageExtraction(BaseModel):
    drone_class: str | None = None
    battery: BatteryInfo = Field(default_factory=BatteryInfo)
    payload_present: bool = False
    payload_type: str | None = None
    features: list[str] = Field(default_factory=list)  # camera, gimbal, gps, antennas...
    num_motors: int | None = None
    field_confidence: dict[str, float] = Field(default_factory=dict)


# The instruction handed to the VLM. Demands strict JSON, no prose.
EXTRACTION_PROMPT = """You are inspecting a photo of a multirotor drone. Return ONLY
a JSON object (no prose, no markdown fences) with these keys:
  drone_class: one of "racing","cinematic","survey", or null if unsure
  battery: {visible: bool, printed_mah: number|null, cell_count: int|null,
            dims_mm: [l,w,h]|null, confidence: 0..1}
  payload_present: bool
  payload_type: string|null
  features: array of strings from {"camera","gimbal","gps","antennas","fpv","prop_guards"}
  num_motors: int|null
  field_confidence: object mapping each field name to a 0..1 confidence
Do not guess values you cannot see; use null and low confidence instead."""


def parse_vlm_response(text: str) -> ImageExtraction:
    """Parse a VLM response into ImageExtraction, defensively.

    Handles: bare JSON, JSON wrapped in ```json fences, and leading/trailing
    prose. Raises ValueError on anything that is not recoverable JSON or fails
    schema validation — callers decide how to degrade (usually: contribute no
    image observations rather than fabricate)."""
    candidate = _extract_json_block(text)
    if candidate is None:
        raise ValueError("no JSON object found in VLM response")
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ValueError(f"VLM response was not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ValueError("VLM JSON was not an object")
    try:
        return ImageExtraction(**data)
    except ValidationError as e:
        raise ValueError(f"VLM JSON failed schema validation: {e}") from e


def _extract_json_block(text: str) -> str | None:
    if not text or not text.strip():
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return fence.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return None


def extraction_to_observations(ext: ImageExtraction, cell_voltage_nominal: float = 3.7) -> list[Observation]:
    """Convert a (parsed) extraction into fusion observations.

    Only the directly-read, metric facts become observations. A printed mAh +
    cell count is the 'direct' battery path (spec §2) — it short-circuits the
    residual subtraction and is the only way to hit the battery Wh target."""
    obs: list[Observation] = []
    b = ext.battery
    if b.visible and b.printed_mah and b.cell_count:
        wh = b.printed_mah / 1000.0 * (b.cell_count * cell_voltage_nominal)
        obs.append(Observation(variable="battery_wh", value=wh, sigma=max(0.05 * wh, 0.5),
                               source="image", note="printed label"))
    if ext.num_motors:
        obs.append(Observation(variable="num_motors", value=float(ext.num_motors),
                               sigma=0.25, source="image"))
    return obs


def _media_type(image_path: str | Path) -> str:
    ext = Path(image_path).suffix.lower()
    if ext not in _MEDIA_TYPES:
        raise ValueError(f"unsupported image type {ext!r}; expected one of {sorted(_MEDIA_TYPES)}")
    return _MEDIA_TYPES[ext]


def _encode_image(image_path: str | Path) -> str:
    return base64.standard_b64encode(Path(image_path).read_bytes()).decode("ascii")


def _build_messages(image_b64: str, media_type: str) -> list[dict]:
    """The Messages-API payload: the image, then the strict-JSON instruction."""
    return [{
        "role": "user",
        "content": [
            {"type": "image", "source": {"type": "base64", "media_type": media_type,
                                         "data": image_b64}},
            {"type": "text", "text": EXTRACTION_PROMPT},
        ],
    }]


def _response_text(message) -> str:
    """Concatenate the text blocks of an Anthropic Messages response."""
    parts = [getattr(block, "text", "") for block in getattr(message, "content", [])]
    return "\n".join(p for p in parts if p)


def _default_client():
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError(
            "VLM extraction needs the 'vision' extra: pip install -e '.[vision]'"
        ) from e
    return anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment


def extract_from_image(image_path: str, model: str | None = None, client=None,
                       max_tokens: int = 1024) -> ImageExtraction:
    """Call the VLM on an image and parse the structured extraction.

    `client` is injectable for testing; when omitted, a default Anthropic client
    is built lazily (requiring the vision extra + API key). The model output is
    run through `parse_vlm_response`, so a malformed response raises ValueError
    and callers degrade to contributing no image observations."""
    media_type = _media_type(image_path)
    image_b64 = _encode_image(image_path)
    client = client or _default_client()

    message = client.messages.create(
        model=model or os.environ.get("AUGUR_VLM_MODEL", DEFAULT_VLM_MODEL),
        max_tokens=max_tokens,
        messages=_build_messages(image_b64, media_type),
    )
    return parse_vlm_response(_response_text(message))
