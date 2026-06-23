"""VLM-based extraction of structured facts from a still image.

STATUS: the live model call is intentionally not wired. It needs (a) an Anthropic
API key in the environment and (b) the *current* model id + tool-use setup, which
per the project spec must be checked against live docs (product-self-knowledge /
claude-api skill) rather than hardcoded from memory. `extract_from_image` raises
until that wiring is done.

What IS implemented and tested here is `parse_vlm_response` — strict, defensive
JSON parsing of the model output, including malformed responses — and the
`ImageExtraction` schema the rest of the system consumes. Wiring the call later
is then a thin, low-risk layer on top of a tested parser.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field, ValidationError

from augur.fusion.observations import Observation


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


def extract_from_image(image_path: str, model: str | None = None) -> ImageExtraction:
    """Call the VLM on an image. NOT YET WIRED — see module docstring."""
    raise NotImplementedError(
        "VLM image extraction is not wired yet. Wire the Anthropic call here using "
        "the current model id + tool-use setup from live docs (claude-api skill), "
        "then feed the response text through parse_vlm_response()."
    )
