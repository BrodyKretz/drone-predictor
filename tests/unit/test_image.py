"""Image geometry (pixel->metric) and defensive VLM JSON parsing."""

import numpy as np
import pytest

from augur.image import extract, geometry


def test_scale_from_distance_fov():
    s = geometry.ScaleReference(distance_m=5.0, hfov_deg=60.0, image_width_px=1920)
    mpp = s.resolve_m_per_pixel()
    expected = 2 * 5.0 * np.tan(np.radians(60) / 2) / 1920
    assert mpp == pytest.approx(expected)


def test_frontal_area_metric_when_scale_present():
    s = geometry.ScaleReference(m_per_pixel=0.002)
    dist, is_metric = geometry.frontal_area((200, 150), s, rng=np.random.default_rng(0))
    assert is_metric
    # 200px*0.002 = 0.4m, 150px*0.002=0.3m, *0.55 fill ~ 0.066 m^2
    assert 0.03 < dist.median < 0.12


def test_frontal_area_ratio_only_without_scale():
    dist, is_metric = geometry.frontal_area((200, 150), geometry.ScaleReference(),
                                            rng=np.random.default_rng(0))
    assert not is_metric
    assert dist.relative_width > 0.5  # deliberately wide


def test_frame_diagonal_metric():
    s = geometry.ScaleReference(m_per_pixel=0.003)
    dist, is_metric = geometry.frame_diagonal((300, 400), s, rng=np.random.default_rng(0))
    assert is_metric
    assert dist.median == pytest.approx(500 * 0.003, rel=0.1)  # hypot(300,400)=500


def test_parse_bare_json():
    ext = extract.parse_vlm_response('{"drone_class": "racing", "num_motors": 4}')
    assert ext.drone_class == "racing"
    assert ext.num_motors == 4


def test_parse_fenced_json():
    text = 'Here is the result:\n```json\n{"drone_class": "survey", "payload_present": true}\n```'
    ext = extract.parse_vlm_response(text)
    assert ext.drone_class == "survey"
    assert ext.payload_present is True


def test_parse_json_with_prose():
    text = 'The drone appears to be {"drone_class": "cinematic", "features": ["gimbal","camera"]} based on the gimbal.'
    ext = extract.parse_vlm_response(text)
    assert "gimbal" in ext.features


@pytest.mark.parametrize("bad", ["", "no json here", "{not valid json}", "[1,2,3]", "{"])
def test_parse_malformed_raises(bad):
    with pytest.raises(ValueError):
        extract.parse_vlm_response(bad)


def test_battery_observation_from_label():
    ext = extract.ImageExtraction(**{
        "battery": {"visible": True, "printed_mah": 1500, "cell_count": 4, "confidence": 0.9}
    })
    obs = extract.extraction_to_observations(ext)
    wh = next(o for o in obs if o.variable == "battery_wh")
    # 1500mAh * 4S * 3.7V / 1000 = 22.2 Wh
    assert wh.value == pytest.approx(22.2, rel=0.01)


def test_extract_from_image_not_wired():
    with pytest.raises(NotImplementedError):
        extract.extract_from_image("foo.jpg")
