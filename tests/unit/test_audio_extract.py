"""Audio-track extraction from video (ffmpeg demux -> existing audio pipeline).

Real-ffmpeg tests are skipped when the binary is absent; fixtures are
synthesized with ffmpeg itself (lavfi color source) so no footage is needed.
"""

import shutil
import subprocess

import numpy as np
import pytest
from scipy.io import wavfile

from augur.video import audio_extract

_HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

needs_ffmpeg = pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not installed")


def _write_drone_wav(path, true_rpm=13000.0):
    from augur.physics.simulator import DroneSpec, FlightCondition, synth_waveform

    spec = DroneSpec(num_motors=4, prop_diameter_inch=5.0, prop_pitch_inch=3.0,
                     blade_count=2, c_t=0.11, c_p=0.05, pole_count=14,
                     efficiency=0.8, mass_kg=0.55, battery_wh=55.5,
                     drone_class="racing")
    sig, sr = synth_waveform(spec, FlightCondition(rpm=true_rpm, state="hover"),
                             duration_s=2.0, snr_db=25.0,
                             rng=np.random.default_rng(0))
    pcm = np.int16(np.clip(sig / np.max(np.abs(sig)), -1, 1) * 32767)
    wavfile.write(path, sr, pcm)
    return sr


def _mux_video(wav_path, video_path):
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", "color=black:s=64x64:d=2",
         "-i", str(wav_path), "-shortest", "-c:a", "aac", str(video_path)],
        check=True, capture_output=True,
    )


def _silent_video(video_path):
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", "color=black:s=64x64:d=1", str(video_path)],
        check=True, capture_output=True,
    )


def test_missing_ffmpeg_raises_with_hint(monkeypatch, tmp_path):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="ffmpeg"):
        audio_extract.extract_audio(tmp_path / "any.mp4", tmp_path / "out.wav")


@needs_ffmpeg
def test_video_without_audio_track_returns_none(tmp_path):
    video = tmp_path / "silent.mp4"
    _silent_video(video)
    assert audio_extract.extract_audio(video, tmp_path / "out.wav") is None


@needs_ffmpeg
def test_extracted_audio_recovers_rpm(tmp_path):
    from augur.pipeline import audio_to_inputs

    true_rpm = 13000.0
    wav = tmp_path / "drone.wav"
    _write_drone_wav(wav, true_rpm)
    video = tmp_path / "drone.mp4"
    _mux_video(wav, video)

    out = audio_extract.extract_audio(video, tmp_path / "extracted.wav")
    assert out is not None and out.exists()
    est, _state = audio_to_inputs(out, num_blades_hint=2,
                                  rng=np.random.default_rng(0))
    # AAC is lossy but preserves the blade-pass harmonics; 3% is generous.
    assert abs(est.rpm.median - true_rpm) / true_rpm < 0.03


@needs_ffmpeg
def test_predict_uses_video_audio_track(tmp_path):
    from augur import pipeline

    wav = tmp_path / "drone.wav"
    _write_drone_wav(wav)
    video = tmp_path / "drone.mp4"
    _mux_video(wav, video)

    report = pipeline.predict(video=str(video), n=2000, seed=0)
    assert "audio" in report.inputs_used
    assert any("audio track" in note for note in report.notes)


@needs_ffmpeg
def test_predict_explicit_audio_wins_over_video(tmp_path, monkeypatch):
    from augur import pipeline

    wav = tmp_path / "drone.wav"
    _write_drone_wav(wav)
    video = tmp_path / "drone.mp4"
    _mux_video(wav, video)

    def boom(*args, **kwargs):
        raise AssertionError("extract_audio must not run when --audio is given")

    monkeypatch.setattr(pipeline.audio_extract, "extract_audio", boom)
    report = pipeline.predict(audio=str(wav), video=str(video), n=2000, seed=0)
    assert "audio" in report.inputs_used


@needs_ffmpeg
def test_predict_silent_video_degrades_with_note(tmp_path):
    from augur import pipeline

    video = tmp_path / "silent.mp4"
    _silent_video(video)
    report = pipeline.predict(video=str(video), n=2000, seed=0)
    assert "audio" not in report.inputs_used
    assert any("no audio track" in note for note in report.notes)
