"""Runs the whole pipeline on a synthetic video with manual ranges and a pre-seeded
transcript, so no network, Whisper model or API key is needed."""
import json
import shutil
import sys

import numpy as np
import pytest

import clipper as C
from conftest import SR, decode_audio, ffprobe_streams

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


class NoFaces:
    """Stand-in face finder so the test never downloads the YuNet model."""
    def find(self, frame):
        return []


@pytest.fixture
def run(tmp_path, synthetic_video, monkeypatch):
    def _run(*extra):
        out = tmp_path / "out"
        work = out / "talk"
        work.mkdir(parents=True, exist_ok=True)
        (work / "transcript.json").write_text(json.dumps(
            {"language": "en", "duration": synthetic_video["duration"], "words": synthetic_video["words"]}))
        monkeypatch.setattr(C, "FaceFinder", NoFaces)
        monkeypatch.setattr(sys, "argv", ["clipper.py", str(synthetic_video["video"]), "-o", str(out), *extra])
        C.main()
        return work
    return _run


def test_clips_are_correct(run, synthetic_video):
    words, video = synthetic_video["words"], synthetic_video["video"]
    end = synthetic_video["duration"]
    work = run("--clips", f"0-6,6-{end:.2f}", "--keep-temp")

    results = json.loads((work / "clips.json").read_text())
    assert [r["file"] for r in results] == ["clip_01.mp4", "clip_02.mp4"]

    src = C.probe(video)
    src_frames = C.count_frames(video)
    src_audio = decode_audio(video)
    removed = 0.0

    for r in results:
        path = work / r["file"]
        streams = ffprobe_streams(path)
        v = next(s for s in streams if s["codec_type"] == "video")
        a = next(s for s in streams if s["codec_type"] == "audio")

        # format
        assert (v["width"], v["height"]) == (1080, 1920)
        assert (v["codec_name"], v["pix_fmt"], a["codec_name"]) == ("h264", "yuv420p", "aac")

        # length matches the frame plan exactly, and audio is as long as video
        segs = C.keep_segments(words, r["source_start"], r["source_end"], True, 0.35)
        frames = C.plan_frames(segs, src["fps"], src_frames)
        planned = sum(b - a_ for a_, b in frames)
        assert int(v["nb_read_packets"]) == planned
        assert abs(float(v["duration"]) - float(a["duration"])) < 0.05

        span = r["source_end"] - r["source_start"]
        assert r["length_sec"] <= span + 0.05
        removed += span - r["length_sec"]

        # every kept stretch of audio sits exactly where the frame plan says
        out_audio = decode_audio(path)
        fr = float(src["fps"])
        offset = 0
        for a_, b in frames:
            s0, n = round(a_ / fr * SR), round((b - a_) / fr * SR)
            o0 = round(offset / fr * SR)
            ref, got = src_audio[s0:s0 + n], out_audio[o0:o0 + n]
            if np.std(ref) > 1e-3 and len(got) == n:
                corr = np.dot(ref, got) / (np.linalg.norm(ref) * np.linalg.norm(got) + 1e-9)
                assert corr > 0.97, f"audio out of place in segment {a_}-{b}: corr {corr:.3f}"
            offset += b - a_

        # captions exist for this clip
        assert (work / "tmp" / r["file"].replace(".mp4", ".ass")).exists()

    # the 1.2s and 1.5s pauses (minus padding around words) were cut out
    assert removed > 1.5


def test_no_captions_and_no_tighten(run, synthetic_video):
    work = run("--clips", "1-5", "--no-captions", "--no-tighten")
    r = json.loads((work / "clips.json").read_text())[0]
    assert abs(r["length_sec"] - 4.0) < 0.1          # nothing removed
    assert not (work / "tmp").exists()               # temp files cleaned up by default


def test_cached_rerun_uses_existing_files(run, capsys):
    run("--clips", "1-4", "--no-captions")
    run("--clips", "1-4", "--no-captions")
    assert "Using cached transcript" in capsys.readouterr().out
