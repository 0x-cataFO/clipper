"""Output formats: --format / --size, crop geometry, caption layout and end-to-end renders."""
import json
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest

import clipper as C
from conftest import ffprobe_streams


# ---------------------------------------------------------------- sizes

@pytest.mark.parametrize("text, size", [("720x1280", (720, 1280)), (" 1920 X 1080 ", (1920, 1080)),
                                        ("1080×1350", (1080, 1350))])
def test_parse_size(text, size):
    assert C.parse_size(text) == size


@pytest.mark.parametrize("text", ["720", "720x", "axb", "721x1280", "64x64", "5000x1080", ""])
def test_parse_size_rejects_bad_input(text):
    with pytest.raises(SystemExit):
        C.parse_size(text)


def test_formats_are_even_and_named_by_ratio():
    for name, (w, h) in C.FORMATS.items():
        a, b = map(int, name.split(":"))
        assert w % 2 == 0 and h % 2 == 0
        assert abs(w / h - a / b) < 0.01, name
    assert C.FORMATS[C.DEFAULT_FORMAT] == (1080, 1920)


# ---------------------------------------------------------------- crop geometry

@pytest.mark.parametrize("src", [(1920, 1080), (1280, 720), (1080, 1920), (1440, 1080)])
@pytest.mark.parametrize("out", list(C.FORMATS.values()) + [(720, 1280)])
def test_crop_box_fits_and_keeps_aspect(src, out):
    cw, ch = C.crop_box(*src, *out)
    assert cw <= src[0] and ch <= src[1]
    assert cw % 2 == 0 and ch % 2 == 0
    assert abs(cw / ch - out[0] / out[1]) < 0.01
    # it is the largest such window: one side touches the source edge
    assert cw >= src[0] - 2 or ch >= src[1] - 2


def test_crop_box_landscape_on_landscape_uses_whole_frame():
    assert C.crop_box(1920, 1080, 1920, 1080) == (1920, 1080)


# ---------------------------------------------------------------- caption layout

def test_caption_layout_default_matches_original_look():
    lay = C.caption_layout(1080, 1920)
    assert (lay["size"], lay["outline"], lay["shadow"], lay["margin_lr"], lay["margin_v"], lay["max_chars"]) \
        == (84, 7, 3, 70, 560, 18)


def test_caption_layout_scales_and_respects_explicit_size():
    assert C.caption_layout(720, 1280)["size"] == 56                  # 2/3 of 84 on a 720-wide frame
    assert C.caption_layout(1920, 1080)["size"] == 84                 # same height as 1080 wide
    assert C.caption_layout(1080, 1920, size=100)["size"] == 100
    # wider frames put captions proportionally lower
    lifts = [C.caption_layout(*C.FORMATS[f])["margin_v"] / C.FORMATS[f][1] for f in ("9:16", "4:5", "1:1", "16:9")]
    assert lifts == sorted(lifts, reverse=True)
    assert C.caption_layout(1920, 1080)["max_chars"] > 18             # wider frame fits more text


def test_write_ass_uses_output_size(tmp_path, words):
    p = tmp_path / "c.ass"
    C.write_ass(words, p, "Montserrat", None, "FFE500", False, 1920, 1080)
    text = p.read_text(encoding="utf-8")
    assert "PlayResX: 1920" in text and "PlayResY: 1080" in text
    style = re.search(r"Style: Caption,Montserrat,(\d+),", text)
    assert style and int(style.group(1)) == 84


# ---------------------------------------------------------------- end to end

pytestmark_e2e = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


class FaceAt:
    """Pretends there is always one face at a fixed x position (in source pixels)."""
    def __init__(self, x=1000):
        self.x = x

    def find(self, frame):
        return [(self.x, frame.shape[0] / 2, 5000.0)]


def _run(tmp_path, synthetic_video, monkeypatch, finder, *extra):
    out = tmp_path / "out"
    work = out / "talk"
    work.mkdir(parents=True, exist_ok=True)
    (work / "transcript.json").write_text(json.dumps(
        {"language": "en", "duration": synthetic_video["duration"], "words": synthetic_video["words"]}))
    monkeypatch.setattr(C, "FaceFinder", lambda: finder)
    monkeypatch.setattr(sys, "argv", ["clipper.py", str(synthetic_video["video"]), "-o", str(out), *extra])
    C.main()
    return work


@pytestmark_e2e
@pytest.mark.parametrize("args, size, suffix", [
    (["--format", "4:5"], (1080, 1350), "_1080x1350"),
    (["--format", "1:1"], (1080, 1080), "_1080x1080"),
    (["--format", "16:9"], (1920, 1080), "_1920x1080"),
    (["--size", "720x1280"], (720, 1280), "_720x1280"),
    (["--format", "1:1", "--size", "640x480"], (640, 480), "_640x480"),   # --size wins
    ([], (1080, 1920), ""),
])
def test_render_each_format(tmp_path, synthetic_video, monkeypatch, args, size, suffix):
    work = _run(tmp_path, synthetic_video, monkeypatch, FaceAt(), "--clips", "1-5", *args)
    clip = work / f"clip_01{suffix}.mp4"
    assert clip.exists()
    v = next(s for s in ffprobe_streams(clip) if s["codec_type"] == "video")
    assert (v["width"], v["height"]) == size
    info = json.loads((work / f"clips{suffix}.json").read_text())[0]
    assert (info["file"], info["width"], info["height"]) == (clip.name, *size)


def _frame(path, t, w, h):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(path), "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(h, w).astype(float)


@pytestmark_e2e
@pytest.mark.parametrize("fmt", ["1:1", "4:5"])
def test_crop_follows_the_face(tmp_path, synthetic_video, monkeypatch, fmt):
    """With a face on the right of a 1280x720 source, the square / 4:5 crop must sit on the right."""
    work = _run(tmp_path, synthetic_video, monkeypatch, FaceAt(x=1100), "--clips", "1-5",
                "--format", fmt, "--no-captions", "--no-tighten")
    ow, oh = C.FORMATS[fmt]
    cw, ch = C.crop_box(1280, 720, ow, oh)
    got = _frame(work / f"clip_01_{ow}x{oh}.mp4", 2.0, cw, ch)
    src = _frame(synthetic_video["video"], 3.0, 1280, 720)           # clip starts at 1s
    y0 = (720 - ch) // 2
    right = src[y0:y0 + ch, 1280 - cw:]
    centre = src[y0:y0 + ch, (1280 - cw) // 2:(1280 - cw) // 2 + cw]
    err_right = np.mean(np.abs(got - right))
    err_centre = np.mean(np.abs(got - centre))
    assert err_right < 12 and err_right < err_centre / 2, (err_right, err_centre)
