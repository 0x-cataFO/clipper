"""Fast tests for the pure-Python parts: no ffmpeg, network or API needed."""
import re
from fractions import Fraction

import pytest

import clipper as C


# ---------------------------------------------------------------- small helpers

@pytest.mark.parametrize("text, seconds", [
    ("75", 75), ("1:15", 75), ("1:02:05", 3725), ("12.5", 12.5), (" 0:30 ", 30),
])
def test_parse_time(text, seconds):
    assert C.parse_time(text) == seconds


def test_fmt_ts():
    assert C.fmt_ts(75) == "1:15"
    assert C.fmt_ts(3725) == "1:02:05"


def test_ass_helpers():
    assert C.ass_color("FFE500") == "&H0000E5FF&"     # RGB -> ASS BGR
    assert C.ass_color("#00e5ff") == "&H00FFE500&"
    assert C.ass_time(3725.456) == "1:02:05.46"
    assert C.clean_word("{x}\\y", True) == "(X)/Y"     # no ASS override injection
    with pytest.raises(SystemExit):
        C.ass_color("yellow")


# ---------------------------------------------------------------- manual ranges

def test_manual_clips():
    clips = C.manual_clips("1:05-1:08, 3-9", duration=100)
    assert [(c["start"], c["end"]) for c in clips] == [(65, 68), (3, 9)]
    assert C.manual_clips("90-999", duration=100)[0]["end"] == 100   # clamped to video end


@pytest.mark.parametrize("spec", ["abc", "50-40", "200-210", ","])
def test_manual_clips_rejects_bad_ranges(spec):
    with pytest.raises(SystemExit):
        C.manual_clips(spec, duration=100)


# ---------------------------------------------------------------- transcript lines

def test_build_lines_splits_on_sentences(words):
    lines = C.build_lines(words)
    assert len(lines) == 5
    for a, b in lines:
        assert words[b]["text"].endswith(".")
    assert lines[0][0] == 0 and lines[-1][1] == len(words) - 1


def test_build_lines_caps_length():
    words = [{"start": i * 0.3, "end": i * 0.3 + 0.2, "text": "word"} for i in range(50)]
    assert all(b - a + 1 <= 22 for a, b in C.build_lines(words))


# ---------------------------------------------------------------- clip validation

def _line_times(words):
    lines = C.build_lines(words)
    return lines, [(words[a]["start"], words[b]["end"]) for a, b in lines]


def test_select_clips_validates_model_output(words):
    lines, _ = _line_times(words)
    raw = [
        {"start_line": 0, "end_line": 3, "title": "best", "hook": "h", "score": 9},
        {"start_line": 2, "end_line": 4, "title": "overlaps best", "hook": "h", "score": 7},
        {"start_line": "4", "end_line": "4", "title": "strings", "hook": "h", "score": "8"},
        {"start_line": 99, "end_line": 120, "title": "out of range", "score": 10},
        {"title": "missing lines", "score": 10},
        "not a dict",
    ]
    out = C.select_clips(raw, words, lines, n=5, min_s=2, max_s=30, duration=60)
    assert [c["title"] for c in out] == ["best", "strings"]
    for a, b in zip(out, out[1:]):
        assert a["end"] <= b["start"] or b["end"] <= a["start"]


def test_select_clips_fixes_reversed_ranges(words):
    lines, _ = _line_times(words)
    out = C.select_clips([{"start_line": 2, "end_line": 0, "title": "r", "score": 5}],
                         words, lines, n=5, min_s=2, max_s=30, duration=60)
    assert len(out) == 1 and out[0]["start"] < out[0]["end"]


def test_select_clips_trims_to_max_length(words):
    lines, times = _line_times(words)
    max_s = times[1][1] - times[0][0] + 0.1          # room for lines 0-1 only
    out = C.select_clips([{"start_line": 0, "end_line": 4, "title": "long", "score": 9}],
                         words, lines, n=5, min_s=1, max_s=max_s, duration=60)
    assert len(out) == 1
    assert out[0]["end"] - out[0]["start"] <= max_s + 0.5   # + small lead-in / tail room


def test_select_clips_drops_too_short_and_limits_count(words):
    lines, _ = _line_times(words)
    raw = [{"start_line": i, "end_line": i, "title": str(i), "score": 10 - i} for i in range(5)]
    assert C.select_clips(raw, words, lines, n=5, min_s=20, max_s=30, duration=60) == []
    assert len(C.select_clips(raw, words, lines, n=2, min_s=0.5, max_s=30, duration=60)) == 2


# ---------------------------------------------------------------- pause removal

def test_keep_segments_cuts_only_long_pauses(words):
    start, end = 0.0, words[-1]["end"] + 0.5
    segs = C.keep_segments(words, start, end, tighten=True, max_gap=0.35)
    # the transcript has two pauses longer than 0.35s between sentences (1.2s and 1.5s)
    assert len(segs) == 3
    assert all(s < e for s, e in segs)
    assert all(a[1] < b[0] for a, b in zip(segs, segs[1:]))
    # every word is fully inside a kept segment
    for w in words:
        assert any(s <= w["start"] and w["end"] <= e for s, e in segs), w


def test_keep_segments_without_tightening(words):
    assert C.keep_segments(words, 1.0, 5.0, tighten=False, max_gap=0.35) == [(1.0, 5.0)]


# ---------------------------------------------------------------- frame planning

def test_plan_frames_snaps_and_never_overlaps():
    fps = Fraction(30000, 1001)
    frames = C.plan_frames([(0.0, 1.0), (0.99, 2.0), (5.0, 99.0)], fps, total_frames=200)
    assert frames[0][0] == 0
    assert all(isinstance(a, int) and isinstance(b, int) and a < b for a, b in frames)
    assert all(x[1] <= y[0] for x, y in zip(frames, frames[1:]))
    assert frames[-1][1] == 200                       # clamped to the end of the video


# ---------------------------------------------------------------- captions

def test_write_ass_timing_and_highlight(tmp_path, words):
    path = tmp_path / "c.ass"
    C.write_ass(words, path, font="Montserrat", size=84, highlight="FFE500", upper=False)
    text = path.read_text(encoding="utf-8")
    assert "PlayResX: 1080" in text and "PlayResY: 1920" in text
    events = re.findall(r"Dialogue: 0,([\d:.]+),([\d:.]+),Caption,,0,0,0,,(.*)", text)

    def sec(s):
        h, m, x = s.split(":")
        return int(h) * 3600 + int(m) * 60 + float(x)

    spans = [(sec(a), sec(b), t) for a, b, t in events]
    assert all(a < b for a, b, _ in spans)
    assert all(x[1] <= y[0] + 1e-9 for x, y in zip(spans, spans[1:])), "caption events overlap"

    hl = re.compile(r"\{\\c&H0000E5FF&\}(.*?)\{\\c&H00FFFFFF&\}")
    for w in words:                                  # the spoken word is the highlighted one
        t = w["start"] + 0.02
        live = [txt for a, b, txt in spans if a <= t < b]
        assert live and hl.findall(live[0]) == [w["text"]], w


def test_write_ass_chunks_are_short(tmp_path, words):
    path = tmp_path / "c.ass"
    C.write_ass(words, path, font="X", size=84, highlight="FFFFFF", upper=True)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Dialogue"):
            plain = re.sub(r"\{.*?\}", "", line.split(",,0,0,0,,", 1)[1])
            assert len(plain.split()) <= 3 and plain == plain.upper()
