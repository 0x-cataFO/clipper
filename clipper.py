#!/usr/bin/env python3
"""
clipper.py - turn a long YouTube video (or local file) into captioned vertical clips.

Pipeline
  1. Download        yt-dlp (skipped for local files; cached on re-runs)
  2. Transcribe      faster-whisper with word-level timestamps (cached)
  3. Pick highlights Claude reads the transcript and returns the best segments (cached)
  4. Cut + tighten   ffmpeg cuts each segment and removes long pauses
  5. Reframe         OpenCV tracks the speaker's face and crops to 9:16, 4:5, 1:1 or 16:9
  6. Captions        word-by-word highlighted captions burned in (ASS / libass)

Requirements
  ffmpeg + ffprobe on PATH (built with libass - most builds are)
  pip install -r requirements.txt   (Python 3.10-3.13, 64-bit)
  export ANTHROPIC_API_KEY=...      (not needed if you pass --clips)
  Full documentation: README.md

Usage
  python clipper.py "https://www.youtube.com/watch?v=VIDEO_ID" -n 5
  python clipper.py podcast.mp4 -n 3 --min-sec 25 --max-sec 60
  python clipper.py podcast.mp4 --clips 12:05-13:20,41:00-42:10   # manual ranges, no API call
  python clipper.py URL --repick          # re-run highlight selection, reuse download + transcript
  python clipper.py URL --hook-rules my_hooks.md --uppercase --fonts-dir ./fonts
  python clipper.py URL --format 1:1          # 9:16 (default), 4:5, 1:1 or 16:9
  python clipper.py URL --size 720x1280       # any exact size

Output (in ./clips/<video-id>/)
  clip_01.mp4 ...   1080x1920 H.264/AAC, ranked best first
                    (other sizes are named clip_01_1080x1080.mp4 etc., so formats don't clash)
  clips.json        title, hook, score and source timestamps for each clip
  transcript.json   cached transcript      highlights.json   cached picks
  (delete transcript.json to re-transcribe, e.g. after changing --whisper-model)

Notes
  - Speaker tracking uses OpenCV's YuNet face model (~230KB, downloaded once to
    ~/.cache/clipper). If it can't be downloaded you still get clips, centre-cropped.
  - Captions use Montserrat if installed (free on Google Fonts); otherwise pass
    --fonts-dir with any .ttf/.otf and --font with its family name.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

FORMATS = {                     # --format name -> output size (width, height)
    "9:16": (1080, 1920),       # TikTok, Shorts, Reels
    "4:5": (1080, 1350),        # Instagram / Facebook feed
    "1:1": (1080, 1080),        # square posts
    "16:9": (1920, 1080),       # regular YouTube / landscape
}
DEFAULT_FORMAT = "9:16"
DEFAULT_CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")
YUNET_URL = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
CACHE_DIR = Path.home() / ".cache" / "clipper"
VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".m4v"}


# ----------------------------------------------------------------------------- helpers

def log(msg: str) -> None:
    print(msg, flush=True)


def die(msg: str) -> None:
    print(f"\nerror: {msg}", file=sys.stderr)
    sys.exit(1)


def run(cmd: list, cwd: Path | None = None) -> subprocess.CompletedProcess:
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, cwd=cwd)
    if r.returncode != 0:
        die(f"{cmd[0]} failed:\n{r.stderr.strip()[-3000:]}")
    return r


def safe_name(s: str) -> str:
    s = re.sub(r"[^\w.-]+", "_", s).strip("._")
    return s[:80] or "video"


def check_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            die(f"{tool} not found on PATH. Install ffmpeg first.")
    filters = run(["ffmpeg", "-hide_banner", "-filters"]).stdout
    if not re.search(r"^\s*\S+\s+ass\s", filters, re.M):
        die("your ffmpeg build has no libass, which is needed for captions. "
            "Install a full build (e.g. from ffmpeg.org / gyan.dev / `brew install ffmpeg`) "
            "or run with --no-captions.")


def probe(path: Path) -> dict:
    out = run(["ffprobe", "-v", "error", "-print_format", "json",
               "-show_streams", "-show_format", path]).stdout
    info = json.loads(out)
    v = next((s for s in info["streams"] if s.get("codec_type") == "video"), None)
    a = next((s for s in info["streams"] if s.get("codec_type") == "audio"), None)
    if v is None:
        die(f"{path} has no video stream")
    if a is None:
        die(f"{path} has no audio stream")
    fps = None
    for key in ("avg_frame_rate", "r_frame_rate"):
        try:
            f = Fraction(v.get(key, "0/0"))
        except (ValueError, ZeroDivisionError):
            continue
        if 1 <= f <= 120:
            fps = f.limit_denominator(1001)
            break
    if fps is None:
        fps = Fraction(30)
    duration = float(v.get("duration") or info["format"].get("duration") or 0)
    if duration <= 0:
        die(f"could not read the duration of {path}")
    return {"width": int(v["width"]), "height": int(v["height"]), "fps": fps,
            "duration": duration, "frames": round(duration * float(fps))}


def count_frames(path: Path) -> int:
    """Exact video frame count (demux only, fast)."""
    out = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
               "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", path]).stdout
    try:
        return int(out.strip().split(",")[0])
    except ValueError:
        die(f"could not count frames in {path}")


def fps_str(f: Fraction) -> str:
    return f"{f.numerator}/{f.denominator}"


def parse_time(s: str) -> float:
    total = 0.0
    for part in s.strip().split(":"):
        total = total * 60 + float(part)
    return total


def parse_size(s: str) -> tuple[int, int]:
    """'720x1280' -> (720, 1280). Sizes must be even (H.264 needs it) and sensible."""
    m = re.fullmatch(r"\s*(\d+)\s*[xX×]\s*(\d+)\s*", s or "")
    if not m:
        die(f"bad --size '{s}'. Use WIDTHxHEIGHT, e.g. 720x1280")
    w, h = int(m.group(1)), int(m.group(2))
    if not (128 <= w <= 4096 and 128 <= h <= 4096):
        die(f"--size {w}x{h} is out of range; width and height must be 128-4096")
    if w % 2 or h % 2:
        die(f"--size {w}x{h}: width and height must be even numbers")
    return w, h


def fmt_ts(t: float) -> str:
    m, s = divmod(int(round(t)), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# ----------------------------------------------------------------------------- 1. download

def fetch_source(src: str, out_root: Path) -> tuple[Path, Path, str]:
    """Returns (video_path, workdir, title)."""
    p = Path(src).expanduser()
    if p.is_file():
        workdir = out_root / safe_name(p.stem)
        workdir.mkdir(parents=True, exist_ok=True)
        return p.resolve(), workdir, p.stem

    if not re.match(r"https?://", src):
        die(f"'{src}' is neither an existing file nor a URL")
    try:
        import yt_dlp
    except ImportError:
        die("yt-dlp is not installed: pip install yt-dlp")

    base = {"quiet": True, "no_warnings": True, "noplaylist": True, "noprogress": True}
    try:
        with yt_dlp.YoutubeDL(base) as ydl:
            info = ydl.extract_info(src, download=False)
    except yt_dlp.utils.DownloadError as e:
        die(f"could not read video info: {e}")
    if info.get("_type") == "playlist":
        die("that URL is a playlist; pass a single video URL")

    vid = info.get("id") or "video"
    title = info.get("title") or vid
    workdir = out_root / safe_name(vid)
    workdir.mkdir(parents=True, exist_ok=True)

    cached = [f for f in sorted(workdir.glob("source.*")) if f.suffix.lower() in VIDEO_EXTS]
    if cached:
        log(f"[1/6] Using cached download: {cached[0].name}")
        return cached[0], workdir, title

    log(f"[1/6] Downloading: {title}")
    opts = dict(base,
                format="bv*[height<=1080][ext=mp4]+ba[ext=m4a]/bv*[height<=1080]+ba/b[height<=1080]/b",
                merge_output_format="mp4",
                outtmpl=str(workdir / "source.%(ext)s"))
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([src])
    except yt_dlp.utils.DownloadError as e:
        die(f"download failed: {e}")
    files = [f for f in sorted(workdir.glob("source.*")) if f.suffix.lower() in VIDEO_EXTS]
    if not files:
        die("download finished but no video file was found")
    return files[0], workdir, title


# ----------------------------------------------------------------------------- 2. transcribe

def transcribe(video: Path, cache: Path, model_size: str, language: str | None) -> list[dict]:
    if cache.exists():
        log("[2/6] Using cached transcript")
        return json.loads(cache.read_text(encoding="utf-8"))["words"]
    try:
        import ctranslate2
        from faster_whisper import WhisperModel
    except ImportError:
        die("faster-whisper is not installed: pip install faster-whisper")

    gpu = ctranslate2.get_cuda_device_count() > 0
    log(f"[2/6] Transcribing with whisper '{model_size}' on {'GPU' if gpu else 'CPU'} "
        "(first run downloads the model)")
    model = WhisperModel(model_size, device="cuda" if gpu else "cpu",
                         compute_type="float16" if gpu else "int8")
    segments, info = model.transcribe(str(video), language=language, beam_size=5,
                                      word_timestamps=True, vad_filter=True)
    words, last_pct = [], -10
    for seg in segments:  # generator: transcription happens while iterating
        for w in seg.words or []:
            text = w.word.strip()
            if text:
                start = float(w.start)
                words.append({"start": round(start, 3),
                              "end": round(max(float(w.end), start + 0.05), 3),
                              "text": text})
        if info.duration:
            pct = int(seg.end / info.duration * 100)
            if pct >= last_pct + 10:
                log(f"      {pct}%")
                last_pct = pct
    if not words:
        die("no speech found in the video")
    cache.write_text(json.dumps({"language": info.language, "duration": info.duration,
                                 "words": words}, ensure_ascii=False), encoding="utf-8")
    log(f"      {len(words)} words, language: {info.language}")
    return words


def build_lines(words: list[dict], max_words: int = 22, max_gap: float = 1.0) -> list[tuple[int, int]]:
    """Group words into sentence-like lines. Returns (first_word_idx, last_word_idx) pairs."""
    lines, first = [], 0
    for i, w in enumerate(words):
        nxt = words[i + 1] if i + 1 < len(words) else None
        sentence_end = w["text"].rstrip("\"')]»”’").endswith((".", "?", "!"))
        if (nxt is None or sentence_end or i - first + 1 >= max_words
                or nxt["start"] - w["end"] > max_gap):
            lines.append((first, i))
            first = i + 1
    return lines


# ----------------------------------------------------------------------------- 3. highlights

SYSTEM_PROMPT = """You are a senior short-form video editor. You find the segments of a long \
video that will perform best as standalone TikToks, YouTube Shorts and Reels.

A good clip:
- Opens with a hook in its first sentence: a bold claim, surprising fact, strong opinion, \
direct question, conflict or the start of a story. It never starts mid-thought \
("and so...", "that's why...", "like I said...").
- Is self-contained: a viewer with zero context understands it.
- Has a payoff: it ends on a punchline, insight or conclusion, never mid-sentence.
- Is between {min_s:.0f} and {max_s:.0f} seconds long (use the line timestamps).
- Does not overlap any other clip.

Score every clip 1-10 for viral potential and be harsh: only genuinely gripping moments \
deserve 8+. Return fewer clips rather than weak ones."""

CLIP_TOOL = {
    "name": "submit_clips",
    "description": "Submit the selected clips, best first.",
    "input_schema": {
        "type": "object",
        "properties": {
            "clips": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "start_line": {"type": "integer", "description": "L number of the first line"},
                        "end_line": {"type": "integer", "description": "L number of the last line (inclusive)"},
                        "title": {"type": "string", "description": "Short post title, under 70 characters"},
                        "hook": {"type": "string", "description": "The opening line that hooks the viewer"},
                        "score": {"type": "integer", "description": "Viral potential 1-10"},
                        "why": {"type": "string", "description": "One sentence on why this works"},
                    },
                    "required": ["start_line", "end_line", "title", "hook", "score"],
                },
            }
        },
        "required": ["clips"],
    },
}


def ask_claude(transcript_lines: list[str], title: str, n: int, min_s: float, max_s: float,
               model: str, hook_rules: str = "") -> list[dict]:
    try:
        import anthropic
    except ImportError:
        die("anthropic is not installed: pip install anthropic")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        die("ANTHROPIC_API_KEY is not set. Set it, or pass --clips to choose ranges manually.")

    client = anthropic.Anthropic()
    system = SYSTEM_PROMPT.format(min_s=min_s, max_s=max_s)
    if hook_rules.strip():
        system += ("\n\nThe creator's own hook rules. Follow them when judging and scoring clips:\n"
                   + hook_rules.strip())
    # Very long transcripts are split into chunks so each request stays well inside the context.
    budget, chunks, cur, size = 300_000, [], [], 0
    for line in transcript_lines:
        if cur and size + len(line) > budget:
            chunks.append(cur)
            cur, size = [], 0
        cur.append(line)
        size += len(line) + 1
    if cur:
        chunks.append(cur)

    picks = []
    for k, chunk in enumerate(chunks, 1):
        if len(chunks) > 1:
            log(f"      transcript part {k}/{len(chunks)}")
        prompt = (f"Video title: {title}\n\n"
                  f"Find up to {n} clips and submit them with the submit_clips tool.\n"
                  f"Transcript, one line per sentence as L<number> [start-end seconds] text:\n\n"
                  + "\n".join(chunk))
        try:
            resp = client.messages.create(
                model=model, max_tokens=8000, system=system,
                tools=[CLIP_TOOL], tool_choice={"type": "tool", "name": "submit_clips"},
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.APIError as e:
            die(f"Claude API error: {e}")
        block = next((b for b in resp.content if b.type == "tool_use"), None)
        if block is not None:
            picks.extend(c for c in block.input.get("clips", []) if isinstance(c, dict))
    return picks


def select_clips(raw: list[dict], words: list[dict], lines: list[tuple[int, int]],
                 n: int, min_s: float, max_s: float, duration: float) -> list[dict]:
    """Validate the model's picks: clamp, enforce length, drop overlaps, keep the top n."""
    def l_start(i): return words[lines[i][0]]["start"]
    def l_end(i): return words[lines[i][1]]["end"]

    candidates = []
    for c in raw:
        try:
            a, b = int(c["start_line"]), int(c["end_line"])
            score = float(c.get("score", 0))
        except (KeyError, TypeError, ValueError):
            continue
        if a > b:
            a, b = b, a
        if b < 0 or a >= len(lines):
            continue
        a, b = max(0, a), min(len(lines) - 1, b)
        while b > a and l_end(b) - l_start(a) > max_s:   # too long: drop trailing sentences
            b -= 1
        length = l_end(b) - l_start(a)
        if length > max_s * 1.15 or length < min_s * 0.6:
            continue
        candidates.append({
            "start": round(max(0.0, l_start(a) - 0.15), 3),
            "end": round(min(duration, l_end(b) + 0.35), 3),
            "title": str(c.get("title", "")).strip(),
            "hook": str(c.get("hook", "")).strip(),
            "score": score,
            "why": str(c.get("why", "")).strip(),
        })
    candidates.sort(key=lambda c: -c["score"])
    chosen = []
    for c in candidates:
        if all(c["end"] <= o["start"] or c["start"] >= o["end"] for o in chosen):
            chosen.append(c)
        if len(chosen) == n:
            break
    return chosen


def manual_clips(spec: str, duration: float) -> list[dict]:
    clips = []
    for part in filter(None, (p.strip() for p in spec.split(","))):
        try:
            a, b = (parse_time(x) for x in part.split("-", 1))
        except ValueError:
            die(f"bad --clips range '{part}'. Use start-end, e.g. 75-140 or 1:15-2:20")
        b = min(b, duration)
        if not 0 <= a < b:
            die(f"--clips range '{part}' is empty or outside the video (length {fmt_ts(duration)})")
        clips.append({"start": a, "end": b, "title": "", "hook": "", "score": 0, "why": "manual"})
    if not clips:
        die("--clips was given but contained no ranges")
    return clips


# ----------------------------------------------------------------------------- 4. cut + tighten

def keep_segments(words: list[dict], start: float, end: float, tighten: bool,
                  max_gap: float, pad: float = 0.12) -> list[tuple[float, float]]:
    """Speech intervals inside [start, end]; pauses longer than max_gap are cut out."""
    inside = [w for w in words if w["end"] > start and w["start"] < end]
    if not tighten or not inside:
        return [(start, end)]
    segs: list[list[float]] = []
    for w in inside:
        s, e = max(start, w["start"] - pad), min(end, w["end"] + pad)
        if segs and s - segs[-1][1] <= max_gap:
            segs[-1][1] = max(segs[-1][1], e)
        else:
            segs.append([s, e])
    # Trim dead air at the edges, but leave a little room so the first and last words
    # are not clipped (whisper timestamps are often a touch early on word endings).
    segs[0][0] = max(start, inside[0]["start"] - 0.15)
    segs[-1][1] = min(end, inside[-1]["end"] + 0.35)
    return [(s, e) for s, e in segs if e - s >= 0.05]


def plan_frames(segs: list[tuple[float, float]], fps: Fraction,
                total_frames: int) -> list[tuple[int, int]]:
    """Snap segments to whole source frames: [(first_frame, end_frame_exclusive), ...]."""
    fr = float(fps)
    frames = []
    for s, e in segs:
        a, b = max(0, round(s * fr)), min(total_frames, round(e * fr))
        if frames and a < frames[-1][1]:
            a = frames[-1][1]
        if b > a:
            frames.append((a, b))
    return frames


def cut_clip(src: Path, frames: list[tuple[int, int]], fps: Fraction, out: Path,
             words: list[dict]) -> list[dict]:
    """Cut the frame ranges and join them into one constant-frame-rate file. Every edge sits
    exactly on a source frame, so audio, video and captions stay locked together.
    Returns the words re-timed to the output timeline."""
    fr = float(fps)
    a0 = frames[0][0]
    # Seek half a frame before the first wanted frame so it is unambiguously the first decoded.
    ss = max(0.0, (a0 - 0.5) / fr)
    k = len(frames)
    graph = [f"[0:v]setpts=PTS-STARTPTS,fps=fps={fps_str(fps)},split={k}" + "".join(f"[vi{i}]" for i in range(k)),
             f"[0:a]asplit={k}" + "".join(f"[ai{i}]" for i in range(k))]
    for i, (a, b) in enumerate(frames):
        graph.append(f"[vi{i}]trim=start_frame={a - a0}:end_frame={b - a0},setpts=PTS-STARTPTS[v{i}]")
        graph.append(f"[ai{i}]atrim=start={a / fr - ss:.6f}:end={b / fr - ss:.6f},asetpts=PTS-STARTPTS[a{i}]")
    graph.append("".join(f"[v{i}][a{i}]" for i in range(k)) + f"concat=n={k}:v=1:a=1[v][a]")
    span = frames[-1][1] / fr - ss + 1.0
    run(["ffmpeg", "-y", "-v", "error", "-ss", f"{ss:.6f}", "-t", f"{span:.3f}", "-i", src,
         "-filter_complex", ";".join(graph), "-map", "[v]", "-map", "[a]",
         "-r", fps_str(fps), "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", out])

    # Re-time words onto the joined timeline.
    timed, offset = [], 0.0
    for a, b in frames:
        s, e = a / fr, b / fr
        for w in words:
            if s <= w["start"] < e:
                timed.append({"start": offset + w["start"] - s,
                              "end": offset + min(w["end"], e) - s,
                              "text": w["text"]})
        offset += (b - a) / fr
    return timed


# ----------------------------------------------------------------------------- 5. reframe

class FaceFinder:
    """YuNet (accurate, auto-downloaded ~230KB). Falls back to a Haar cascade (OpenCV 4.x only),
    and if neither is available, to a plain centre crop."""

    def __init__(self):
        self.yunet = None
        self.haar = None
        model = CACHE_DIR / "face_detection_yunet_2023mar.onnx"
        try:
            if not model.exists() or model.stat().st_size < 100_000:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                part = model.with_suffix(".part")
                urllib.request.urlretrieve(YUNET_URL, part)
                if part.stat().st_size < 100_000:
                    raise RuntimeError("model download was incomplete")
                part.replace(model)
            create = getattr(cv2, "FaceDetectorYN", None)
            create = create.create if create is not None else cv2.FaceDetectorYN_create
            self.yunet = create(str(model), "", (320, 320), 0.7, 0.3, 20)
        except Exception as e:  # network blocked, old OpenCV, etc.
            log(f"      (YuNet face detector unavailable: {e})")
            self.yunet = None
        if self.yunet is None and hasattr(cv2, "CascadeClassifier"):
            haar = cv2.CascadeClassifier(
                os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
            if not haar.empty():
                self.haar = haar
                log("      using the Haar cascade face detector instead")
        if self.yunet is None and self.haar is None:
            log("      WARNING: no face detector available, clips will use a centre crop. "
                f"Download {YUNET_URL} to {model} to enable speaker tracking.")

    def find(self, frame: np.ndarray) -> list[tuple[float, float, float]]:
        """Returns faces as (center_x, center_y, area) in original-frame pixels."""
        h, w = frame.shape[:2]
        scale = 640 / w if w > 640 else 1.0
        small = cv2.resize(frame, (round(w * scale), round(h * scale))) if scale != 1.0 else frame
        sh, sw = small.shape[:2]
        if self.yunet is not None:
            self.yunet.setInputSize((sw, sh))
            _, faces = self.yunet.detect(small)
            boxes = [] if faces is None else [f[:4] for f in faces]
        elif self.haar is None:
            return []
        else:
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            min_side = max(24, sh // 12)
            boxes = self.haar.detectMultiScale(gray, 1.1, 6, minSize=(min_side, min_side))
        out = []
        for x, y, bw, bh in boxes:
            if bw * bh <= 0:
                continue
            out.append(((x + bw / 2) / scale, (y + bh / 2) / scale, (bw * bh) / scale ** 2))
        return out


def median_filter(x: np.ndarray, k: int) -> np.ndarray:
    if len(x) < 3 or k < 3:
        return x.copy()
    r = k // 2
    padded = np.pad(x, r, mode="edge")
    return np.array([np.median(padded[i:i + k]) for i in range(len(x))])


def plan_camera(sample_frames: list[int], sample_cx: list[float], nframes: int,
                src_w: int, crop_w: int, sample_hz: float) -> np.ndarray:
    """Per-frame horizontal crop offset: steady framing, smooth pans, hard jumps on scene cuts."""
    center = np.full(nframes, src_w / 2)
    cx = np.array(sample_cx, dtype=float)
    if len(cx) == 0 or np.all(np.isnan(cx)):
        return np.clip(center - crop_w / 2, 0, src_w - crop_w).round().astype(int)

    # Hold the last known position through moments with no detected face.
    good = ~np.isnan(cx)
    first = cx[np.argmax(good)]
    for i in range(len(cx)):
        if np.isnan(cx[i]):
            cx[i] = cx[i - 1] if i else first

    cx = median_filter(cx, 5)                      # kill single-sample detection glitches
    cut_jump = 0.12 * src_w
    run_id = np.concatenate([[0], np.cumsum(np.abs(np.diff(cx)) > cut_jump)])

    smooth = np.empty_like(cx)
    dead = 0.05 * crop_w
    win = max(1, int(round(sample_hz * 0.8)))
    for r in np.unique(run_id):
        idx = np.where(run_id == r)[0]
        cam, path = cx[idx[0]], []
        for c in cx[idx]:                           # lazy camera with a dead zone
            if abs(c - cam) > dead:
                cam += (c - cam - math.copysign(dead, c - cam)) * 0.5
            path.append(cam)
        path = np.array(path)
        if len(path) > win:                         # moving average for smooth motion
            kernel = np.ones(win) / win
            path = np.convolve(np.pad(path, (win // 2, win - 1 - win // 2), mode="edge"), kernel, "valid")
        smooth[idx] = path

    frames = np.array(sample_frames)
    for f in range(nframes):
        j = int(np.searchsorted(frames, f))
        if j <= 0:
            center[f] = smooth[0]
        elif j >= len(frames):
            center[f] = smooth[-1]
        elif run_id[j - 1] == run_id[j]:            # same shot: interpolate
            f0, f1 = frames[j - 1], frames[j]
            t = (f - f0) / (f1 - f0)
            center[f] = smooth[j - 1] + t * (smooth[j] - smooth[j - 1])
        else:                                       # across a cut: jump, never pan
            center[f] = smooth[j - 1] if f - frames[j - 1] < frames[j] - f else smooth[j]
    return np.clip(center - crop_w / 2, 0, src_w - crop_w).round().astype(int)


def crop_box(src_w: int, src_h: int, out_w: int, out_h: int) -> tuple[int, int]:
    """Largest window with the output's aspect ratio that fits inside the source frame."""
    ar = out_w / out_h
    crop_w = max(2, min(src_w, round(src_h * ar)) // 2 * 2)
    crop_h = max(2, min(src_h, round(crop_w / ar)) // 2 * 2)
    return crop_w, crop_h


def render_clip(cut_path: Path, ass_path: Path | None, out_path: Path,
                finder: FaceFinder, fonts_dir: Path | None, out_w: int, out_h: int) -> None:
    info = probe(cut_path)
    fps = info["fps"]
    cap = cv2.VideoCapture(str(cut_path))
    if not cap.isOpened():
        die(f"OpenCV could not open {cut_path}")
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    crop_w, crop_h = crop_box(src_w, src_h, out_w, out_h)
    track = crop_w < src_w          # nothing to pan when the crop spans the full width

    # Pass 1: count frames and, if the crop can move, sample faces ~5 times a second.
    step = max(1, round(float(fps) / 5))
    sample_frames, sample_cx, prev, n = [], [], None, 0
    while True:
        if track and n % step == 0:
            ok, frame = cap.read()
            if not ok:
                break
            faces = finder.find(frame)
            if faces:
                def rank(f):   # prefer big faces, and stick with the current speaker
                    sticky = 1.6 if prev is not None and abs(f[0] - prev) < 0.1 * src_w else 1.0
                    return f[2] * sticky
                best = max(faces, key=rank)
                prev = best[0]
                sample_cx.append(best[0])
            else:
                sample_cx.append(float("nan"))
            sample_frames.append(n)
        elif not cap.grab():
            break
        n += 1
    nframes = n
    cap.release()
    if nframes == 0:
        die(f"no frames decoded from {cut_path}")

    y0 = (src_h - crop_h) // 2
    xs = plan_camera(sample_frames, sample_cx, nframes, src_w, crop_w, float(fps) / step)
    interp = cv2.INTER_AREA if crop_w > out_w else cv2.INTER_CUBIC

    # Pass 2: crop every frame and stream it to ffmpeg, which adds audio + captions.
    vf = []
    if ass_path is not None:
        vf = ["-vf", f"ass={ass_path.name}" + (f":fontsdir={fonts_dir.name}" if fonts_dir else "")]
    cmd = ["ffmpeg", "-y", "-v", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{out_w}x{out_h}", "-r", fps_str(fps), "-i", "-",
           "-i", cut_path.name, "-map", "0:v:0", "-map", "1:a:0", *vf,
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
           "-c:a", "copy", "-movflags", "+faststart", out_path.name]
    err_log = cut_path.with_suffix(".ffmpeg.log")
    with open(err_log, "w") as err:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=err, cwd=cut_path.parent)
        cap = cv2.VideoCapture(str(cut_path))
        f = 0
        try:
            while f < nframes:
                ok, frame = cap.read()
                if not ok:
                    break
                x0 = int(xs[f])
                crop = frame[y0:y0 + crop_h, x0:x0 + crop_w]
                proc.stdin.write(cv2.resize(crop, (out_w, out_h), interpolation=interp).tobytes())
                f += 1
            proc.stdin.close()
        except BrokenPipeError:
            pass
        finally:
            cap.release()
        code = proc.wait()
    if code != 0:
        die(f"ffmpeg failed while rendering {out_path.name}:\n{err_log.read_text()[-3000:]}")
    err_log.unlink(missing_ok=True)


# ----------------------------------------------------------------------------- 6. captions

def ass_time(t: float) -> str:
    cs = max(0, int(round(t * 100)))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def ass_color(hex_rgb: str) -> str:
    hex_rgb = hex_rgb.lstrip("#")
    if not re.fullmatch(r"[0-9a-fA-F]{6}", hex_rgb):
        die(f"bad colour '{hex_rgb}', use 6-digit hex like FFE500")
    r, g, b = hex_rgb[0:2], hex_rgb[2:4], hex_rgb[4:6]
    return f"&H00{b}{g}{r}&".upper()


def clean_word(t: str, upper: bool) -> str:
    t = t.replace("\\", "/").replace("{", "(").replace("}", ")").replace("\n", " ")
    return t.upper() if upper else t


def caption_layout(out_w: int, out_h: int, size: int | None = None) -> dict:
    """Caption style scaled to the output frame. The 9:16 1080x1920 look (84px text,
    sitting in the lower third) is the reference; other shapes keep the same proportions,
    and wider frames put captions lower so they stay clear of the speaker's face."""
    scale = min(out_w, out_h) / 1080
    ratio = out_h / out_w
    if ratio >= 1.5:        # tall: 9:16
        lift = 0.2917
    elif ratio >= 1.2:      # 4:5
        lift = 0.16
    elif ratio >= 0.95:     # square
        lift = 0.12
    else:                   # landscape: classic subtitle position
        lift = 0.08
    return {
        "size": size if size else max(12, round(84 * scale)),
        "outline": max(2, round(7 * scale)),
        "shadow": max(1, round(3 * scale)),
        "margin_lr": round(0.065 * out_w),
        "margin_v": round(lift * out_h),
        "max_chars": max(18, round(18 * out_w / (1080 * scale))),
    }


def write_ass(words: list[dict], path: Path, font: str, size: int | None, highlight: str,
              upper: bool, out_w: int = 1080, out_h: int = 1920, max_words: int = 3) -> None:
    hl = ass_color(highlight)
    lay = caption_layout(out_w, out_h, size)
    max_chars = lay["max_chars"]
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {out_w}
PlayResY: {out_h}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{font},{lay['size']},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,{lay['outline']},{lay['shadow']},2,{lay['margin_lr']},{lay['margin_lr']},{lay['margin_v']},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    # Group words into short punchy chunks.
    chunks, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        chars = sum(len(x["text"]) + 1 for x in cur)
        if (nxt is None or len(cur) >= max_words or chars + len(nxt["text"]) > max_chars
                or w["text"].endswith((".", ",", "?", "!", ":", ";"))
                or nxt["start"] - w["end"] > 0.5):
            chunks.append(cur)
            cur = []

    events = []
    for ci, chunk in enumerate(chunks):
        nxt_start = chunks[ci + 1][0]["start"] if ci + 1 < len(chunks) else None
        chunk_end = chunk[-1]["end"]
        if nxt_start is not None and 0 <= nxt_start - chunk_end < 0.4:
            chunk_end = nxt_start                    # no flicker between close chunks
        if nxt_start is not None:
            chunk_end = min(chunk_end, nxt_start)    # never overlap the next chunk
        for wi, w in enumerate(chunk):
            start = w["start"]
            end = chunk[wi + 1]["start"] if wi + 1 < len(chunk) else chunk_end
            if end - start < 0.01:
                continue
            text = " ".join(
                (f"{{\\c{hl}}}{clean_word(x['text'], upper)}{{\\c&H00FFFFFF&}}" if j == wi
                 else clean_word(x["text"], upper))
                for j, x in enumerate(chunk))
            events.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Caption,,0,0,0,,{text}")
    path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description="Turn a long video into short captioned clips.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("source", help="YouTube URL or local video file")
    ap.add_argument("-o", "--out", default="clips", help="output folder")
    ap.add_argument("-n", "--num-clips", type=int, default=5, help="how many clips to make")
    ap.add_argument("--min-sec", type=float, default=30, help="shortest clip (before pause removal)")
    ap.add_argument("--max-sec", type=float, default=90, help="longest clip (before pause removal)")
    ap.add_argument("--clips", help="manual ranges instead of Claude, e.g. 75-140,12:05-13:20")
    ap.add_argument("--repick", action="store_true", help="ignore cached highlight picks")
    ap.add_argument("--model", default=DEFAULT_CLAUDE_MODEL, help="Claude model for picking clips")
    ap.add_argument("--hook-rules", help="text/markdown file with your own hook rules for picking clips")
    ap.add_argument("--whisper-model", default="small",
                    help="tiny / base / small / medium / large-v3 (bigger = slower, more accurate)")
    ap.add_argument("--language", help="force transcript language, e.g. en (None = auto-detect)")
    ap.add_argument("--no-tighten", action="store_true", help="keep pauses")
    ap.add_argument("--max-pause", type=float, default=0.35, help="pauses longer than this are cut (s)")
    ap.add_argument("--format", choices=list(FORMATS), default=DEFAULT_FORMAT,
                    help="output shape: " + ", ".join(f"{k} = {w}x{h}" for k, (w, h) in FORMATS.items()))
    ap.add_argument("--size", help="exact output size as WIDTHxHEIGHT, e.g. 720x1280 (overrides --format)")
    ap.add_argument("--no-captions", action="store_true", help="skip burned-in captions")
    ap.add_argument("--font", default="Montserrat", help="caption font family (falls back if missing)")
    ap.add_argument("--fonts-dir", help="folder with .ttf/.otf files to use for captions")
    ap.add_argument("--font-size", type=int,
                    help="caption size in pixels (None = 84 on a 1080-wide frame, scaled to the output size)")
    ap.add_argument("--highlight", default="FFE500", help="active-word colour (hex RGB)")
    ap.add_argument("--uppercase", action="store_true", help="ALL-CAPS captions")
    ap.add_argument("--keep-temp", action="store_true", help="keep intermediate files")
    args = ap.parse_args()

    if args.num_clips < 1:
        die("--num-clips must be at least 1")
    if not 0 < args.min_sec < args.max_sec:
        die("--min-sec must be positive and smaller than --max-sec")
    if args.font_size is not None and args.font_size < 8:
        die("--font-size must be at least 8")
    out_w, out_h = parse_size(args.size) if args.size else FORMATS[args.format]
    # Default 9:16 keeps the plain names; other sizes get a suffix so formats don't overwrite each other.
    suffix = "" if (out_w, out_h) == FORMATS[DEFAULT_FORMAT] else f"_{out_w}x{out_h}"
    if args.no_captions:
        for tool in ("ffmpeg", "ffprobe"):
            if not shutil.which(tool):
                die(f"{tool} not found on PATH. Install ffmpeg first.")
    else:
        check_ffmpeg()

    video, workdir, title = fetch_source(args.source, Path(args.out))
    if Path(args.source).expanduser().is_file():
        log(f"[1/6] Using local file: {video.name}")
    src = probe(video)
    src["frames"] = count_frames(video)
    words = transcribe(video, workdir / "transcript.json", args.whisper_model, args.language)

    # 3. pick clips
    lines = build_lines(words)
    picks_cache = workdir / "highlights.json"
    if args.clips:
        log("[3/6] Using manual ranges")
        clips = manual_clips(args.clips, src["duration"])
    elif picks_cache.exists() and not args.repick:
        log("[3/6] Using cached highlight picks (pass --repick to choose again)")
        clips = json.loads(picks_cache.read_text(encoding="utf-8"))
    else:
        log(f"[3/6] Asking {args.model} for the best {args.num_clips} clips")
        transcript_lines = [
            f"L{i} [{words[a]['start']:.1f}-{words[b]['end']:.1f}] "
            + " ".join(w["text"] for w in words[a:b + 1])
            for i, (a, b) in enumerate(lines)]
        rules = ""
        if args.hook_rules:
            rules_path = Path(args.hook_rules).expanduser()
            if not rules_path.is_file():
                die(f"--hook-rules file '{rules_path}' not found")
            rules = rules_path.read_text(encoding="utf-8")
        raw = ask_claude(transcript_lines, title, args.num_clips, args.min_sec, args.max_sec,
                         args.model, rules)
        clips = select_clips(raw, words, lines, args.num_clips, args.min_sec, args.max_sec, src["duration"])
        if not clips:
            die("no usable clips came back. Try --repick, a wider --min-sec/--max-sec, or --clips.")
        picks_cache.write_text(json.dumps(clips, indent=2, ensure_ascii=False), encoding="utf-8")
    clips = clips[: args.num_clips] if not args.clips else clips

    tmp = workdir / "tmp"
    tmp.mkdir(exist_ok=True)
    fonts_dir = None
    if args.fonts_dir:   # copied next to the subtitle file so ffmpeg needs no path escaping
        user_fonts = Path(args.fonts_dir).expanduser()
        if not user_fonts.is_dir():
            die(f"--fonts-dir '{user_fonts}' is not a folder")
        fonts_dir = tmp / "fonts"
        fonts_dir.mkdir(exist_ok=True)
        for f in user_fonts.iterdir():
            if f.suffix.lower() in (".ttf", ".otf"):
                shutil.copy2(f, fonts_dir / f.name)

    finder = FaceFinder()
    log(f"      output size: {out_w}x{out_h}")
    results = []
    for i, c in enumerate(clips, 1):
        name = f"clip_{i:02d}{suffix}"
        log(f"[4-6/6] {name}: {fmt_ts(c['start'])}-{fmt_ts(c['end'])}"
            + (f"  \"{c['title']}\"" if c.get("title") else ""))
        segs = keep_segments(words, c["start"], c["end"], not args.no_tighten, args.max_pause)
        frames = plan_frames(segs, src["fps"], src["frames"])
        if not frames:
            log("      skipped (range is empty)")
            continue
        cut_path = tmp / f"{name}_cut.mp4"
        timed = cut_clip(video, frames, src["fps"], cut_path, words)
        ass_path = None
        if not args.no_captions and timed:
            ass_path = tmp / f"{name}.ass"
            write_ass(timed, ass_path, args.font, args.font_size, args.highlight, args.uppercase,
                      out_w, out_h)
        out_path = tmp / f"{name}.mp4"
        render_clip(cut_path, ass_path, out_path, finder, fonts_dir, out_w, out_h)
        final = workdir / f"{name}.mp4"
        shutil.move(str(out_path), final)
        length = probe(final)["duration"]
        results.append({"file": final.name, "title": c.get("title", ""), "hook": c.get("hook", ""),
                        "score": c.get("score", 0), "why": c.get("why", ""),
                        "source_start": c["start"], "source_end": c["end"],
                        "length_sec": round(length, 2), "width": out_w, "height": out_h})
        log(f"      -> {final}  ({length:.1f}s)")

    (workdir / f"clips{suffix}.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    if not args.keep_temp:
        shutil.rmtree(tmp, ignore_errors=True)
    log(f"\nDone: {len(results)} clip(s) in {workdir.resolve()}")


if __name__ == "__main__":
    main()
