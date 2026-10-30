import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FPS = "30000/1001"
SR = 48000


def make_words():
    """Fake word-level transcript: five sentences with short gaps between words and
    deliberately long pauses (0.8-1.5s) between some sentences."""
    sentences = [
        ("Nobody tells you this about building a startup.", 0.2),
        ("The hardest part is not the code.", 1.2),
        ("It is getting the first users to trust you.", 0.2),
        ("We deleted half the product and retention tripled.", 1.5),
        ("Small numbers that grow every week beat big promises.", 0.8),
    ]
    words, t = [], 0.5
    for text, pause in sentences:
        for w in text.split():
            dur = 0.12 + 0.045 * len(w.strip(".,"))
            words.append({"start": round(t, 3), "end": round(t + dur, 3), "text": w})
            t += dur + 0.06
        t += pause
    return words, t + 0.6


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory):
    """A 1280x720 @ 29.97fps test video whose audio is a distinct tone for every word
    and silence everywhere else, plus the matching transcript."""
    d = tmp_path_factory.mktemp("media")
    words, duration = make_words()
    audio = np.zeros(int(duration * SR), dtype=np.float32)
    for i, w in enumerate(words):
        a, b = int(w["start"] * SR), int(w["end"] * SR)
        t = np.arange(b - a) / SR
        audio[a:b] = 0.4 * np.sin(2 * np.pi * (220 + 37 * i) * t)
    wav = d / "speech.wav"
    with wave.open(str(wav), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes((audio * 32767).astype(np.int16).tobytes())
    video = d / "talk.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", f"testsrc2=size=1280x720:rate={FPS}:duration={duration:.3f}",
         "-i", str(wav), "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-shortest", str(video)],
        check=True)
    return {"video": video, "words": words, "duration": duration}


@pytest.fixture
def words():
    return make_words()[0]


def ffprobe_streams(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_packets", "-print_format", "json", "-show_streams", str(path)],
        capture_output=True, text=True, check=True).stdout
    return json.loads(out)["streams"]


def decode_audio(path, sr=SR):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(sr),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32)
