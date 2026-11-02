# clipper

Turn a long YouTube video, podcast or livestream into short clips for TikTok, YouTube Shorts, Reels and feed posts.
Each clip is reframed around the speaker (vertical 9:16 by default, or 4:5, 1:1, 16:9 or any custom size), has its dead air cut out, and has word-by-word captions burned in.

```
python clipper.py "https://www.youtube.com/watch?v=VIDEO_ID" -n 5
```

```
clips/VIDEO_ID/
├── clip_01.mp4        1080×1920, H.264 + AAC, ranked best first
├── clip_02.mp4
├── ...
└── clips.json         title, hook, score and source timestamps for every clip
```

---

## Contents

- [What it does](#what-it-does)
- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [All options](#all-options)
- [Output files](#output-files)
- [Choosing the output size](#choosing-the-output-size)
- [Customising clip selection](#customising-clip-selection)
- [Customising captions](#customising-captions)
- [Troubleshooting](#troubleshooting)
- [Running the tests](#running-the-tests)
- [Responsible use](#responsible-use)
- [License](#license)

---

## What it does

| Step | What happens | Tool |
|---|---|---|
| 1. Download | Pulls the video (up to 1080p). Skipped for local files. | yt-dlp |
| 2. Transcribe | Speech-to-text with a timestamp for every word. Runs locally, no API needed. | faster-whisper |
| 3. Pick highlights | Claude reads the transcript and picks self-contained moments with a strong hook and a payoff. Optional: you can give the times yourself. | Anthropic API |
| 4. Cut and tighten | Cuts each clip and removes pauses longer than 0.35s. | ffmpeg |
| 5. Reframe | Crops to 9:16 (or the shape you choose), keeping the speaker's face centred with smooth pans and hard jumps on scene cuts. | OpenCV (YuNet) |
| 6. Captions | Short 1–3 word captions with the spoken word highlighted, burned into the video. | libass |

Every expensive step is cached. Re-running the same video reuses the download, the transcript and the picks, so tweaking captions or clip times takes seconds instead of minutes.

## How it works

**Clip selection.** The transcript is split into numbered sentence lines with timestamps and sent to Claude with a short-form editor brief (strong first-line hook, self-contained, ends on a payoff, within your length range). Claude returns its picks through a structured tool call. The script then validates every pick: it fixes reversed ranges, trims clips that run too long, drops ones that are too short or out of range, removes overlaps (keeping the higher score), and keeps the top `-n`.

**Frame-accurate cutting.** Each kept stretch of speech is snapped to whole source frames before cutting, and audio is trimmed to exactly the same boundaries. Captions are re-timed onto the new, shorter timeline from the same frame map. On the test video every output frame matched the expected source frame and audio stayed within 0.1 ms, so nothing drifts however many pauses are removed.

**Speaker tracking.** Faces are detected about five times a second. The camera follows the largest face and prefers the one it is already on, so it doesn't flick between people. Detection glitches are filtered out, small movements inside a dead zone are ignored, and real moves are smoothed into a gentle pan. A large sudden jump is treated as a scene cut and the frame jumps with it instead of panning across.

## Requirements

- **Python 3.10–3.13, 64-bit.** 3.12 is recommended. Some dependencies (onnxruntime, ctranslate2) don't ship for every newer or 32-bit Python yet.
- **ffmpeg** on your PATH, built with libass. Standard builds from ffmpeg.org, gyan.dev (Windows), Homebrew and most Linux distros include it.
- **A JavaScript runtime such as Deno**, which recent versions of yt-dlp need for YouTube downloads. Not needed for local files.
- **An Anthropic API key**, only if you want Claude to pick the clips. Without one, pass `--clips` with your own times.
- Disk space: about 500 MB for the default Whisper model, plus the downloaded video.

A GPU is optional. If an NVIDIA GPU with CUDA is available, transcription uses it automatically. Otherwise it runs on the CPU.

## Installation

### Windows (PowerShell)

```powershell
winget install Python.Python.3.12
winget install Gyan.FFmpeg
winget install DenoLand.Deno
# close and reopen PowerShell so the new tools are found

git clone https://github.com/0x-cataFO/clipper.git
cd clipper
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If `Activate.ps1` fails with *"running scripts is disabled on this system"*, run this once and try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### macOS

```bash
brew install python@3.12 ffmpeg deno
git clone https://github.com/0x-cataFO/clipper.git
cd clipper
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Linux (Debian / Ubuntu)

```bash
sudo apt install python3 python3-venv ffmpeg
curl -fsSL https://deno.land/install.sh | sh
git clone https://github.com/0x-cataFO/clipper.git
cd clipper
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Check the install

```bash
python -c "import cv2, faster_whisper, yt_dlp, anthropic; print('all good')"
ffmpeg -version
```

### Add your Anthropic API key (optional)

Create a key at [console.anthropic.com](https://console.anthropic.com), then:

```powershell
# Windows: saved for future windows; open a new PowerShell afterwards
setx ANTHROPIC_API_KEY "sk-ant-..."
```

```bash
# macOS / Linux: add this line to ~/.zshrc or ~/.bashrc to keep it
export ANTHROPIC_API_KEY="sk-ant-..."
```

API usage is pay-as-you-go and billed separately from a Claude subscription. Picking clips is one request per video (more for very long transcripts, which are split into parts).

## Usage

Activate the virtual environment each time you open a new terminal (`.venv\Scripts\Activate.ps1` on Windows, `source .venv/bin/activate` elsewhere).

**Let Claude pick the clips**

```bash
python clipper.py "https://www.youtube.com/watch?v=VIDEO_ID" -n 5
python clipper.py "https://www.youtube.com/watch?v=VIDEO_ID" -n 8 --min-sec 20 --max-sec 45
```

**Pick the clips yourself (no API key needed)**

Pass comma-separated `start-end` ranges. Times can be seconds (`75-140`), `m:ss` or `h:mm:ss`:

```bash
python clipper.py "https://www.youtube.com/watch?v=VIDEO_ID" --clips 2:15-3:30,14:05-15:10,1:02:10-1:02:50
```

**Clip a video file you already have**

```bash
python clipper.py "C:\Users\me\Videos\podcast.mp4" -n 3
python clipper.py ~/Videos/podcast.mp4 --clips 75-140
```

**Re-pick clips from the same video**

Picks are cached, so a second run on the same video reuses them. Use `--repick` to ask Claude again. The download and transcript are still reused.

```bash
python clipper.py "https://www.youtube.com/watch?v=VIDEO_ID" -n 5 --repick
```

**Faster or more accurate transcription**

```bash
python clipper.py VIDEO --whisper-model base       # faster, slightly less accurate
python clipper.py VIDEO --whisper-model large-v3   # most accurate, much slower on CPU
```

The transcript is cached in `transcript.json`. Delete it if you change `--whisper-model` and want to transcribe again.

**The first run** downloads the Whisper model (~500 MB for `small`) and transcribes the whole video. On a CPU this can take roughly as long as the video itself, or longer for big models. Later runs on the same video skip both steps.

## All options

| Option | Default | Description |
|---|---|---|
| `source` | — | YouTube URL or path to a local video file |
| `-o, --out` | `clips` | Output folder. Each video gets its own subfolder. |
| `-n, --num-clips` | `5` | How many clips Claude should pick (the most it will make) |
| `--min-sec` | `30` | Shortest clip, measured before pauses are removed |
| `--max-sec` | `90` | Longest clip, measured before pauses are removed |
| `--clips` | — | Your own ranges, e.g. `75-140,12:05-13:20`. Skips Claude entirely. |
| `--repick` | off | Ignore cached picks and ask Claude again |
| `--model` | `claude-sonnet-5-5` | Claude model used for picking. Can also be set with the `CLAUDE_MODEL` environment variable. |
| `--hook-rules` | — | Text or Markdown file with your own hook rules, added to Claude's brief |
| `--whisper-model` | `small` | `tiny`, `base`, `small`, `medium` or `large-v3` |
| `--language` | auto | Force the transcript language, e.g. `en` |
| `--no-tighten` | off | Keep pauses instead of cutting them |
| `--max-pause` | `0.35` | Pauses longer than this many seconds are cut |
| `--format` | `9:16` | Output shape: `9:16` (1080×1920), `4:5` (1080×1350), `1:1` (1080×1080) or `16:9` (1920×1080) |
| `--size` | — | Exact output size as `WIDTHxHEIGHT`, e.g. `720x1280`. Overrides `--format`. |
| `--no-captions` | off | Don't burn in captions |
| `--font` | `Montserrat` | Caption font family name |
| `--fonts-dir` | — | Folder of `.ttf`/`.otf` files to use for captions |
| `--font-size` | scaled | Caption size in pixels. By default 84 on a 1080-wide frame, scaled to the output size. |
| `--highlight` | `FFE500` | Colour of the word being spoken, as hex RGB |
| `--uppercase` | off | ALL-CAPS captions |
| `--keep-temp` | off | Keep intermediate files (cut clips, `.ass` subtitle files) in `tmp/` |

Run `python clipper.py --help` for the same list.

## Output files

Everything for one video goes in `clips/<video-id>/` (or `clips/<file-name>/` for local files):

| File | What it is |
|---|---|
| `clip_01.mp4`, `clip_02.mp4`, … | Finished 9:16 clips (1080×1920), ranked best first when Claude picked them |
| `clip_01_1080x1080.mp4`, … | Clips in any other size. The size is added to the name so formats don't overwrite each other. |
| `clips.json` | For each clip: file, title, hook, score, Claude's reason, source start/end, final length and size. Other sizes get their own file, e.g. `clips_1080x1080.json`. |
| `source.mp4` | The downloaded video (YouTube only). Reused on re-runs. |
| `transcript.json` | Word-level transcript. Reused on re-runs. |
| `highlights.json` | Claude's validated picks. Reused unless you pass `--repick`. |

Example `clips.json` entry:

```json
{
  "file": "clip_01.mp4",
  "title": "Why we deleted half our product",
  "hook": "Nobody tells you this about building a startup",
  "score": 9,
  "why": "Personal story with a clear payoff",
  "source_start": 0.65,
  "source_end": 35.02,
  "length_sec": 30.66,
  "width": 1080,
  "height": 1920
}
```

## Choosing the output size

Clips are vertical 9:16 by default. Pick another shape with `--format`, or any exact size with `--size`:

| Option | Size | Good for |
|---|---|---|
| `--format 9:16` (default) | 1080×1920 | TikTok, YouTube Shorts, Reels, Stories |
| `--format 4:5` | 1080×1350 | Instagram and Facebook feed posts |
| `--format 1:1` | 1080×1080 | Square posts, LinkedIn, X |
| `--format 16:9` | 1920×1080 | Regular YouTube, keeping the full original frame |
| `--size 720x1280` | anything | Smaller files or a platform's exact spec. Width and height must be even, 128–4096. |

```bash
python clipper.py VIDEO --format 1:1
python clipper.py VIDEO --format 16:9 --clips 2:15-3:30
python clipper.py VIDEO --size 720x1280
```

- The crop is always the largest window of that shape that fits the source, and it follows the speaker's face. If the window already spans the full width (for example 16:9 from a 16:9 video), there is nothing to pan, so the whole frame is used.
- Captions scale with the frame and sit lower on wider shapes (about 29% up on 9:16, 8% up on 16:9) to stay clear of faces. `--font-size` overrides the scaled size.
- You can make several shapes from one video: run the command again with a different `--format`. The download, transcript and picks are reused, and each size gets its own file names.

## Customising clip selection

Claude's built-in brief asks for clips that:

- open with a hook (bold claim, surprising fact, question, conflict or the start of a story), never mid-thought
- make sense to someone with no context
- end on a punchline, insight or conclusion, never mid-sentence

Add your own rules with `--hook-rules`. They are appended to the brief and used when Claude scores clips. See [`examples/hook_rules.md`](examples/hook_rules.md) for a starting point:

```bash
python clipper.py VIDEO -n 6 --hook-rules examples/hook_rules.md
```

## Customising captions

Captions show 1–3 words at a time (at most about 18 characters), positioned low in the frame with a thick outline and shadow. Size and position scale with the output shape (see [Choosing the output size](#choosing-the-output-size)). The word being spoken is highlighted.

```bash
# Yellow-highlighted, all caps, bigger
python clipper.py VIDEO --uppercase --font-size 96

# Your own font and colour
python clipper.py VIDEO --fonts-dir ./fonts --font "Poppins" --highlight 00E5FF
```

`--font` must be the font's family name (what your system's font viewer shows), not its file name. If the font isn't found, libass falls back to a default font and the clip still renders.

## Troubleshooting

**`ResolutionImpossible` … `onnxruntime` / `no matching distributions available`**
Your virtual environment was built with a Python that onnxruntime doesn't support (often a very new or 32-bit Python). Rebuild it on 64-bit Python 3.12:

```powershell
deactivate
Remove-Item -Recurse -Force .venv        # macOS/Linux: rm -rf .venv
py -3.12 -m venv .venv                   # macOS/Linux: python3.12 -m venv .venv
.venv\Scripts\Activate.ps1               # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

**`THESE PACKAGES DO NOT MATCH THE HASHES` during install**
A download was cut off or corrupted. It is not tampering. Clear the cache and retry with a longer timeout:

```bash
pip cache purge
pip install --no-cache-dir --default-timeout=180 --retries 10 -r requirements.txt
```

If the large OpenCV download keeps failing, install `opencv-python-headless` instead of `opencv-python`. The script works with either.

**`TypeError: open() got an unexpected keyword argument 'metadata_errors'`**
PyAV 19 removed an option that faster-whisper still uses. `requirements.txt` pins it, but if you installed packages separately:

```bash
pip install "av<19"
```

**YouTube download errors (`Sign in to confirm`, `HTTP Error 403`, `Requested format is not available`)**
YouTube changes often. Update yt-dlp first, and make sure Deno (or another JavaScript runtime) is installed:

```bash
pip install -U yt-dlp
```

**`your ffmpeg build has no libass`**
Install a full ffmpeg build (see [Requirements](#requirements)), or run with `--no-captions`.

**`YuNet face detector unavailable` … `clips will use a centre crop`**
The small face model couldn't be downloaded, so clips are centre-cropped instead of following the speaker. The message prints the URL and the folder to save it in, so you can download it manually.

**Warnings about `huggingface_hub` symlinks or `HF_TOKEN` (Windows)**
Harmless. The model cache uses a bit more disk space without symlinks. To hide them, set `HF_HUB_DISABLE_SYMLINKS_WARNING=1`.

**A clip starts or ends mid-word**
Nudge that range by a second in `--clips` and run again. The download and transcript are reused, so this is quick.

**Transcription is very slow**
Use `--whisper-model base` or `tiny`. Transcribing runs once per video and is then cached.

## Running the tests

The tests generate a short synthetic video with ffmpeg, so they need no network, API key or Whisper model.

```bash
pip install pytest
pytest -v
```

They check clip validation, pause removal, caption timing, the frame-accurate cutting (frame counts and audio/video lengths) that each finished clip is H.264/AAC at the requested size for every format, and that the crop follows the speaker in non-vertical shapes. They also run automatically on GitHub Actions for every push (`.github/workflows/tests.yml`).

## Responsible use

Only clip videos you own or have permission to use, such as creators who run clipping programs or campaigns, or who have said yes. Reposting other people's content without permission can get your accounts struck or demonetised, and may infringe copyright. This tool doesn't check permissions for you.

## License

[MIT](LICENSE)
