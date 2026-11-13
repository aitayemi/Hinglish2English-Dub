# Hinglish2English-Dub: Turn Mixed Hindi/English Training Videos into English-Dubbed Videos

> **GitHub summary (repo description):** Offline-first pipeline that extracts audio from mixed Hindi/English tutorial videos, transcribes it with Whisper, translates it to clean English with Claude, synthesises a male English voice with Piper TTS, and muxes the new audio back into the video with ffmpeg. Tested on CentOS Stream 9 (CPU only).

## Why this exists

I wanted to learn **Red Hat Satellite 6** from about 12 hours of YouTube tutorials spread across 8 videos. The videos were advertised as English, but the instructors spoke mostly Hindi with English technical terms mixed in. Rather than give up, I built a pipeline that produces an English-only version of each video.

## How it works

```
video.mp4
   │  ffmpeg (extract audio, no re-encode)
   ▼
video.aac
   │  Pass 1: faster-whisper (large-v3), transcribe in the original language(s)
   │  Pass 2: Claude translates each line to English, using neighbouring lines
   │          and a topic hint for context and technical terms
   ▼
video.srt  (English subtitles with the original timestamps)
   │  Piper TTS (male voice), speed up clips slightly if they overrun,
   │  place them on a timeline at each subtitle's start time
   ▼
video.dub.wav
   │  ffmpeg (copy video stream, replace audio stream)
   ▼
video.dubbed.mp4   (or overwrite the original with --replace)
```

| Script | Purpose |
|---|---|
| `audio_to_english_srt_twopass.py` | Audio to English SRT (Whisper pass + Claude translation pass) |
| `srt_to_dubbed_video.py` | English SRT to spoken audio track, merged into the video |

Both scripts **cache their progress**, so you can safely re-run them after an interruption.

## Tested environment

- Dell Latitude 5420, **no GPU**
- CentOS Stream 9
- Python 3 virtual environment

> **Performance warning:** Whisper `large-v3` on CPU is very slow. On this laptop, transcribing a **2-hour** audio file took roughly **20 hours**. Plan for that, or use a GPU, or switch to a smaller model (see [Tuning](#tuning)). Run long jobs inside `tmux` or `nohup` so they survive a disconnect.

## Prerequisites

- A Hugging Face account and access token (for model downloads without throttling)
- An Anthropic API key (for the translation pass; this is a paid API, so keep an eye on usage)
- Enough disk space for the intermediate files (WAV timelines and per-line clips are large)

## 1. Operating system packages

ffmpeg is not in the default CentOS Stream 9 repos, so enable EPEL, CRB and RPM Fusion first.

```bash
# EPEL and CodeReady Builder (CRB), required by RPM Fusion on EL9
sudo dnf install -y epel-release
sudo dnf config-manager --set-enabled crb

# RPM Fusion (free)
sudo dnf install -y --nogpgcheck \
  https://mirrors.rpmfusion.org/free/el/rpmfusion-free-release-$(rpm -E %rhel).noarch.rpm

# ffmpeg (includes ffprobe, which the dubbing script also uses)
sudo dnf install -y ffmpeg

# Verify
ffmpeg -version
ffprobe -version
python3 --version
```

If the stock Python 3 on your system is too old for a package below, install a newer interpreter (for example `sudo dnf install -y python3.11`) and use `python3.11 -m venv .venv` in the next step.

## 2. Python virtual environment

```bash
mkdir -p ~/satellite-dub && cd ~/satellite-dub
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Install only what the scripts actually import:

```bash
pip install -U faster-whisper srt anthropic numpy piper-tts

# Optional: online Microsoft voices instead of Piper
pip install -U edge-tts
```

`faster-whisper` pulls in `huggingface_hub` and `onnxruntime` automatically.

Create a `requirements.txt` if you want reproducible installs:

```
faster-whisper
srt
anthropic
numpy
piper-tts
edge-tts
```

> **Note:** You don't need `openai-whisper`, `pydub`, `pyaudioop`, `audioop-lts`, or the standalone `piper` package. The pipeline uses `faster-whisper` and `piper-tts`, and `ffmpeg` is a system binary, not a pip package. Installing extras can cause conflicts.

## 3. Download a Piper voice

Voices are downloaded into the **current directory**. Run this from the project folder, since the script's `PIPER_VOICE` points to a relative path.

```bash
# Male voice used by this project
python -m piper.download_voices en_US-hfc_male-medium

# Optional sanity check with another voice
python -m piper.download_voices en_US-lessac-medium
echo "Testing one two three" | piper -m en_US-lessac-medium.onnx -f /tmp/t.wav
ffplay -autoexit /tmp/t.wav
```

You should now have `en_US-hfc_male-medium.onnx` and `en_US-hfc_male-medium.onnx.json` in the folder. I chose a male voice because the instructors were male.

## 4. Credentials

Export the keys as environment variables in the shell where you run the scripts.

```bash
export HF_TOKEN="<your-hugging-face-token>"
export ANTHROPIC_API_KEY="<your-anthropic-api-key>"
```

- **HF_TOKEN:** Without it, Hugging Face throttles anonymous downloads of the Whisper model and shows a warning. Create a read-only token at <https://huggingface.co/settings/tokens>.
- **ANTHROPIC_API_KEY:** Required by pass 2. The script exits immediately if it is missing.

> **Never commit keys to the repo.** Don't put them in scripts, README files, or shell history you share. If a key is ever exposed, revoke it and create a new one. Consider storing them in a file outside the repo and running `source ~/.secrets/satellite-dub.env`.

Suggested `.gitignore`:

```
.venv/
*.mp4
*.aac
*.wav
*.onnx
*.onnx.json
*.tts/
*.transcript.json
*.translation.json
*.srt
.env
```

## 5. Extract audio from the videos

Extraction copies the audio stream without re-encoding, so it takes seconds. Adjust the names to match your files.

```bash
for f in vida vidb vid1 vid2 vid3 vid4 vid5; do
  ffmpeg -i "$f.mp4" -vn -c:a copy "$f.aac"
done
```

If a file's audio isn't AAC, `-c:a copy` to `.aac` will fail. Check with `ffprobe` and either change the extension to match (for example `.m4a` or `.mka`) or re-encode with `-c:a aac`.

## 6. Script 1: audio to English SRT

Configure the options at the top of `audio_to_english_srt_twopass.py`, especially `TOPIC_HINT`, which helps Claude restore technical terms (here: Capsule Server, content views, activation keys, and so on).

Key settings:

| Setting | Default | Meaning |
|---|---|---|
| `MODEL_SIZE` | `large-v3` | Whisper model. Use `medium` or `turbo` for speed |
| `DEVICE` / `COMPUTE_TYPE` | `auto` / `int8` | `int8` is the fast, low-RAM CPU option. Use `cuda` for an NVIDIA GPU |
| `SOURCE_LANGUAGE` | `hi` | Works well for Hinglish. `None` = auto-detect |
| `CLAUDE_MODEL` | `claude-sonnet-5-5` | Translation model. `claude-haiku-5-5` is cheaper and faster |
| `BATCH_SIZE` | `40` | Subtitle lines per API request |
| `CONTEXT_LINES` | `6` | Previously translated lines supplied for continuity |

Run it once per audio file (this is the slow step):

```bash
source .venv/bin/activate
export HF_TOKEN="<your-hugging-face-token>"
export ANTHROPIC_API_KEY="<your-anthropic-api-key>"

python audio_to_english_srt_twopass.py vida.aac
python audio_to_english_srt_twopass.py vidb.aac
# ...repeat for vid1 to vid5
```

Or unattended in the background:

```bash
nohup bash -c 'for f in vida vidb vid1 vid2 vid3 vid4 vid5; do
  python audio_to_english_srt_twopass.py "$f.aac"
done' > transcribe.log 2>&1 &
tail -f transcribe.log
```

Outputs per file (for `vida.aac`):

| File | Contents |
|---|---|
| `vida.transcript.json` | Pass 1 cache (original-language Whisper output) |
| `vida.original.srt` | Untranslated subtitles, for checking |
| `vida.translation.json` | Pass 2 cache (translated lines) |
| `vida.srt` | **Final English subtitles**, used by the next script |

If any batch fails after retries, the script keeps the original text for those lines and tells you how many; just re-run it to retry only the failed lines. Review `vida.srt` quickly before dubbing. Fixing a mistranslated product name here is much cheaper than fixing it after synthesis.

## 7. Script 2: SRT to dubbed video

`srt_to_dubbed_video.py` looks for `<video-name>.srt` next to each video. For every subtitle line it:

1. Synthesises speech with Piper (offline), edge-tts (online) or a test beep engine.
2. Speeds the clip up (never beyond `MAX_SPEEDUP`, default 1.35x) if it would run into the next line.
3. Places the clip on a timeline at the subtitle start time. Overruns push the next clip back slightly so clips never overlap.
4. Writes one audio track and swaps it into the video, copying the video stream without re-encoding.

Do a quick timing test first (no TTS needed):

```bash
python srt_to_dubbed_video.py vida.mp4 --engine beep
```

Then run the real thing. This produces `vida.dubbed.mp4`, leaving the original untouched:

```bash
python srt_to_dubbed_video.py vida.mp4 --voice en_US-hfc_male-medium.onnx
```

To process everything and **overwrite the originals** (what I did):

```bash
python srt_to_dubbed_video.py vida.mp4 vidb.mp4 vid1.mp4 vid2.mp4 vid3.mp4 vid4.mp4 vid5.mp4 \
  --replace --voice en_US-hfc_male-medium.onnx
```

> **Back up first.** `--replace` overwrites the original video. The script checks that the new file's duration matches the source before replacing, but a backup copy is cheap insurance. Alternatively, use `--keep-original-audio` to retain the old audio as a second, non-default track.

Options:

| Flag | Effect |
|---|---|
| `--engine {piper,edge,beep}` | Choose the speech engine (default `piper`) |
| `--voice` | Piper `.onnx` path, or an edge-tts voice name |
| `--srt` | Use a specific SRT (single video only) |
| `--replace` | Overwrite the original video |
| `--keep-original-audio` | Keep the old audio as a second track |
| `--keep-temp` | Keep the combined `.dub.wav` timeline |

The script prints a summary such as how many clips were sped up, the largest delay versus the subtitle start, and how many clips still overran. A large delay means the English is much longer than the Hindi it replaced; you can raise `MAX_SPEEDUP` slightly or shorten those lines in the SRT.

Synthesised clips are cached in a `<video>.tts/` folder, keyed on the line text and voice, so editing a few subtitle lines only re-synthesises those lines.

## Tuning

- **Too slow?** Set `MODEL_SIZE = "medium"` (or `"turbo"`), or run on a machine with an NVIDIA GPU with `DEVICE = "cuda"`.
- **Cheaper translation?** Switch `CLAUDE_MODEL` to `claude-haiku-5-5`.
- **Different voice?** Download another Piper voice, or use `--engine edge --voice en-GB-RyanNeural` (requires internet).
- **Dub sounds rushed?** Lower `MAX_SPEEDUP`; expect more drift against the video.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Hugging Face throttling or unauthenticated warning | Export `HF_TOKEN` before running |
| `Error: set the ANTHROPIC_API_KEY environment variable first.` | Export the key in the same shell |
| Piper produces no audio for the first lines | Wrong voice path, missing `.onnx.json`, or incompatible `piper-tts`; test with the `echo ... | piper` command above |
| `Command failed: ffmpeg ...` | ffmpeg not installed or RPM Fusion/CRB not enabled |
| `[!] vidX.srt not found - skipping` | Run script 1 first; the SRT must share the video's base name |
| `Only 1 core is available` warning | Your shell may be pinned; check `taskset -cp $$` |
| Output duration mismatch error | Source file may have odd timestamps; inspect with `ffprobe` |

## Limitations

- Machine translation of technical speech can still contain errors; skim the SRT for product names and commands.
- The voice is a single synthetic voice for all speakers, with no emotion or lip-sync.
- Intended for personal study of content you have the right to use; respect the original creators' licences.

## Licence

Choose a licence for your repo (MIT is common). Piper, Whisper and each Piper voice have their own licences, so check them if you redistribute anything.
