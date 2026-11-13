#!/usr/bin/env python3
"""English SRT -> spoken English audio track -> merged back into the video.

For every subtitle line the script:
  1. synthesises speech (Piper = local/offline, or edge-tts = online, or a test "beep" engine),
  2. speeds the clip up slightly (never more than MAX_SPEEDUP) if it would not fit before the next line,
  3. places it on a timeline at the subtitle's start time (a line that overruns pushes the next one back a
     little, and later lines catch up by being sped up, so clips never overlap),
then writes the whole timeline as one audio file and replaces the video's audio stream with it
(video is copied, not re-encoded).

Usage:   python srt_to_dubbed_video.py lecture.mp4                  # uses lecture.srt
         python srt_to_dubbed_video.py vid1.mp4 vid2.mp4 vid3.mp4   # several files, one after another
         python srt_to_dubbed_video.py lecture.mp4 --engine beep    # quick timing test, no TTS needed
         python srt_to_dubbed_video.py lecture.mp4 --replace        # overwrite the original video

Setup:   sudo dnf install ffmpeg            (needs RPM Fusion on Rocky/RHEL)
         pip install -U srt numpy piper-tts          # offline voice (recommended)
         python -m piper.download_voices en_US-lessac-medium
         # or, for the online Microsoft voices:  pip install -U edge-tts
Both the synthesised clips and the timeline are cached, so re-running after an interruption is cheap.
"""

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import wave

import numpy as np
import srt

# --- CONFIGURATION ---
ENGINE = "piper"                         # "piper" (offline), "edge" (online), "beep" (timing test only)
PIPER_VOICE = "en_US-hfc_male-medium.onnx" # path to the downloaded Piper voice (.onnx; its .json sits beside it)
EDGE_VOICE = "en-US-GuyNeural"           # e.g. en-US-AriaNeural, en-IN-NeerjaNeural, en-GB-RyanNeural
SAMPLE_RATE = 24000                      # sample rate of the final audio track
MAX_SPEEDUP = 1.35                       # fastest a clip may be played back (1.0 = never speed up)
GAP = 0.08                               # seconds of silence kept between consecutive clips
AUDIO_BITRATE = "160k"                   # AAC bitrate of the new audio track
# ---------------------


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"Command failed: {' '.join(cmd)}\n{r.stderr[-1500:]}")
    return r.stdout


def media_duration(path):
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "default=nw=1:nk=1", path])
    return float(out.strip())


def clean_text(content):
    text = re.sub(r"<[^>]+>|\{[^}]*\}", "", content)      # drop <i>..</i> and {\an8} style tags
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------- speech engines
def write_silence(path, rate, seconds=0.3):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(rate)
        wf.writeframes(np.zeros(int(rate * seconds), dtype=np.int16).tobytes())


class Piper:
    """Works with the different piper-tts APIs (1.3+ synthesize_wav, 1.3 generator style, older wav_file style).
    A line that produces no audio at all (nothing speakable in it) becomes a short silence instead of a crash."""
    def __init__(self, voice_path):
        import inspect
        from piper import PiperVoice
        self.voice = PiperVoice.load(voice_path)
        self.inspect = inspect
        self.rate = int(self.voice.config.sample_rate)
        self.calls = 0
        self.empty = 0

    def _synth(self, text, out_wav):
        v = self.voice
        with wave.open(out_wav, "wb") as wf:
            if hasattr(v, "synthesize_wav"):                       # piper-tts 1.3+
                v.synthesize_wav(text, wf)
                return
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(self.rate)
            if "wav_file" in self.inspect.signature(v.synthesize).parameters:   # old style
                v.synthesize(text, wf)
            else:                                                  # generator style
                for chunk in v.synthesize(text):
                    wf.writeframes(chunk.audio_int16_bytes)

    def __call__(self, text, out_wav):
        self.calls += 1
        try:
            self._synth(text, out_wav)
            if raw_duration(out_wav) > 0:
                return
        except wave.Error:
            pass                                                   # Piper produced no audio for this text
        self.empty += 1
        print(f"\n    [Warning] Piper produced no audio for: {text!r} - using silence")
        if self.calls >= 5 and self.empty == self.calls:
            sys.exit("[Error] Piper has produced no audio for the first 5 lines, so the voice itself is not "
                     "working (missing espeak-ng data, wrong voice file, or an incompatible piper-tts). "
                     "Test it alone:  echo 'Testing one two three' | piper -m <voice>.onnx -f /tmp/t.wav")
        write_silence(out_wav, self.rate)


class Edge:
    def __init__(self, voice):
        import edge_tts
        self.edge_tts, self.voice = edge_tts, voice

    def __call__(self, text, out_wav):
        mp3 = out_wav + ".mp3"
        for attempt in range(4):
            try:
                asyncio.run(self.edge_tts.Communicate(text, self.voice).save(mp3))
                break
            except Exception as e:
                if attempt == 3:
                    raise
                print(f"    [Warning] edge-tts retry after {type(e).__name__}: {e}")
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp3, out_wav])
        os.remove(mp3)


class Beep:
    """Test engine: a quiet tone whose length follows the text length (about 15 characters per second)."""
    def __call__(self, text, out_wav):
        n = int(max(0.4, len(text) / 15.0) * 22050)
        t = np.arange(n) / 22050
        tone = (0.15 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
        with wave.open(out_wav, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(22050)
            wf.writeframes(tone.tobytes())


def make_engine(args):
    if args.engine == "piper":
        return Piper(args.voice or PIPER_VOICE)
    if args.engine == "edge":
        return Edge(args.voice or EDGE_VOICE)
    return Beep()


# ---------------------------------------------------------------- audio helpers
def fit_clip(raw_wav, speed, out_wav):
    """Convert to the final sample rate / mono / 16-bit, applying the speed-up in the same step."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", raw_wav]
    if speed > 1.005:
        cmd += ["-filter:a", f"atempo={speed:.4f}"]
    cmd += ["-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", out_wav]
    run(cmd)


def read_pcm(path):
    with wave.open(path, "rb") as wf:
        return np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)


def raw_duration(path):
    with wave.open(path, "rb") as wf:
        return wf.getnframes() / wf.getframerate()


# ---------------------------------------------------------------- main work
def dub(video, srt_path, args):
    base, ext = os.path.splitext(video)
    srt_path = srt_path or f"{base}.srt"
    if not os.path.exists(srt_path):
        print(f"[!] {srt_path} not found - skipping {video}")
        return False

    with open(srt_path, encoding="utf-8") as f:
        subs = [s for s in srt.parse(f.read()) if clean_text(s.content)]
    if not subs:
        print(f"[!] {srt_path} has no subtitle text - skipping")
        return False

    video_len = media_duration(video)
    tts_dir = f"{base}.tts"
    os.makedirs(tts_dir, exist_ok=True)
    print(f"[+] {video}: {len(subs)} lines, {video_len / 60:.1f} min, engine={args.engine}")

    # 1) synthesise every line (cached per line; cache is keyed on the text so edited lines are redone)
    engine = None
    meta_path = os.path.join(tts_dir, "texts.json")
    old = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
    old_engine = old.get("__engine__")
    cache = old.get("lines", {}) if old_engine == f"{args.engine}:{args.voice or ''}" else {}
    for n, s in enumerate(subs, start=1):
        text = clean_text(s.content)
        raw = os.path.join(tts_dir, f"{n:05d}.raw.wav")
        if cache.get(str(n)) == text and os.path.exists(raw):
            continue
        if engine is None:
            engine = make_engine(args)
        engine(text, raw)
        cache[str(n)] = text
        if n % 20 == 0 or n == len(subs):
            print(f"    synthesised {n}/{len(subs)} lines", end="\r")
            json.dump({"__engine__": f"{args.engine}:{args.voice or ''}", "lines": cache},
                      open(meta_path, "w", encoding="utf-8"), ensure_ascii=False)
    json.dump({"__engine__": f"{args.engine}:{args.voice or ''}", "lines": cache},
              open(meta_path, "w", encoding="utf-8"), ensure_ascii=False)
    print()

    # 2) place the clips on a timeline
    total = int((video_len + 1.0) * SAMPLE_RATE)
    timeline = np.zeros(total, dtype=np.int16)
    cursor = 0.0
    sped = 0
    max_drift = 0.0
    overruns = 0
    for n, s in enumerate(subs, start=1):
        raw = os.path.join(tts_dir, f"{n:05d}.raw.wav")
        dur = raw_duration(raw)
        start = s.start.total_seconds()
        placed = max(start, cursor)
        next_start = subs[n].start.total_seconds() if n < len(subs) else video_len
        slot = next_start - placed - GAP

        speed = 1.0
        if dur > slot:
            speed = MAX_SPEEDUP if slot < 0.2 else min(dur / slot, MAX_SPEEDUP)
        fitted = os.path.join(tts_dir, f"{n:05d}.fit.wav")
        fit_clip(raw, speed, fitted)
        pcm = read_pcm(fitted)
        clip_len = len(pcm) / SAMPLE_RATE

        i0 = int(placed * SAMPLE_RATE)
        if i0 >= total:
            overruns += 1
            continue
        i1 = min(total, i0 + len(pcm))
        timeline[i0:i1] = pcm[:i1 - i0]
        cursor = placed + clip_len + GAP
        max_drift = max(max_drift, placed - start)
        sped += speed > 1.005
        if cursor > next_start + 0.5:
            overruns += 1
        if n % 20 == 0 or n == len(subs):
            print(f"    placed {n}/{len(subs)} clips", end="\r")
    print()
    print(f"[+] {sped} clip(s) sped up (max x{MAX_SPEEDUP}); largest delay vs. subtitle start: "
          f"{max_drift:.1f}s; clips still running into the next line: {overruns}")

    master = f"{base}.dub.wav"
    with wave.open(master, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
        wf.writeframes(timeline.tobytes())

    # 3) swap the audio stream
    out = f"{base}.dubbed{ext}"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-i", master,
           "-map", "0:v:0", "-map", "1:a:0"]
    if args.keep_original_audio:
        cmd += ["-map", "0:a:0?", "-disposition:a:1", "0"]
    ext_l = ext.lower()
    if ext_l in (".mp4", ".m4v", ".mov", ".mkv"):          # carry over any subtitle tracks the video already has
        cmd += ["-map", "0:s?", "-c:s", "copy" if ext_l == ".mkv" else "mov_text"]
    cmd += ["-c:v", "copy", "-c:a", "aac", "-b:a", AUDIO_BITRATE,
            "-metadata:s:a:0", "language=eng", "-disposition:a:0", "default",
            "-shortest", out]
    run(cmd)

    out_len = media_duration(out)
    if abs(out_len - video_len) > 2.0:
        sys.exit(f"[Error] {out} is {out_len:.1f}s long but the video is {video_len:.1f}s - not replacing anything.")
    print(f"[Success] {out}")

    if args.replace:
        os.replace(out, video)
        print(f"[+] Original overwritten: {video}")
    if not args.keep_temp:
        os.remove(master)
    return True


def main():
    p = argparse.ArgumentParser(description="English SRT -> spoken audio -> replace the video's audio")
    p.add_argument("videos", nargs="+", help="video file(s); the SRT is expected next to each, same base name")
    p.add_argument("--srt", help="SRT file to use (only valid with a single video)")
    p.add_argument("--engine", choices=["piper", "edge", "beep"], default=ENGINE)
    p.add_argument("--voice", help="Piper .onnx path, or an edge-tts voice name")
    p.add_argument("--replace", action="store_true", help="overwrite the original video with the dubbed one")
    p.add_argument("--keep-original-audio", action="store_true", help="keep the old audio as a second (non-default) track")
    p.add_argument("--keep-temp", action="store_true", help="keep the combined .dub.wav timeline file")
    args = p.parse_args()
    if args.srt and len(args.videos) != 1:
        sys.exit("--srt can only be used with a single video")
    for v in args.videos:
        dub(v, args.srt, args)


if __name__ == "__main__":
    main()
