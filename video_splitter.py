#!/usr/bin/env python3
import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from PIL import Image
from silero_vad import get_speech_timestamps, load_silero_vad
from transformers import BlipForConditionalGeneration, BlipProcessor

console = Console()
logger = logging.getLogger("video_splitter")

VAD_SAMPLE_RATE = 16000
RMS_HOP_SECONDS = 0.02
DEFAULT_WHISPER_MODEL = "base"
DEFAULT_CAPTION_MODEL = "Salesforce/blip-image-captioning-base"


def extract_audio(video_path: Path, wav_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-ac", "1", "-ar", str(VAD_SAMPLE_RATE), "-vn",
            str(wav_path),
        ],
        check=True,
        capture_output=True,
    )


def get_duration(video_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def silence_intervals_from_speech(speech_intervals, duration):
    """Complement of speech intervals: gaps before/after/between speech, in seconds."""
    silences = []
    cursor = 0.0
    for start, end in speech_intervals:
        if start > cursor:
            silences.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration:
        silences.append((cursor, duration))
    return silences


def rms_envelope(samples: np.ndarray, sample_rate: int, hop_seconds: float = RMS_HOP_SECONDS):
    """Return (times, rms) computed over non-overlapping hops."""
    hop_size = max(1, int(sample_rate * hop_seconds))
    n_hops = len(samples) // hop_size
    if n_hops == 0:
        return np.array([0.0]), np.array([0.0])
    trimmed = samples[: n_hops * hop_size].reshape(n_hops, hop_size)
    rms = np.sqrt(np.mean(trimmed.astype(np.float64) ** 2, axis=1))
    times = (np.arange(n_hops) + 0.5) * hop_size / sample_rate
    return times, rms


def quietest_point_in_window(window_start, window_end, hop_times, hop_rms):
    """Find the timestamp of minimum RMS energy within [window_start, window_end]."""
    mask = (hop_times >= window_start) & (hop_times <= window_end)
    if not np.any(mask):
        return (window_start + window_end) / 2.0
    idx = np.where(mask)[0]
    best = idx[np.argmin(hop_rms[idx])]
    return float(hop_times[best])


def compute_cut_points(duration, speech_intervals, hop_times, hop_rms, clip_length=5.0, tolerance=1.0):
    """
    Pure function: given the audio timeline description, decide where to cut.

    Returns a list of (start, end, used_fallback) tuples covering [0, duration].
    No ffmpeg/VAD calls happen here so this can be unit tested with synthetic data.
    """
    silences = silence_intervals_from_speech(speech_intervals, duration)

    clips = []
    current = 0.0
    while current < duration:
        target = current + clip_length
        if target >= duration:
            clips.append((current, duration, False))
            break

        window_start = max(current, target - tolerance)
        window_end = min(duration, target + tolerance)

        overlap = None
        for s_start, s_end in silences:
            overlap_start = max(s_start, window_start)
            overlap_end = min(s_end, window_end)
            if overlap_start < overlap_end:
                if overlap is None or abs((overlap_start + overlap_end) / 2 - target) < abs(
                    (overlap[0] + overlap[1]) / 2 - target
                ):
                    overlap = (overlap_start, overlap_end)

        if overlap is not None:
            cut = min(max(target, overlap[0]), overlap[1])
            used_fallback = False
        else:
            cut = quietest_point_in_window(window_start, window_end, hop_times, hop_rms)
            used_fallback = True

        clips.append((current, cut, used_fallback))
        current = cut

    return clips


def cut_clip(video_path: Path, start: float, end: float, out_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-ss", str(start), "-to", str(end),
            "-c:v", "libx264", "-preset", "veryfast",
            "-c:a", "aac", "-avoid_negative_ts", "make_zero",
            str(out_path),
        ],
        check=True,
        capture_output=True,
    )


def extract_frame(video_path: Path, timestamp: float, frame_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-ss", str(timestamp), "-i", str(video_path),
            "-frames:v", "1", str(frame_path),
        ],
        check=True,
        capture_output=True,
    )


def load_caption_model(model_name: str):
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    return processor, model


def caption_frame(caption_model, frame_path: Path) -> str:
    processor, model = caption_model
    image = Image.open(frame_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=30)
    return processor.decode(output[0], skip_special_tokens=True).strip()


def transcribe_audio(whisper_model: WhisperModel, samples: np.ndarray) -> str:
    if len(samples) == 0:
        return ""
    segments, _ = whisper_model.transcribe(samples, language="en")
    return " ".join(seg.text.strip() for seg in segments).strip()


def build_description(caption: str, transcript: str) -> str:
    """Pure function: merge a visual caption and a speech transcript into one description."""
    caption = caption.strip()
    if caption and not caption.endswith((".", "!", "?")):
        caption += "."
    if not transcript:
        return caption
    return f'{caption} Speech: "{transcript}"' if caption else f'Speech: "{transcript}"'


def describe_clip(clip_path: Path, clip_duration: float, clip_samples: np.ndarray, whisper_model, caption_model, tmp_dir: Path):
    frame_path = tmp_dir / f"{clip_path.stem}_frame.jpg"
    extract_frame(clip_path, clip_duration / 2, frame_path)
    caption = caption_frame(caption_model, frame_path)
    transcript = transcribe_audio(whisper_model, clip_samples)
    return caption, transcript, build_description(caption, transcript)


def process_video(video_path: Path, output_root: Path, vad_model, clip_length: float, tolerance: float, progress: Progress, run_timestamp: str, whisper_model=None, caption_model=None):
    logger.info("Processing [bold]%s[/bold]", video_path.name, extra={"markup": True})
    duration = get_duration(video_path)

    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = Path(tmp_dir) / "audio.wav"
        extract_audio(video_path, wav_path)

        samples, sample_rate = sf.read(str(wav_path), dtype="float32")
        assert sample_rate == VAD_SAMPLE_RATE

        wav_tensor = torch.from_numpy(samples)
        speech_timestamps = get_speech_timestamps(
            wav_tensor, vad_model, sampling_rate=VAD_SAMPLE_RATE, return_seconds=True
        )
        speech_intervals = [(seg["start"], seg["end"]) for seg in speech_timestamps]

        hop_times, hop_rms = rms_envelope(samples, sample_rate)

    clips = compute_cut_points(duration, speech_intervals, hop_times, hop_rms, clip_length, tolerance)

    out_dir = output_root / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    fallback_count = 0
    manifest = []
    describe = whisper_model is not None and caption_model is not None
    clip_task = progress.add_task(f"[green]{video_path.name}", total=len(clips))
    with tempfile.TemporaryDirectory() as frame_tmp_dir:
        for i, (start, end, used_fallback) in enumerate(clips, start=1):
            clip_duration = end - start
            if used_fallback:
                fallback_count += 1
                logger.warning("Clip %d (%.2fs-%.2fs) cut at quietest point, not true silence", i, start, end)
            out_path = out_dir / f"clip_{i:03d}_{run_timestamp}.mp4"
            cut_clip(video_path, start, end, out_path)

            delta = clip_duration - clip_length
            delta_color = "green" if abs(delta) <= tolerance / 2 else "yellow"
            logger.info(
                "Clip %d: %s (%.2fs, target %.2fs [%s]%+.2fs[/%s])",
                i, out_path.name, clip_duration, clip_length, delta_color, delta, delta_color,
                extra={"markup": True},
            )

            entry = {
                "file": out_path.name,
                "start": start,
                "end": end,
                "used_fallback": used_fallback,
            }

            if describe:
                start_idx = int(start * sample_rate)
                end_idx = int(end * sample_rate)
                clip_samples = samples[start_idx:end_idx]
                caption, transcript, description = describe_clip(
                    out_path, clip_duration, clip_samples, whisper_model, caption_model, Path(frame_tmp_dir)
                )
                txt_path = out_dir / f"clip_{i:03d}_{run_timestamp}.txt"
                txt_path.write_text(description + "\n")
                entry.update({"txt_file": txt_path.name, "caption": caption, "transcript": transcript})
                logger.info("  [dim]-> %s[/dim]", description, extra={"markup": True})

            manifest.append(entry)
            progress.advance(clip_task)

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    fallback_note = f"[yellow]{fallback_count} used quiet-point fallback[/yellow]" if fallback_count else "[green]0 fallbacks[/green]"
    logger.info(
        "%s: wrote %d clips to %s (%s)",
        video_path.name, len(clips), out_dir, fallback_note,
        extra={"markup": True},
    )


def find_videos(input_folder: Path, extensions):
    exts = {e.strip().lower() for e in extensions.split(",")}
    return sorted(
        p for p in input_folder.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )


def check_ffmpeg_available():
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        sys.exit(
            "ffmpeg/ffprobe not found on PATH. Install with `brew install ffmpeg` (macOS) "
            "or your platform's package manager, then re-run."
        )


def main():
    parser = argparse.ArgumentParser(description="Split videos into speech-aware clips of a target length.")
    parser.add_argument("input_folder", type=Path, help="Folder containing input videos")
    parser.add_argument("--output-folder", type=Path, default=Path("output"))
    parser.add_argument("--clip-length", type=float, default=5.0)
    parser.add_argument("--tolerance", type=float, default=1.0)
    parser.add_argument("--extensions", type=str, default=".mp4,.mov,.mkv,.3gp,.3gpp")
    parser.add_argument("--skip-description", action="store_true",
                         help="Skip generating per-clip .txt descriptions (visual caption + speech transcript)")
    parser.add_argument("--whisper-model", type=str, default=DEFAULT_WHISPER_MODEL,
                         help="faster-whisper model size (default: base)")
    parser.add_argument("--caption-model", type=str, default=DEFAULT_CAPTION_MODEL,
                         help="Hugging Face image-captioning model (default: %(default)s)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_time=False, show_path=False, markup=True)],
    )

    check_ffmpeg_available()

    videos = find_videos(args.input_folder, args.extensions)
    if not videos:
        logger.warning("No videos found in %s with extensions %s", args.input_folder, args.extensions)
        return

    with console.status("[bold cyan]Loading Silero VAD model..."):
        vad_model = load_silero_vad()

    whisper_model = None
    caption_model = None
    if not args.skip_description:
        with console.status(f"[bold cyan]Loading Whisper model ({args.whisper_model})..."):
            whisper_model = WhisperModel(args.whisper_model, device="cpu", compute_type="int8")
        with console.status(f"[bold cyan]Loading captioning model ({args.caption_model})..."):
            caption_model = load_caption_model(args.caption_model)

    args.output_folder.mkdir(parents=True, exist_ok=True)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        overall_task = progress.add_task("[cyan]Videos", total=len(videos))
        for video_path in videos:
            process_video(
                video_path, args.output_folder, vad_model, args.clip_length, args.tolerance,
                progress, run_timestamp, whisper_model=whisper_model, caption_model=caption_model,
            )
            progress.advance(overall_task)


if __name__ == "__main__":
    main()
