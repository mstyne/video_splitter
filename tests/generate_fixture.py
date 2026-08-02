#!/usr/bin/env python3
"""
Builds a deterministic test video with known ground-truth speech/silence timing,
so the full video_splitter pipeline can be checked end-to-end against real data.

Uses macOS `say` to synthesize real speech (so Silero VAD has something genuine
to detect, not synthetic tones), pads it with precisely-measured silence, and
mixes in a plain color video track of matching duration.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
VIDEO_PATH = FIXTURES_DIR / "test_video.mp4"
GROUND_TRUTH_PATH = FIXTURES_DIR / "ground_truth.json"

SENTENCES = [
    "This is the first test sentence for the video splitter.",
    "Here comes a second sentence to check speech detection.",
    "And finally a short third sentence.",
]
LEAD_SILENCE = 0.5
GAP_SILENCE = 2.5
TRAIL_SILENCE = 0.5


def synthesize_sentence(sentence: str, tmp_dir: Path, index: int) -> np.ndarray:
    aiff_path = tmp_dir / f"speech_{index}.aiff"
    wav_path = tmp_dir / f"speech_{index}.wav"
    subprocess.run(["say", "-o", str(aiff_path), sentence], check=True, capture_output=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(aiff_path), "-ac", "1", "-ar", str(SAMPLE_RATE), str(wav_path)],
        check=True, capture_output=True,
    )
    samples, sr = sf.read(str(wav_path))
    assert sr == SAMPLE_RATE
    return samples


def generate_fixture():
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        speech_samples = [synthesize_sentence(s, tmp_dir, i) for i, s in enumerate(SENTENCES)]

        def silence(seconds):
            return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float64)

        chunks = [silence(LEAD_SILENCE)]
        silence_gaps = [(0.0, LEAD_SILENCE)]
        cursor = LEAD_SILENCE

        for i, speech in enumerate(speech_samples):
            chunks.append(speech)
            cursor += len(speech) / SAMPLE_RATE

            gap_len = TRAIL_SILENCE if i == len(speech_samples) - 1 else GAP_SILENCE
            chunks.append(silence(gap_len))
            silence_gaps.append((cursor, cursor + gap_len))
            cursor += gap_len

        combined = np.concatenate(chunks)
        duration = len(combined) / SAMPLE_RATE

        combined_wav = tmp_dir / "combined.wav"
        sf.write(str(combined_wav), combined, SAMPLE_RATE)

        video_only = tmp_dir / "video_only.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-f", "lavfi",
                "-i", f"color=c=blue:s=320x240:d={duration}:r=25",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(video_only),
            ],
            check=True, capture_output=True,
        )

        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(video_only), "-i", str(combined_wav),
                "-c:v", "copy", "-c:a", "aac", "-shortest",
                str(VIDEO_PATH),
            ],
            check=True, capture_output=True,
        )

    ground_truth = {
        "duration": duration,
        "silence_gaps": silence_gaps,
    }
    with open(GROUND_TRUTH_PATH, "w") as f:
        json.dump(ground_truth, f, indent=2)

    return ground_truth


if __name__ == "__main__":
    gt = generate_fixture()
    print(f"Generated {VIDEO_PATH} (duration={gt['duration']:.2f}s)")
    print(f"Silence gaps: {gt['silence_gaps']}")
    sys.exit(0)
