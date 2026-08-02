# Video Splitter

[![Tests](https://github.com/mstyne/video_splitter/actions/workflows/tests.yml/badge.svg)](https://github.com/mstyne/video_splitter/actions/workflows/tests.yml)

Splits MP4/MOV/MKV files into ~5 second clips without cutting words mid-speech.
Uses Silero VAD to find speech/silence boundaries, and falls back to cutting
at the quietest point in the target window when no true silence gap is found.

## Prerequisites

`ffmpeg` and `ffprobe` must be on your `PATH`. On macOS:

```
brew install ffmpeg
```

## Setup

```
python3 -m venv .venv
pip install -r requirements.txt -r requirements-dev.txt
```

## Usage

```
python video_splitter.py <input_folder> [--output-folder output] [--clip-length 5.0] [--tolerance 1.0] [--extensions .mp4,.mov,.mkv,.3gp,.3gpp]
```

- `--clip-length`: target clip length in seconds (default `5.0`).
- `--tolerance`: how far a cut point may drift from the target length while
  searching for a silence gap (default `1.0`, so clips land between ~4s and ~6s).
- Output is written to `<output-folder>/<video_stem>/clip_001_<run_timestamp>.mp4`, ...
  one subfolder per input video. The run timestamp is shared across all clips in a
  run so re-running the script doesn't overwrite a previous run's output.

Clips are re-encoded (not stream-copied) so cut points are frame/sample accurate.
When the log shows a `WARNING` for a clip, that cut used the quiet-point fallback
(no true silence gap was found in the tolerance window), so it may occasionally
clip a word.

## Clip descriptions

By default, each clip also gets a matching `.txt` file describing its content:
a visual caption of a mid-clip frame (via a BLIP image-captioning model) plus a
speech-to-text transcript of the clip's audio (via `faster-whisper`), e.g.:

```
a man standing in a room. Speech: "Hello world!"
```

- `--skip-description`: skip this step entirely (faster, no extra model downloads).
- `--whisper-model`: faster-whisper model size (default `base`).
- `--caption-model`: Hugging Face image-captioning model (default
  `Salesforce/blip-image-captioning-base`).

The captioning and transcription models are downloaded from Hugging Face on first
use, so an internet connection is required the first time you run with description
enabled.

## Tests

```
pytest tests/
```

Unit tests (`test_cut_points.py`) exercise the cut-point algorithm with synthetic
data. `test_pipeline.py` generates a synthetic fixture video with known
ground-truth silence gaps (via `generate_fixture.py`, using macOS `say` for real
speech) and runs the full pipeline end-to-end against it.

## License

Licensed under the [Apache License 2.0](LICENSE.md).

## Authorship

This code was written by Claude (model `claude-sonnet-5`, reasoning effort `low`),
Anthropic's coding assistant, working with the repo owner in Claude Code.
