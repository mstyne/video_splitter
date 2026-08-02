import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
VIDEO_PATH = FIXTURES_DIR / "test_video.mp4"
GROUND_TRUTH_PATH = FIXTURES_DIR / "ground_truth.json"

CLIP_LENGTH = 5.0
TOLERANCE = 3.0
GAP_EPSILON = 0.5  # VAD boundary imprecision allowance


def _ensure_fixture():
    if VIDEO_PATH.exists() and GROUND_TRUTH_PATH.exists():
        with open(GROUND_TRUTH_PATH) as f:
            return json.load(f)

    if shutil.which("say") is None:
        pytest.skip("macOS `say` command not available to generate speech fixture")

    sys.path.insert(0, str(TESTS_DIR))
    from generate_fixture import generate_fixture

    return generate_fixture()


def _cut_in_a_gap(cut, gaps, epsilon=GAP_EPSILON):
    return any(gap_start - epsilon <= cut <= gap_end + epsilon for gap_start, gap_end in gaps)


def test_full_pipeline_against_fixture(tmp_path):
    ground_truth = _ensure_fixture()
    gaps = ground_truth["silence_gaps"]
    duration = ground_truth["duration"]

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    shutil.copy(VIDEO_PATH, input_dir / "test_video.mp4")
    output_dir = tmp_path / "output"

    subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "video_splitter.py"),
            str(input_dir),
            "--output-folder", str(output_dir),
            "--clip-length", str(CLIP_LENGTH),
            "--tolerance", str(TOLERANCE),
            # Skip caption/transcript generation here: it needs Whisper + BLIP
            # model downloads and is covered separately by test_description.py.
            "--skip-description",
        ],
        check=True,
        cwd=REPO_ROOT,
    )

    manifest_path = output_dir / "test_video" / "manifest.json"
    assert manifest_path.exists()
    with open(manifest_path) as f:
        manifest = json.load(f)

    assert len(manifest) > 1

    # Clips are contiguous and cover the whole duration. ffprobe's container
    # duration can differ slightly (sub-frame) from the raw audio duration.
    assert manifest[0]["start"] == pytest.approx(0.0, abs=1e-6)
    assert manifest[-1]["end"] == pytest.approx(duration, abs=0.05)
    for prev_clip, next_clip in zip(manifest, manifest[1:]):
        assert prev_clip["end"] == pytest.approx(next_clip["start"], abs=1e-6)

    # Every clip but the last (remainder) respects the target length/tolerance.
    for clip in manifest[:-1]:
        clip_duration = clip["end"] - clip["start"]
        assert CLIP_LENGTH - TOLERANCE <= clip_duration <= CLIP_LENGTH + TOLERANCE

    # Every non-fallback cut should land inside (or very near) a known silence gap.
    non_fallback_cuts = [c["end"] for c in manifest[:-1] if not c["used_fallback"]]
    for cut in non_fallback_cuts:
        assert _cut_in_a_gap(cut, gaps), f"cut at {cut} not inside a known silence gap {gaps}"

    # Sanity check: with generous silence gaps, VAD should find at least one real gap.
    assert len(non_fallback_cuts) > 0

    # Output files actually exist and are non-trivial.
    for clip in manifest:
        clip_path = output_dir / "test_video" / clip["file"]
        assert clip_path.exists()
        assert clip_path.stat().st_size > 0
