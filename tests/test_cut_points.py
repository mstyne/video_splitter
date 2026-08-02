import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from video_splitter import compute_cut_points, silence_intervals_from_speech, quietest_point_in_window


def make_hops(duration, hop=0.02):
    times = np.arange(0, duration, hop) + hop / 2
    return times


def test_silence_intervals_from_speech_basic():
    speech = [(1.0, 2.0), (3.0, 4.0)]
    silences = silence_intervals_from_speech(speech, duration=5.0)
    assert silences == [(0.0, 1.0), (2.0, 3.0), (4.0, 5.0)]


def test_silence_intervals_no_speech():
    silences = silence_intervals_from_speech([], duration=5.0)
    assert silences == [(0.0, 5.0)]


def test_cut_point_lands_in_silence_gap():
    duration = 10.0
    speech = [(0.0, 4.7), (5.3, 10.0)]
    hop_times = make_hops(duration)
    hop_rms = np.ones_like(hop_times)

    clips = compute_cut_points(duration, speech, hop_times, hop_rms, clip_length=5.0, tolerance=1.0)

    assert len(clips) == 2
    start, cut, used_fallback = clips[0]
    assert start == 0.0
    assert not used_fallback
    assert 4.7 <= cut <= 5.3

    _, end, _ = clips[1]
    assert end == duration


def test_fallback_triggers_on_continuous_speech():
    duration = 12.0
    speech = [(0.0, 12.0)]  # no silence anywhere
    hop_times = make_hops(duration)
    hop_rms = np.ones_like(hop_times)
    quiet_idx = np.argmin(np.abs(hop_times - 5.0))
    hop_rms[quiet_idx] = 0.001  # a clear energy dip near the target

    clips = compute_cut_points(duration, speech, hop_times, hop_rms, clip_length=5.0, tolerance=1.0)

    start, cut, used_fallback = clips[0]
    assert used_fallback
    assert abs(cut - hop_times[quiet_idx]) < 0.05


def test_quietest_point_picks_minimum_rms():
    hop_times = np.array([4.0, 4.5, 5.0, 5.5, 6.0])
    hop_rms = np.array([1.0, 1.0, 0.01, 1.0, 1.0])
    point = quietest_point_in_window(4.0, 6.0, hop_times, hop_rms)
    assert point == 5.0


def test_final_remainder_clip_when_short():
    duration = 4.5
    speech = [(0.0, 4.5)]
    hop_times = make_hops(duration)
    hop_rms = np.ones_like(hop_times)

    clips = compute_cut_points(duration, speech, hop_times, hop_rms, clip_length=5.0, tolerance=1.0)

    assert len(clips) == 1
    assert clips[0] == (0.0, duration, False)
