import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from video_splitter import build_description


def test_caption_only_no_speech():
    description = build_description("a man standing in a room", "")
    assert description == "a man standing in a room."


def test_caption_already_punctuated():
    description = build_description("A man standing in a room.", "")
    assert description == "A man standing in a room."


def test_caption_with_transcript():
    description = build_description("a man standing in a room", "Hello world!")
    assert description == 'a man standing in a room. Speech: "Hello world!"'


def test_transcript_only_when_caption_empty():
    description = build_description("", "Hello world!")
    assert description == 'Speech: "Hello world!"'


def test_empty_caption_and_transcript():
    assert build_description("", "") == ""
