"""Tests for mac_captions.pipeline — pure logic, no heavy dependencies."""

from mac_captions.pipeline import (
    FRAME_BYTES,
    MAX_SPEECH_FRAMES,
    MIN_SPEECH_FRAMES,
    SILENCE_FRAMES,
    VadSegmenter,
    chunk_audio,
    postprocess_text,
)


# ---------------------------------------------------------------------------
# postprocess_text
# ---------------------------------------------------------------------------
class TestPostprocessText:
    def test_collapses_whitespace(self):
        assert postprocess_text("hello   world") == "hello world"

    def test_collapses_newlines(self):
        assert postprocess_text("hello\nworld") == "hello world"

    def test_strips_leading_trailing(self):
        assert postprocess_text("  hello  ") == "hello"

    def test_short_output_returns_silence_marker(self):
        # Single word under 4 meaningful chars
        assert postprocess_text("Mm") == "[ ... ]"

    def test_very_short_returns_silence_marker(self):
        assert postprocess_text("Ah") == "[ ... ]"

    def test_normal_sentence_passes_through(self):
        result = postprocess_text("El clima es agradable hoy.")
        assert result == "El clima es agradable hoy."

    def test_empty_string_returns_silence_marker(self):
        assert postprocess_text("") == "[ ... ]"

    def test_exactly_four_meaningful_chars_passes(self):
        # "okay" → 4 chars after strip, should pass through
        assert postprocess_text("okay") == "okay"

    def test_three_meaningful_chars_returns_silence(self):
        assert postprocess_text("ok!") == "[ ... ]"


# ---------------------------------------------------------------------------
# chunk_audio
# ---------------------------------------------------------------------------
class TestChunkAudio:
    def test_exact_fit(self):
        buf = b"\x00" * FRAME_BYTES * 3
        frames, leftover = chunk_audio(buf, b"", FRAME_BYTES)
        assert len(frames) == 3
        assert leftover == b""

    def test_leftover_carried(self):
        buf = b"\x01" * (FRAME_BYTES + 10)
        frames, leftover = chunk_audio(buf, b"", FRAME_BYTES)
        assert len(frames) == 1
        assert len(leftover) == 10

    def test_previous_leftover_prepended(self):
        leftover_in = b"\xaa" * (FRAME_BYTES - 1)
        buf = b"\xbb" * 1  # completes one frame with leftover_in
        frames, leftover = chunk_audio(buf, leftover_in, FRAME_BYTES)
        assert len(frames) == 1
        assert frames[0] == leftover_in + b"\xbb"
        assert leftover == b""

    def test_empty_buf_returns_empty(self):
        frames, leftover = chunk_audio(b"", b"", FRAME_BYTES)
        assert frames == []
        assert leftover == b""

    def test_smaller_than_frame_all_leftover(self):
        buf = b"\xff" * (FRAME_BYTES - 1)
        frames, leftover = chunk_audio(buf, b"", FRAME_BYTES)
        assert frames == []
        assert leftover == buf


# ---------------------------------------------------------------------------
# VadSegmenter
# ---------------------------------------------------------------------------
SPEECH_FRAME = b"\x10" * FRAME_BYTES
SILENT_FRAME = b"\x00" * FRAME_BYTES


def _push_n(segmenter: VadSegmenter, frame: bytes, is_speech: bool, n: int):
    """Push the same frame n times; return the last non-None result (or None)."""
    result = None
    for _ in range(n):
        r = segmenter.push(frame, is_speech)
        if r is not None:
            result = r
    return result


class TestVadSegmenter:
    def test_no_output_mid_speech(self):
        s = VadSegmenter()
        for _ in range(MIN_SPEECH_FRAMES - 1):
            assert s.push(SPEECH_FRAME, True) is None

    def test_emits_on_silence_after_speech(self):
        s = VadSegmenter()
        # Accumulate enough speech
        for _ in range(MIN_SPEECH_FRAMES):
            s.push(SPEECH_FRAME, True)
        # Then push silence until cut
        result = None
        for _ in range(SILENCE_FRAMES + 1):
            result = s.push(SILENT_FRAME, False)
            if result is not None:
                break
        assert result is not None
        assert isinstance(result, bytes)

    def test_emits_on_max_duration(self):
        s = VadSegmenter()
        result = None
        for _ in range(MAX_SPEECH_FRAMES + 1):
            result = s.push(SPEECH_FRAME, True)
            if result is not None:
                break
        assert result is not None

    def test_skips_low_speech_ratio(self):
        """A segment that is mostly silence should be discarded (returns None)."""
        s = VadSegmenter()
        # 1 speech frame, then many silent frames to trigger end-of-utterance
        s.push(SPEECH_FRAME, True)
        result = None
        for _ in range(SILENCE_FRAMES + 1):
            result = s.push(SILENT_FRAME, False)
            if result is not None:
                break
        # 1 speech out of (1 + SILENCE_FRAMES+1) total → well below MIN_SPEECH_RATIO
        assert result is None

    def test_resets_after_emit(self):
        """After a segment is emitted the segmenter should accept new speech cleanly."""
        s = VadSegmenter()
        # First segment
        for _ in range(MIN_SPEECH_FRAMES):
            s.push(SPEECH_FRAME, True)
        for _ in range(SILENCE_FRAMES + 1):
            s.push(SILENT_FRAME, False)

        # Second segment — should also emit
        for _ in range(MIN_SPEECH_FRAMES):
            s.push(SPEECH_FRAME, True)
        result = None
        for _ in range(SILENCE_FRAMES + 1):
            result = s.push(SILENT_FRAME, False)
            if result is not None:
                break
        assert result is not None

    def test_no_speech_never_emits(self):
        s = VadSegmenter()
        for _ in range(MAX_SPEECH_FRAMES * 2):
            assert s.push(SILENT_FRAME, False) is None
