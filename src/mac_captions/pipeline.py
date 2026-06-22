"""
Pure pipeline logic — no heavy dependencies (no mlx_audio, no sounddevice).

This module is safe to import on any platform and is the target for unit tests.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Audio constants (must match live.py)
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16_000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 320 samples
FRAME_BYTES = FRAME_SAMPLES * 2  # int16 = 2 bytes/sample

VAD_AGGRESSIVENESS = 3
SILENCE_FRAMES = 25  # ~500 ms trailing silence triggers a cut
MIN_SPEECH_FRAMES = 25  # ~500 ms minimum segment length
MAX_SPEECH_FRAMES = 100  # ~2 s max before a forced cut
MIN_SPEECH_RATIO = 0.45  # at least 45% of frames must be VAD-classified speech


# ---------------------------------------------------------------------------
# Text post-processing
# ---------------------------------------------------------------------------
def postprocess_text(raw: str) -> str:
    """
    Normalise Granite's output to a single clean line.

    - Collapses internal whitespace/newlines to single spaces.
    - Replaces very short outputs (noise artefacts like "¡oh!") with "[ ... ]".
    """
    text = " ".join(raw.split())
    if len(text.replace(" ", "")) < 4:
        return "[ ... ]"
    return text


# ---------------------------------------------------------------------------
# Audio frame chunker
# ---------------------------------------------------------------------------
def chunk_audio(buf: bytes, leftover: bytes, frame_bytes: int) -> tuple[list[bytes], bytes]:
    """
    Split *buf* (prepended with any *leftover* from a previous call) into
    exact *frame_bytes*-sized chunks.

    Returns ``(frames, new_leftover)`` where ``new_leftover`` is the partial
    frame (< frame_bytes) that should be passed back as *leftover* next call.
    """
    combined = leftover + buf
    frames: list[bytes] = []
    while len(combined) >= frame_bytes:
        frames.append(combined[:frame_bytes])
        combined = combined[frame_bytes:]
    return frames, combined


# ---------------------------------------------------------------------------
# VAD segmenter state machine
# ---------------------------------------------------------------------------
class VadSegmenter:
    """
    Accumulates 20 ms PCM frames and emits complete speech segments.

    Usage::

        segmenter = VadSegmenter()
        for frame, is_speech in zip(frames, vad_decisions):
            pcm = segmenter.push(frame, is_speech)
            if pcm is not None:
                # pcm is a bytes object containing the full segment's int16 PCM
                translate(pcm)

    Configuration is taken from module-level constants so tests can rely on
    predictable thresholds.
    """

    def __init__(
        self,
        silence_frames: int = SILENCE_FRAMES,
        min_speech_frames: int = MIN_SPEECH_FRAMES,
        max_speech_frames: int = MAX_SPEECH_FRAMES,
        min_speech_ratio: float = MIN_SPEECH_RATIO,
    ) -> None:
        self.silence_frames = silence_frames
        self.min_speech_frames = min_speech_frames
        self.max_speech_frames = max_speech_frames
        self.min_speech_ratio = min_speech_ratio

        self._speech_frames: list[bytes] = []
        self._speech_frame_count = 0
        self._silent_count = 0
        self._in_speech = False

    # ------------------------------------------------------------------
    def push(self, frame: bytes, is_speech: bool) -> bytes | None:
        """
        Feed one frame and its VAD decision.

        Returns the complete segment's PCM bytes when a segment boundary is
        detected, or ``None`` if the segment is still accumulating.
        """
        if is_speech:
            self._speech_frames.append(frame)
            self._speech_frame_count += 1
            self._silent_count = 0
            self._in_speech = True

        force_cut = self._in_speech and len(self._speech_frames) >= self.max_speech_frames
        end_of_utterance = self._in_speech and not is_speech and self._silent_count + 1 >= self.silence_frames

        if not is_speech and self._in_speech:
            self._speech_frames.append(frame)
            self._silent_count += 1

        if force_cut or end_of_utterance:
            return self._flush()

        return None

    # ------------------------------------------------------------------
    def _flush(self) -> bytes | None:
        """
        Finalise and return the current segment if it meets quality thresholds,
        otherwise discard it silently.  Resets state regardless.
        """
        total = len(self._speech_frames)
        ratio = self._speech_frame_count / total if total > 0 else 0.0

        result: bytes | None = None
        if total >= self.min_speech_frames and ratio >= self.min_speech_ratio:
            result = b"".join(self._speech_frames)

        self._speech_frames = []
        self._speech_frame_count = 0
        self._silent_count = 0
        self._in_speech = False
        return result
