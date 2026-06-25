"""
Live English → Spanish caption pipeline.

Captures microphone audio, detects speech segments via WebRTC VAD, translates
each segment using Granite Speech 4.1 via mlx-audio, and writes one caption
line per segment to stdout.

Pipe into the Swift caption overlay:
    mac-captions | ./.build/caption-overlay

Or run standalone to see raw text output:
    mac-captions
"""

from __future__ import annotations

import os
import queue
import sys

import numpy as np
import sounddevice as sd
import webrtcvad
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from mlx_audio.stt.utils import load_model

from mac_captions.pipeline import (
    FRAME_BYTES,
    FRAME_SAMPLES,
    SAMPLE_RATE,
    VAD_AGGRESSIVENESS,
    VadSegmenter,
    chunk_audio,
    postprocess_text,
)

MODEL_ID = "ibm-granite/granite-speech-4.1-2b"
LANGUAGE = "es"  # target translation language

# ---------------------------------------------------------------------------
# Model resolution
#
# Default (cache-first): start instantly from the locally cached weights on
# every run — never blocks on the network.
#
# To check Hugging Face for an updated model revision:
#   MAC_CAPTIONS_UPDATE=1 ./run-captions.sh
# ---------------------------------------------------------------------------


def _resolve_model_path() -> str:
    """Return a local path to the cached model, downloading only when needed.

    By default the cache is always used directly so startup is instant and
    works with no internet.  Set MAC_CAPTIONS_UPDATE=1 to force an online
    update check (falls back to cache within ~5 s if the network is slow).
    """
    update = os.environ.get("MAC_CAPTIONS_UPDATE") not in (None, "", "0")

    if not update:
        try:
            return snapshot_download(MODEL_ID, local_files_only=True)
        except LocalEntryNotFoundError:
            print(
                "Model not cached — downloading once (~2 GB). Needs internet…",
                file=sys.stderr,
                flush=True,
            )

    # Either an explicit update was requested, or nothing is cached yet.
    try:
        if update:
            print("Checking Hugging Face for model updates…", file=sys.stderr, flush=True)
        return snapshot_download(MODEL_ID, etag_timeout=5)
    except LocalEntryNotFoundError:
        print(
            f"ERROR: model '{MODEL_ID}' is not cached and Hugging Face is unreachable.\n"
            "Connect to the internet once to download the model (~2 GB); "
            "after that it runs fully offline.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _load() -> object:
    model_path = _resolve_model_path()
    print(f"Loading model {MODEL_ID} …", file=sys.stderr, flush=True)
    model = load_model(model_path)
    print("Model ready. Listening…", file=sys.stderr, flush=True)
    return model


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------
def _translate(model, pcm_int16: np.ndarray) -> str:  # noqa: ANN001
    audio_f32 = pcm_int16.astype(np.float32) / 32768.0
    out = model.generate(audio_f32, language=LANGUAGE, temperature=0.0, max_tokens=100, verbose=False)
    return postprocess_text(out.text)


# ---------------------------------------------------------------------------
# VAD segmenter loop (must run on same thread as model load — MLX GPU stream)
# ---------------------------------------------------------------------------
def _segmenter_loop(model, frame_queue: queue.Queue) -> None:  # noqa: ANN001
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    segmenter = VadSegmenter()
    segment_n = 0

    print("[segmenter] started", file=sys.stderr, flush=True)

    while True:
        frame: bytes = frame_queue.get()

        try:
            is_speech = vad.is_speech(frame, SAMPLE_RATE)
        except Exception as exc:  # noqa: BLE001
            print(f"[vad error] {exc}", file=sys.stderr, flush=True)
            continue

        pcm = segmenter.push(frame, is_speech)
        if pcm is None:
            continue

        segment_n += 1
        dur_ms = (len(pcm) // FRAME_BYTES) * 20
        print(f"[segment {segment_n}] {dur_ms}ms — translating…", file=sys.stderr, flush=True)

        try:
            audio = np.frombuffer(pcm, dtype=np.int16)
            text = _translate(model, audio)
            if text:
                try:
                    print(text, flush=True)
                    print(f"[segment {segment_n}] → {text!r}", file=sys.stderr, flush=True)
                except BrokenPipeError:
                    print("[pipe broken — overlay exited]", file=sys.stderr, flush=True)
                    sys.exit(0)
        except Exception as exc:  # noqa: BLE001
            print(f"[translate error] {exc}", file=sys.stderr, flush=True)

        print("[segmenter] listening…", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run() -> None:
    frame_queue: queue.Queue = queue.Queue()
    state = {"leftover": b""}

    def audio_callback(indata, frames: int, time, status) -> None:  # noqa: ANN001
        if status:
            print(f"[audio] {status}", file=sys.stderr, flush=True)
        new_frames, state["leftover"] = chunk_audio(bytes(indata), state["leftover"], FRAME_BYTES)
        for f in new_frames:
            frame_queue.put(f)

    print("Press Ctrl+C to stop.", file=sys.stderr, flush=True)
    try:
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SAMPLES * 4,
            dtype="int16",
            channels=1,
            callback=audio_callback,
        ):
            # Model loading and generate() must share the same thread (MLX GPU stream)
            model = _load()
            _segmenter_loop(model, frame_queue)
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr, flush=True)


if __name__ == "__main__":
    run()
