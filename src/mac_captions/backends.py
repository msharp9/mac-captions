"""
Model backend abstraction.

Two implementations are provided:
  mlx          — Granite Speech 4.1 via mlx-audio (Apple Silicon only, fastest)
  transformers — Granite Speech 4.1 via HuggingFace transformers + PyTorch CPU
                 (cross-platform; works on Intel macOS and Apple Silicon)

Backend selection (in priority order):
  1. MAC_CAPTIONS_BACKEND env var: "mlx" | "transformers"
  2. Auto-detect: "mlx" on arm64, "transformers" everywhere else
"""

from __future__ import annotations

import os
import platform
import sys
from typing import Protocol

import numpy as np

from mac_captions.pipeline import postprocess_text

LANGUAGE = "es"  # ISO 639-1 code — used by the MLX backend
LANGUAGE_NAME = "Spanish"  # full name — used in the transformers chat-template prompt


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


class Backend(Protocol):
    def translate(self, pcm_int16: np.ndarray) -> str: ...


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------


def detect_backend() -> str:
    """Return the backend name to use: 'mlx' or 'transformers'.

    Priority:
      1. MAC_CAPTIONS_BACKEND env var if set to a recognised value.
      2. 'mlx' on Apple Silicon (arm64); 'transformers' on all other arches.
    """
    env = os.environ.get("MAC_CAPTIONS_BACKEND", "").strip().lower()
    if env in ("mlx", "transformers"):
        return env
    if env:
        print(
            f"[backend] unknown MAC_CAPTIONS_BACKEND={env!r}; "
            "must be 'mlx' or 'transformers'. Falling back to auto-detect.",
            file=sys.stderr,
            flush=True,
        )
    return "mlx" if platform.machine() == "arm64" else "transformers"


# ---------------------------------------------------------------------------
# Backend loader
# ---------------------------------------------------------------------------


def load_backend(model_path: str) -> Backend:
    """Instantiate and return the appropriate backend.

    Must be called on the thread that will also call translate() — this is
    required by the MLX GPU stream constraint and is harmless for the
    transformers backend.
    """
    name = detect_backend()
    print(f"[backend] using {name!r} backend", file=sys.stderr, flush=True)
    if name == "mlx":
        return MlxBackend(model_path)
    return TransformersBackend(model_path)


# ---------------------------------------------------------------------------
# MLX backend (Apple Silicon only)
# ---------------------------------------------------------------------------


class MlxBackend:  # pragma: no cover
    """Granite Speech 4.1 via mlx-audio — Apple Silicon only."""

    def __init__(self, model_path: str) -> None:
        from mlx_audio.stt.utils import load_model  # type: ignore[import]

        self._model = load_model(model_path)

    def translate(self, pcm_int16: np.ndarray) -> str:
        audio_f32 = pcm_int16.astype(np.float32) / 32768.0
        out = self._model.generate(
            audio_f32,
            language=LANGUAGE,
            temperature=0.0,
            max_tokens=100,
            verbose=False,
        )
        return postprocess_text(out.text)


# ---------------------------------------------------------------------------
# Transformers backend (CPU — cross-platform, including Intel macOS)
# ---------------------------------------------------------------------------


class TransformersBackend:  # pragma: no cover
    """Granite Speech 4.1 via HuggingFace transformers + PyTorch (CPU).

    Works on any platform that can install PyTorch, including Intel Macs where
    mlx-audio/mlx-metal is unavailable.  Inference is slower than MLX on Apple
    Silicon but correct on all supported hardware.
    """

    def __init__(self, model_path: str) -> None:
        import torch  # type: ignore[import]
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor  # type: ignore[import]

        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(model_path)
        self._tokenizer = self._processor.tokenizer
        # Use float32 — bfloat16 is unreliable / slow on CPU.
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
        )
        self._model.eval()

    def translate(self, pcm_int16: np.ndarray) -> str:
        torch = self._torch

        # Build the chat prompt following the Granite Speech model-card pattern.
        chat = [
            {
                "role": "user",
                "content": f"<|audio|>translate the speech to {LANGUAGE_NAME}.",
            }
        ]
        prompt = self._tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        # Convert int16 PCM → normalised float32 tensor shaped (1, N) at 16 kHz.
        audio_f32 = pcm_int16.astype("float32") / 32768.0
        wav = torch.from_numpy(audio_f32).unsqueeze(0)  # (1, N)

        inputs = self._processor(prompt, wav, return_tensors="pt")
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                num_beams=1,
            )

        # Strip input-prompt tokens before decoding the generated translation.
        num_input = inputs["input_ids"].shape[-1]
        new_tokens = output_ids[0, num_input:].unsqueeze(0)
        decoded = self._tokenizer.batch_decode(
            new_tokens,
            add_special_tokens=False,
            skip_special_tokens=True,
        )
        text = decoded[0] if decoded else ""
        return postprocess_text(text)
