"""
Model backend abstraction.

Three implementations are provided:
  mlx          — Granite Speech 4.1 via mlx-audio (Apple Silicon only, fastest)
  transformers — Granite Speech 4.1 via HuggingFace transformers + PyTorch CPU
                 (cross-platform; works on Intel macOS and Apple Silicon)
  llamacpp     — Granite Speech 4.1 GGUF via llama-server subprocess
                 (Intel macOS default; faster than transformers on CPU)

Backend selection (in priority order):
  1. MAC_CAPTIONS_BACKEND env var: "mlx" | "transformers" | "llamacpp"
  2. Auto-detect: "mlx" on arm64, "llamacpp" everywhere else
"""

from __future__ import annotations

import atexit
import base64
import io
import json
import os
import platform
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import wave
from subprocess import DEVNULL
from typing import Protocol

import numpy as np

from mac_captions.pipeline import SAMPLE_RATE, postprocess_text

# Target caption languages supported by Granite Speech 4.1.
# Maps ISO 639-1 code → full English name used in chat-template / llama-server prompts.
SUPPORTED_LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "de": "German",
    "fr": "French",
    "it": "Italian",
    "pt": "Portuguese",
    "ja": "Japanese",
    "zh": "Chinese",
}


def _resolve_language() -> tuple[str, str]:
    """Return (iso_code, full_name) for the target caption language.

    Controlled by MAC_CAPTIONS_LANG (ISO 639-1 code); defaults to Spanish.
    Falls back to Spanish with a warning if the code is unrecognised.
    """
    code = os.environ.get("MAC_CAPTIONS_LANG", "es").strip().lower()
    if code in SUPPORTED_LANGUAGES:
        return code, SUPPORTED_LANGUAGES[code]
    print(
        f"[lang] unknown MAC_CAPTIONS_LANG={code!r}; "
        f"must be one of {', '.join(sorted(SUPPORTED_LANGUAGES))}. Falling back to 'es'.",
        file=sys.stderr,
        flush=True,
    )
    return "es", SUPPORTED_LANGUAGES["es"]


LANGUAGE, LANGUAGE_NAME = _resolve_language()  # ISO code + full name for the target language

# GGUF repo constants — used by LlamaCppBackend
GGUF_MODEL_ID = "ibm-granite/granite-speech-4.1-2b-GGUF"
GGUF_QUANT = os.environ.get("MAC_CAPTIONS_GGUF_QUANT", "Q5_K_M")
MMPROJ_FILE = "mmproj-model-f16.gguf"


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


class Backend(Protocol):
    def translate(self, pcm_int16: np.ndarray) -> str: ...


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------


def detect_backend() -> str:
    """Return the backend name to use: 'mlx', 'transformers', or 'llamacpp'.

    Priority:
      1. MAC_CAPTIONS_BACKEND env var if set to a recognised value.
      2. 'mlx' on Apple Silicon (arm64); 'llamacpp' on all other arches.
    """
    env = os.environ.get("MAC_CAPTIONS_BACKEND", "").strip().lower()
    if env in ("mlx", "transformers", "llamacpp"):
        return env
    if env:
        print(
            f"[backend] unknown MAC_CAPTIONS_BACKEND={env!r}; "
            "must be 'mlx', 'transformers', or 'llamacpp'. Falling back to auto-detect.",
            file=sys.stderr,
            flush=True,
        )
    return "mlx" if platform.machine() == "arm64" else "llamacpp"


# ---------------------------------------------------------------------------
# Backend loader
# ---------------------------------------------------------------------------


def load_backend(model_path: str) -> Backend:
    """Instantiate and return the appropriate backend.

    Must be called on the thread that will also call translate() — this is
    required by the MLX GPU stream constraint and is harmless for other backends.

    For the llamacpp backend, model_path is unused (the backend self-resolves
    its GGUF files from the GGUF HuggingFace repo).
    """
    name = detect_backend()
    print(f"[backend] using {name!r} backend", file=sys.stderr, flush=True)
    if name == "mlx":
        return MlxBackend(model_path)
    if name == "llamacpp":
        return LlamaCppBackend()
    return TransformersBackend(model_path)


# ---------------------------------------------------------------------------
# GGUF file resolution (used by LlamaCppBackend)
# ---------------------------------------------------------------------------


def _resolve_gguf_files() -> tuple[str, str]:  # pragma: no cover
    """Return (gguf_path, mmproj_path) for the chosen quantization.

    Mirrors the cache-first / MAC_CAPTIONS_UPDATE / LocalEntryNotFoundError
    pattern from live._resolve_model_path(), but fetches only the chosen quant
    and the mmproj file — not the full GGUF repo.

    Returns absolute paths to the .gguf and the mmproj inside the snapshot dir.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    gguf_filename = f"granite-speech-4.1-2b-{GGUF_QUANT}.gguf"
    patterns = [gguf_filename, MMPROJ_FILE]
    update = os.environ.get("MAC_CAPTIONS_UPDATE") not in (None, "", "0")

    if not update:
        try:
            snap_dir = snapshot_download(GGUF_MODEL_ID, allow_patterns=patterns, local_files_only=True)
        except LocalEntryNotFoundError:
            print(
                "GGUF model not cached — downloading once (~1.5 GB). Needs internet…",
                file=sys.stderr,
                flush=True,
            )
            snap_dir = snapshot_download(GGUF_MODEL_ID, allow_patterns=patterns)
    else:
        try:
            print("Checking Hugging Face for GGUF model updates…", file=sys.stderr, flush=True)
            snap_dir = snapshot_download(GGUF_MODEL_ID, allow_patterns=patterns, etag_timeout=5)
        except LocalEntryNotFoundError:
            print(
                f"ERROR: GGUF model '{GGUF_MODEL_ID}' is not cached and Hugging Face is unreachable.\n"
                "Connect to the internet once to download the model; after that it runs fully offline.",
                file=sys.stderr,
                flush=True,
            )
            raise SystemExit(1)

    gguf_path = os.path.join(snap_dir, gguf_filename)
    mmproj_path = os.path.join(snap_dir, MMPROJ_FILE)

    for path, label in ((gguf_path, "GGUF model"), (mmproj_path, "mmproj")):
        if not os.path.isfile(path):
            print(
                f"ERROR: {label} file not found at {path!r}.\nTry: MAC_CAPTIONS_UPDATE=1 ./run-captions.sh",
                file=sys.stderr,
                flush=True,
            )
            raise SystemExit(1)

    return gguf_path, mmproj_path


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
        # Pass an explicit full-name prompt rather than `language=LANGUAGE`.
        # mlx-audio's built-in LANGUAGE_CODES map omits some languages (e.g. zh, it),
        # so a bare code falls through as "Translate the speech to zh." — which the
        # model ignores and transcribes in English. The prompt path works for all.
        out = self._model.generate(
            audio_f32,
            prompt=f"Translate the speech to {LANGUAGE_NAME}.",
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
    Silicon but correct on all supported hardware.  On Intel macOS, prefer the
    LlamaCppBackend (default) for better CPU performance.
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


# ---------------------------------------------------------------------------
# llama.cpp backend (Intel macOS default; any platform with llama-server)
# ---------------------------------------------------------------------------


def _free_port() -> int:  # pragma: no cover
    """Bind an ephemeral socket to get a free port number, then release it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LlamaCppBackend:  # pragma: no cover
    """Granite Speech 4.1 GGUF via llama-server (faster CPU inference on Intel).

    Launches llama-server as a long-lived subprocess, loaded once with the
    quantized GGUF model and the audio mmproj. Each translate() call encodes the
    PCM segment as a WAV and POSTs it to /v1/chat/completions with the Spanish
    translation prompt.

    Environment:
      MAC_CAPTIONS_LLAMA_BIN   — path to llama-server binary (default: llama-server)
      MAC_CAPTIONS_LLAMA_PORT  — port override (default: auto-picked ephemeral port)
      MAC_CAPTIONS_GGUF_QUANT  — quantization to use (default: Q5_K_M)
    """

    def __init__(self) -> None:
        gguf_path, mmproj_path = _resolve_gguf_files()

        port_env = os.environ.get("MAC_CAPTIONS_LLAMA_PORT", "").strip()
        port = int(port_env) if port_env else _free_port()
        self._base = f"http://127.0.0.1:{port}"

        binary = os.environ.get("MAC_CAPTIONS_LLAMA_BIN", "llama-server").strip()

        print(
            f"[llamacpp] starting llama-server on port {port} "
            f"(quant={GGUF_QUANT}, model={os.path.basename(gguf_path)})…",
            file=sys.stderr,
            flush=True,
        )

        self._proc = subprocess.Popen(
            [
                binary,
                "--model",
                gguf_path,
                "--mmproj",
                mmproj_path,
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "-c",
                "4096",
            ],
            stdout=DEVNULL,
            stderr=sys.stderr,
            start_new_session=True,
        )
        atexit.register(self.close)
        self._closed = False

        self._wait_for_health()

    def _wait_for_health(self, timeout: float = 120.0) -> None:
        """Poll GET /health until llama-server reports ready, or timeout."""
        deadline = time.monotonic() + timeout
        health_url = f"{self._base}/health"
        last_log = time.monotonic()

        while time.monotonic() < deadline:
            # Detect early process exit
            if self._proc.poll() is not None:
                print(
                    f"[llamacpp] llama-server exited early (code {self._proc.returncode}). "
                    "Check stderr above for details.",
                    file=sys.stderr,
                    flush=True,
                )
                raise SystemExit(1)

            try:
                with urllib.request.urlopen(health_url, timeout=2) as resp:
                    body = json.loads(resp.read())
                    if body.get("status") in ("ok", "loading model"):
                        if body.get("status") == "ok":
                            print("[llamacpp] server ready.", file=sys.stderr, flush=True)
                            return
            except (urllib.error.URLError, OSError, json.JSONDecodeError):
                pass

            now = time.monotonic()
            if now - last_log >= 10.0:
                elapsed = int(now - (deadline - timeout))
                print(f"[llamacpp] waiting for server… ({elapsed}s)", file=sys.stderr, flush=True)
                last_log = now

            time.sleep(0.5)

        self.close()
        print(
            f"[llamacpp] server did not become ready within {timeout:.0f}s.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1)

    def translate(self, pcm_int16: np.ndarray) -> str:
        # Encode int16 mono 16 kHz PCM as an in-memory WAV.
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit = 2 bytes
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm_int16.tobytes())
        b64 = base64.b64encode(buf.getvalue()).decode()

        payload = json.dumps(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {"data": b64, "format": "wav"},
                            },
                            {
                                "type": "text",
                                "text": f"translate the speech to {LANGUAGE_NAME}.",
                            },
                        ],
                    }
                ],
                "temperature": 0,
                "max_tokens": 100,
                "stream": False,
            }
        ).encode()

        req = urllib.request.Request(
            f"{self._base}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
            text = result["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            print(f"[llamacpp] translate error: {exc}", file=sys.stderr, flush=True)
            return ""

        return postprocess_text(text)

    def close(self) -> None:
        """Terminate the llama-server subprocess (idempotent)."""
        if self._closed:
            return
        self._closed = True
        proc = self._proc
        if proc.poll() is not None:
            return  # already exited
        print("[llamacpp] shutting down server…", file=sys.stderr, flush=True)
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), 9)
            except OSError:
                proc.kill()
            proc.wait()

    def __del__(self) -> None:
        self.close()
