# mac-captions

Live English speech → Spanish captions displayed at the bottom of your screen on macOS, powered by [IBM Granite Speech 4.1-2b](https://huggingface.co/ibm-granite/granite-speech-4.1-2b) running fully on-device.

Two inference backends are selected automatically by hardware:

- **Apple Silicon** → [MLX](https://github.com/ml-explore/mlx) (GPU-accelerated, fastest)
- **Intel Macs** → [transformers](https://github.com/huggingface/transformers) + PyTorch on CPU (slower, but works)

Override with `MAC_CAPTIONS_BACKEND=mlx|transformers`.

```
mic → Python VAD → Granite (translate) → stdout | Swift overlay → screen
```

The caption bar floats above every window — including full-screen apps — and passes mouse clicks through so it never interrupts your work.

## Requirements

- macOS (Apple Silicon **or** Intel)
- A microphone
- [Pixi](https://pixi.sh/) for dependency management (provides PyTorch for Intel Macs via conda-forge, which PyPI no longer ships)
- Xcode Command Line Tools (`xcode-select --install`) for the Swift overlay

## Quick start

```bash
# Install dependencies (auto-resolves the right backend for your hardware)
pixi install

# Build the Swift overlay and launch the pipeline
./run-captions.sh
```

The first run downloads the Granite model weights (~2 GB) and caches them locally. After that, the app starts instantly and works fully offline — it never phones home on startup. To check for a model update: `MAC_CAPTIONS_UPDATE=1 ./run-captions.sh`.

Grant the microphone permission when macOS prompts you (one-time).

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  Python (mac_captions.live)                           │
│                                                        │
│  sounddevice mic  →  webrtcvad  →  Granite Speech 4.1 │
│  (16 kHz, mono)      (chunking)    (MLX or PyTorch)   │
│                                         │              │
│                                   stdout (one line     │
│                                   per segment)         │
└─────────────────────────────────────────┼─────────────┘
                                          │  Unix pipe
┌─────────────────────────────────────────▼─────────────┐
│  Swift overlay (Sources/CaptionOverlay/main.swift)     │
│                                                        │
│  stdin reader  →  3-line rolling history  →  NSWindow  │
│  (readLine())      (AppDelegate)             (screen)  │
└────────────────────────────────────────────────────────┘
```

Speech is cut into segments by [WebRTC VAD](https://github.com/dpirch/libfvad):
- Segment ends after ~500 ms of silence, **or** after 2 s of continuous speech (whichever comes first)
- Segments below a 45% speech-frame ratio are discarded (filters knocks, beeps, and breathing)

## Configuration

Edit constants at the top of `src/mac_captions/live.py`:

| Constant | Default | Description |
|---|---|---|
| `LANGUAGE` | `"es"` | Target language (any Granite-supported lang code) |
| `MODEL_ID` | `ibm-granite/granite-speech-4.1-2b` | HuggingFace model ID |

Edit `src/mac_captions/pipeline.py` for VAD tuning:

| Constant | Default | Description |
|---|---|---|
| `VAD_AGGRESSIVENESS` | `3` | 0 (permissive) – 3 (strict) |
| `SILENCE_FRAMES` | `25` | Silent frames before cutting (~500 ms) |
| `MAX_SPEECH_FRAMES` | `100` | Max segment length before forced cut (~2 s) |
| `MIN_SPEECH_RATIO` | `0.45` | Min fraction of frames classified as speech |

## Supported translation languages

Granite Speech 4.1 supports translation between English and: French (`fr`), German (`de`), Spanish (`es`), Portuguese (`pt`), Italian (`it`), Japanese (`ja`), Mandarin (`zh`).

## Known limitations

- **Latency**: Each segment is translated after it ends. Expect 1–3 s delay depending on segment length and hardware. The Intel/CPU (transformers) backend is noticeably slower than MLX on Apple Silicon.
- **macOS only**: The Swift overlay is Apple-platform specific.
- **Mic permission**: The first run prompts for microphone access via macOS TCC. Grant it for the terminal app running the pipeline.
- **Xcode for all-Swift**: The Swift overlay is built with plain `swiftc` (no Xcode needed). A future version will embed the model in-process using [mlx-audio-swift](https://github.com/Blaizzy/mlx-audio-swift), which requires Xcode's Metal toolchain.

## Setup

This project uses [Pixi](https://pixi.sh/) for dependency management. Pixi pulls
PyTorch from conda-forge, which still ships builds for Intel Macs (osx-64) — PyPI
no longer does, capping Intel-Mac PyTorch at an unusably old version.

1. **Install Pixi**:
    ```bash
    curl -fsSL https://pixi.sh/install.sh | sh
    ```

2. **Install dependencies**:
    ```bash
    pixi install
    ```

## Contributing

### Running tests
```bash
pixi run test
```

### Linting and formatting
```bash
pixi run ruff check .
pixi run ruff format .
```
