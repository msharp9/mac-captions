#!/usr/bin/env bash
# Build the Swift overlay (if needed), then start the live caption pipeline.
#
# Usage:
#   ./run-captions.sh
#
# Requirements:
#   - uv venv set up:  uv sync
#   - macOS (Apple Silicon or Intel) with a microphone
#
# Backend is chosen automatically:
#   - Apple Silicon (arm64): mlx-audio  (GPU-accelerated)
#   - Intel (x86_64):        transformers + PyTorch CPU  (slower, but works)
# Override:  MAC_CAPTIONS_BACKEND=mlx|transformers ./run-captions.sh
#
# Press Ctrl+C to stop.
set -euo pipefail

echo "Building overlay…"
mkdir -p .build
swiftc Sources/CaptionOverlay/main.swift \
    -o .build/caption-overlay \
    -framework AppKit

echo "Starting pipeline — speak English, see Spanish captions. Ctrl+C to stop."
PYTHONPATH=src uv run python -m mac_captions.live | .build/caption-overlay
