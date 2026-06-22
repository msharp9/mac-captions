#!/usr/bin/env bash
# Build the Swift overlay (if needed), then start the live caption pipeline.
#
# Usage:
#   ./run-captions.sh
#
# Requirements:
#   - uv venv set up:  uv sync
#   - macOS with Apple Silicon (M-series) and a microphone
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
