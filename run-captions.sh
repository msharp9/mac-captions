#!/usr/bin/env bash
# Build the Swift overlay (if needed), then start the live caption pipeline.
#
# Usage:
#   ./run-captions.sh [LANGUAGE]
#
# LANGUAGE is an ISO 639-1 code for the target caption language (default: es).
# Supported: en, es, de, fr, it, pt, ja, zh
#   ./run-captions.sh          # Spanish captions (default)
#   ./run-captions.sh fr       # French captions
#   ./run-captions.sh ja       # Japanese captions
# You can also set MAC_CAPTIONS_LANG directly instead of passing an argument.
#
# Requirements:
#   - Pixi environment set up:  pixi install
#   - macOS (Apple Silicon or Intel) with a microphone
#
# Backend is chosen automatically:
#   - Apple Silicon (arm64): mlx-audio  (GPU-accelerated)
#   - Intel (x86_64):        llama.cpp GGUF  (faster CPU inference)
# Override:  MAC_CAPTIONS_BACKEND=mlx|llamacpp|transformers ./run-captions.sh
#
# Press Ctrl+C to stop.
set -euo pipefail

# Target caption language: first CLI arg wins, else MAC_CAPTIONS_LANG, else Spanish.
LANG_CODE="${1:-${MAC_CAPTIONS_LANG:-es}}"
export MAC_CAPTIONS_LANG="$LANG_CODE"

# Map ISO 639-1 code → full name (case works on macOS's bash 3.2; no assoc arrays).
case "$LANG_CODE" in
    en) LANG_NAME=English ;;
    es) LANG_NAME=Spanish ;;
    de) LANG_NAME=German ;;
    fr) LANG_NAME=French ;;
    it) LANG_NAME=Italian ;;
    pt) LANG_NAME=Portuguese ;;
    ja) LANG_NAME=Japanese ;;
    zh) LANG_NAME=Chinese ;;
    *)
        echo "Unknown language '$LANG_CODE'. Supported: en es de fr it pt ja zh" >&2
        exit 1
        ;;
esac

echo "Building overlay…"
mkdir -p .build
swiftc Sources/CaptionOverlay/main.swift \
    -o .build/caption-overlay \
    -framework AppKit

echo "Starting pipeline — speak English, see $LANG_NAME captions. Ctrl+C to stop."
pixi run captions | .build/caption-overlay
