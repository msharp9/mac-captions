#!/usr/bin/env bash
# Build and launch the caption overlay.
# Usage: ./run.sh
set -euo pipefail

mkdir -p .build
swiftc Sources/CaptionOverlay/main.swift \
    -o .build/caption-overlay \
    -framework AppKit

echo "Starting caption overlay — press Ctrl+C to quit."
.build/caption-overlay
