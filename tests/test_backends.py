"""
Unit tests for mac_captions.backends.detect_backend().

detect_backend() is pure (no hardware, no heavy imports) and therefore
safe to run in CI on any platform.

Auto-detect results by platform:
  arm64   → 'mlx'
  x86_64  → 'llamacpp'  (Intel macOS default; was 'transformers' before GGUF support)
  other   → 'llamacpp'
"""

from __future__ import annotations

import importlib
from unittest.mock import patch


def _reload_backend_module():
    """Re-import backends so that module-level state is fresh."""
    import mac_captions.backends as mod

    importlib.reload(mod)
    return mod


def test_env_var_mlx(monkeypatch):
    monkeypatch.setenv("MAC_CAPTIONS_BACKEND", "mlx")
    from mac_captions.backends import detect_backend

    assert detect_backend() == "mlx"


def test_env_var_transformers(monkeypatch):
    monkeypatch.setenv("MAC_CAPTIONS_BACKEND", "transformers")
    from mac_captions.backends import detect_backend

    assert detect_backend() == "transformers"


def test_env_var_llamacpp(monkeypatch):
    monkeypatch.setenv("MAC_CAPTIONS_BACKEND", "llamacpp")
    from mac_captions.backends import detect_backend

    assert detect_backend() == "llamacpp"


def test_env_var_case_insensitive(monkeypatch):
    monkeypatch.setenv("MAC_CAPTIONS_BACKEND", "MLX")
    from mac_captions.backends import detect_backend

    assert detect_backend() == "mlx"


def test_env_var_unknown_falls_through_to_autodetect(monkeypatch, capsys):
    """An unrecognised env value should warn and fall back to platform auto-detect."""
    monkeypatch.setenv("MAC_CAPTIONS_BACKEND", "vllm")
    with patch("platform.machine", return_value="arm64"):
        from mac_captions.backends import detect_backend

        result = detect_backend()
    assert result == "mlx"
    captured = capsys.readouterr()
    assert "vllm" in captured.err


def test_autodetect_arm64(monkeypatch):
    monkeypatch.delenv("MAC_CAPTIONS_BACKEND", raising=False)
    with patch("platform.machine", return_value="arm64"):
        from mac_captions.backends import detect_backend

        assert detect_backend() == "mlx"


def test_autodetect_x86_64(monkeypatch):
    monkeypatch.delenv("MAC_CAPTIONS_BACKEND", raising=False)
    with patch("platform.machine", return_value="x86_64"):
        from mac_captions.backends import detect_backend

        assert detect_backend() == "llamacpp"


def test_autodetect_unknown_arch(monkeypatch):
    """Any non-arm64 machine falls back to the llamacpp backend."""
    monkeypatch.delenv("MAC_CAPTIONS_BACKEND", raising=False)
    with patch("platform.machine", return_value="riscv64"):
        from mac_captions.backends import detect_backend

        assert detect_backend() == "llamacpp"
