"""Keep the Android package fast and ARM64-only."""

from __future__ import annotations

import tomllib
from pathlib import Path

FLET_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = FLET_ROOT / "pyproject.toml"
README = FLET_ROOT / "README.md"


def _android_config() -> dict[str, object]:
    with PYPROJECT.open("rb") as config_file:
        config = tomllib.load(config_file)
    return config["tool"]["flet"]["android"]


def _target_architectures() -> list[str]:
    architectures = _android_config()["target_arch"]
    assert isinstance(architectures, list)
    assert all(isinstance(architecture, str) for architecture in architectures)
    return architectures


def test_android_build_targets_only_arm64() -> None:
    assert _target_architectures() == ["arm64-v8a"]


def test_android_build_does_not_target_emulator_x86_64() -> None:
    assert "x86_64" not in _target_architectures()


def test_android_build_keeps_a_single_architecture() -> None:
    assert len(_target_architectures()) == 1


def test_documented_build_command_is_arm64_only() -> None:
    readme = README.read_text(encoding="utf-8")
    assert "flet build apk --arch arm64-v8a" in readme
    assert "--arch arm64-v8a x86_64" not in readme
