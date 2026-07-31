"""Pure helpers for Tkinter/subprocess training state transitions."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def status_progress(status: dict[str, Any]) -> tuple[float, str]:
    epoch = int(status.get("epoch", -1)) + 1
    epochs = max(1, int(status.get("epochs", 1)))
    percent = max(0.0, min(100.0, 100.0 * epoch / epochs))
    text = (
        f'{status.get("state", "unknown")} | '
        f"epoch {epoch}/{epochs} | "
        f'device {status.get("device", "?")} | '
        f'best NME {status.get("best_validation_nme", "?")}'
    )
    return percent, text


def request_graceful_stop(run_directory: str | Path) -> Path:
    run_directory = Path(run_directory).expanduser().resolve()
    if not run_directory.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_directory}")
    stop_path = run_directory / "STOP"
    stop_path.touch()
    return stop_path


def latest_best_checkpoint(run_directory: str | Path) -> Path | None:
    candidate = Path(run_directory).expanduser().resolve() / "best.pt"
    return candidate if candidate.is_file() else None
