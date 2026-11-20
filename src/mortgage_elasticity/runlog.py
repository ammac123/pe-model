"""Run log: everything a run prints, every warning it raises, written to run_log.txt."""

import warnings
from pathlib import Path

_lines: list[str] = []


def log(msg: str = "") -> None:
    print(msg, flush=True)
    _lines.append(str(msg))


def _showwarning(message, category=None, *args, **kwargs) -> None:
    log(f"WARNING: {message}")


def start() -> None:
    """Clear the log and route warnings.warn into it."""
    _lines.clear()
    warnings.showwarning = _showwarning


def write(path: Path) -> None:
    Path(path).write_text("\n".join(_lines) + "\n")
