"""I/O helpers for the pixel recolor pipeline."""
from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from PIL import Image


_LOCK_REGISTRY: dict[Path, threading.Lock] = {}
_LOCK_REGISTRY_GUARD = threading.Lock()


def ensure_dir(path: Path) -> Path:
    """Ensure that *path* exists and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def _normalize_onedrive_path(path: Path) -> Path:

    return path


def _acquire_file_lock(target: Path) -> threading.Lock:
    """Return a re-entrant lock for *target* ensuring cross-thread safety."""

    with _LOCK_REGISTRY_GUARD:
        lock = _LOCK_REGISTRY.get(target)
        if lock is None:
            lock = threading.Lock()
            _LOCK_REGISTRY[target] = lock
    lock.acquire()
    return lock


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """Context manager providing a lightweight file lock."""

    normalized = _normalize_onedrive_path(path)
    temp_lock = normalized.with_suffix(normalized.suffix + ".lock")
    lock = _acquire_file_lock(temp_lock)
    try:
        while True:
            try:
                fd = os.open(temp_lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                break
            except FileExistsError:
                time.sleep(0.05)
        yield
    finally:
        try:
            os.remove(temp_lock)
        except FileNotFoundError:
            pass
        lock.release()


class SafeFileManager:
    """Manage atomic file writes with automatic directory handling."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        ensure_dir(self.base_dir)

    def resolve(self, path: Path | str) -> Path:
        """Resolve *path* relative to :attr:`base_dir`."""

        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.base_dir / candidate
        candidate = _normalize_onedrive_path(candidate)
        ensure_dir(candidate.parent)
        return candidate

    def atomic_save(self, image: Image.Image, path: Path | str, *, format: Optional[str] = None) -> Path:
        """Safely save *image* to *path* using a temporary file."""

        destination = self.resolve(path)
        temp_dir = destination.parent / ".tmp_variants"
        ensure_dir(temp_dir)
        temp_path = temp_dir / f"{destination.name}.tmp"
        with file_lock(destination):
            image.save(temp_path, format=format or "PNG")
            os.replace(temp_path, destination)
        return destination


def atomic_save(image: Image.Image, path: Path | str, base_dir: Optional[Path] = None) -> Path:
    """Convenience wrapper to persist *image* atomically."""

    manager = SafeFileManager(base_dir or Path(path).resolve().parent)
    return manager.atomic_save(image, path)
