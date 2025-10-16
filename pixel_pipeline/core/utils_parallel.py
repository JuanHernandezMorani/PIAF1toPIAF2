"""Parallel execution helpers for the recolor pipeline."""
from __future__ import annotations

import concurrent.futures
import logging
from contextlib import contextmanager
from typing import Callable, Iterator, Optional, Sequence, TypeVar


LOGGER = logging.getLogger("pixel_pipeline.parallel")

T = TypeVar("T")
R = TypeVar("R")


def create_thread_pool(max_workers: Optional[int] = None) -> concurrent.futures.ThreadPoolExecutor:
    """Create a thread pool executor with sane defaults."""

    return concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)


def run_parallel(function: Callable[[T], R], items: Sequence[T], *, max_workers: Optional[int] = None) -> list[R]:
    """Run *function* for each element in *items* concurrently."""

    if not items:
        return []
    with create_thread_pool(max_workers=max_workers) as executor:
        futures = [executor.submit(function, item) for item in items]
        results: list[R] = []
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:  # pragma: no cover - logging path
                LOGGER.exception("Parallel worker failure: %s", exc)
        return results


@contextmanager
def limited_threads(max_workers: Optional[int]) -> Iterator[None]:
    """Context manager that logs thread usage for diagnostics."""

    LOGGER.debug("Starting thread pool with up to %s workers", max_workers)
    try:
        yield
    finally:
        LOGGER.debug("Thread pool with %s workers completed", max_workers)
