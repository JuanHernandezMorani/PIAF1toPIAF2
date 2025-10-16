"""Command line interface for the pixel art recolor pipeline."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict

from .core import config

def _configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(log_path, encoding="utf-8", delay=True)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pixel-art recolor and PBR map generator")
    parser.add_argument("--input", type=Path, default=config.PATH_INPUT, help="Input directory with sprites")
    parser.add_argument("--background", type=Path, default=config.PATH_BACKGROUNDS, help="Directory with background images")
    parser.add_argument("--output", type=Path, default=config.PATH_OUTPUT, help="Directory to write generated variants")
    parser.add_argument("--threads", type=int, default=8, help="Number of worker threads")
    parser.add_argument("--max-variants", type=int, default=500, help="Maximum number of variants per sprite")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration if available")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    return parser.parse_args()


def build_runtime_config(args: argparse.Namespace) -> Dict[str, object]:
    overrides: Dict[str, object] = {
        "PATH_INPUT": args.input.resolve(),
        "PATH_BACKGROUNDS": args.background.resolve(),
        "PATH_OUTPUT": args.output.resolve(),
        "THREADS": args.threads,
        "MAX_VARIANTS": args.max_variants,
        "RANDOM_SEED": args.seed,
        "ENABLE_GPU": args.gpu,
    }
    return config.build_config(overrides)


def main() -> None:
    args = parse_args()
    cfg = build_runtime_config(args)
    _configure_logging(Path(cfg["LOG_FILE"]))
    logging.getLogger("pixel_pipeline").info("Starting recolor pipeline")
    try:
        from .modules.recolor_generator import RecolorPipeline
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise SystemExit(
            "Pixel pipeline requires NumPy and Pillow. Install them before running the pipeline."
        ) from exc
    pipeline = RecolorPipeline(cfg)
    pipeline.run(threads=args.threads)


if __name__ == "__main__":
    main()
