"""Configuration module for the pixel art recolor pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Optional


BASE_DIR = Path(__file__).resolve().parent.parent

PATH_INPUT = BASE_DIR / "input"
PATH_BACKGROUNDS = BASE_DIR / "backgrounds"
PATH_OUTPUT = BASE_DIR / "variants"
PATH_INPUT_MAPS = BASE_DIR / "inputMaps"

PIXEL_RESOLUTION = 16
USE_HLS_METHOD = True
USE_REAL_BACKGROUNDS_ONLY = True


PBR_GENERATION_CONFIG: Dict[str, object] = {
    "use_unified_generation": True,
    "analysis_shared_across_maps": True,
    "generation_module": "pixel_pipeline.modules.pbr.generation",
    "total_pbr_maps": 16,
    "maps_list": [
        "metallic",
        "roughness",
        "normal",
        "height",
        "ao",
        "curvature",
        "transmission",
        "subsurface",
        "specular",
        "ior",
        "emissive",
        "structural",
        "porosity",
        "opacity",
        "fuzz",
        "material",
    ],
}


@dataclass
class PipelineConfig:
    """Runtime configuration for the recolor pipeline."""

    input_path: Path = PATH_INPUT
    backgrounds_path: Path = PATH_BACKGROUNDS
    output_path: Path = PATH_OUTPUT
    input_maps_path: Path = PATH_INPUT_MAPS
    pixel_resolution: int = PIXEL_RESOLUTION
    use_hls_method: bool = USE_HLS_METHOD
    use_real_backgrounds_only: bool = USE_REAL_BACKGROUNDS_ONLY
    rotations: Iterable[int] = (0, 90, 180, 270)
    max_variants: int = 500
    random_seed: Optional[int] = None
    threads: int = 8
    enable_gpu: bool = False
    enable_rotation: bool = True
    enable_pbr: bool = True
    enable_vcolor: bool = True
    log_file: Path = BASE_DIR / "processing.log"
    pbr_generation: Dict[str, object] = field(default_factory=lambda: dict(PBR_GENERATION_CONFIG))

    def as_dict(self) -> Dict[str, object]:
        """Return the configuration as a plain dictionary."""

        return {
            "PATH_INPUT": self.input_path,
            "PATH_BACKGROUNDS": self.backgrounds_path,
            "PATH_OUTPUT": self.output_path,
            "PATH_INPUT_MAPS": self.input_maps_path,
            "PIXEL_RESOLUTION": self.pixel_resolution,
            "USE_HLS_METHOD": self.use_hls_method,
            "USE_REAL_BACKGROUNDS_ONLY": self.use_real_backgrounds_only,
            "ROTATION_ANGLES": tuple(self.rotations),
            "MAX_VARIANTS": self.max_variants,
            "RANDOM_SEED": self.random_seed,
            "THREADS": self.threads,
            "ENABLE_GPU": self.enable_gpu,
            "ENABLE_ROTATION": self.enable_rotation,
            "ENABLE_PBR": self.enable_pbr,
            "ENABLE_VCOLOR": self.enable_vcolor,
            "LOG_FILE": self.log_file,
            "PBR_GENERATION": self.pbr_generation,
        }


def build_config(overrides: Optional[Mapping[str, object]] = None) -> Dict[str, object]:
    """Create a configuration dictionary with optional overrides."""

    config = PipelineConfig()
    if overrides:
        mutable: MutableMapping[str, object] = config.as_dict()
        for key, value in overrides.items():
            if key in mutable:
                mutable[key] = value
        return dict(mutable)
    return config.as_dict()
