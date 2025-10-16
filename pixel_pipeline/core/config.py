"""Configuration module for the pixel art recolor pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Optional


BASE_DIR = Path(__file__).resolve().parent.parent

PATH_INPUT = BASE_DIR / "input"
PATH_BACKGROUNDS = BASE_DIR / "backgrounds"
PATH_OUTPUT = BASE_DIR / "variants"

PIXEL_RESOLUTION = 16
USE_HLS_METHOD = True
USE_REAL_BACKGROUNDS_ONLY = True


MAP_TYPES: Dict[str, Dict[str, str]] = {
    "surface": {
        "normal": "pixel_pipeline.modules.surface_maps.normal_map",
        "specular": "pixel_pipeline.modules.surface_maps.specular_map",
        "roughness": "pixel_pipeline.modules.surface_maps.roughness_map",
        "metallic": "pixel_pipeline.modules.surface_maps.metallic_map",
    },
    "geometry": {
        "height": "pixel_pipeline.modules.geometry_maps.height_map",
        "ao": "pixel_pipeline.modules.geometry_maps.ambient_occlusion",
        "curvature": "pixel_pipeline.modules.geometry_maps.curvature_map",
    },
    "pixelart": {
        "porosity": "pixel_pipeline.modules.pixelart_maps.porosity_map",
        "fuzz": "pixel_pipeline.modules.pixelart_maps.fuzz_map",
    },
    "illumination": {
        "transmission": "pixel_pipeline.modules.illumination_maps.transmission_map",
        "subsurface": "pixel_pipeline.modules.illumination_maps.subsurface_map",
    },
    "semantic": {
        "material": "pixel_pipeline.modules.semantic_maps.material_type",
        "structural": "pixel_pipeline.modules.semantic_maps.structural_map",
    },
    "optical": {
        "ior": "pixel_pipeline.modules.optical_maps.ior_map",
        "opacity": "pixel_pipeline.modules.optical_maps.opacity_map",
    },
}


@dataclass
class PipelineConfig:
    """Runtime configuration for the recolor pipeline."""

    input_path: Path = PATH_INPUT
    backgrounds_path: Path = PATH_BACKGROUNDS
    output_path: Path = PATH_OUTPUT
    pixel_resolution: int = PIXEL_RESOLUTION
    use_hls_method: bool = USE_HLS_METHOD
    use_real_backgrounds_only: bool = USE_REAL_BACKGROUNDS_ONLY
    rotations: Iterable[int] = (0, 90, 180, 270)
    max_variants: int = 500
    random_seed: Optional[int] = None
    threads: int = 8
    enable_gpu: bool = False
    log_file: Path = BASE_DIR / "processing.log"
    map_types: Dict[str, Dict[str, str]] = field(default_factory=lambda: {k: dict(v) for k, v in MAP_TYPES.items()})

    def as_dict(self) -> Dict[str, object]:
        """Return the configuration as a plain dictionary."""

        return {
            "PATH_INPUT": self.input_path,
            "PATH_BACKGROUNDS": self.backgrounds_path,
            "PATH_OUTPUT": self.output_path,
            "PIXEL_RESOLUTION": self.pixel_resolution,
            "USE_HLS_METHOD": self.use_hls_method,
            "USE_REAL_BACKGROUNDS_ONLY": self.use_real_backgrounds_only,
            "ROTATION_ANGLES": tuple(self.rotations),
            "MAX_VARIANTS": self.max_variants,
            "RANDOM_SEED": self.random_seed,
            "THREADS": self.threads,
            "ENABLE_GPU": self.enable_gpu,
            "LOG_FILE": self.log_file,
            "MAP_TYPES": self.map_types,
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
