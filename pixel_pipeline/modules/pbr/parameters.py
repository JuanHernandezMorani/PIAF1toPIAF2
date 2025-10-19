"""Critical parameters for the physically accurate PBR pipeline."""
from __future__ import annotations

CRITICAL_PARAMETERS = {
    "MIN_ENTROPY": 0.03,
    "MAX_UNIFORM_RATIO": 0.95,
    "FLAT_DETECTION_SENSITIVITY": 1.2,
    # Metallic controls
    "METALLIC_ORGANIC_TOLERANCE": 0.05,
    "METALLIC_BINARY_STRICT": True,
    "METALLIC_SPECULAR_MIN": 0.6,
    # IOR configuration
    "IOR_OPAQUE_DEFAULT": 1.0,
    "IOR_TRANSLUCENT_RANGES": {
        "water": 1.33,
        "ice": 1.31,
        "glass": 1.5,
        "crystal": 1.6,
        "diamond": 2.42,
    },
    # Alpha behaviour
    "ALPHA_BACKGROUND_MAX": 0.01,
    "ALPHA_OBJECT_MIN": 0.98,
    "ALPHA_EDGE_FALLOFF": 2,
    # Transmission limits
    "TRANSMISSION_OPAQUE_MAX": 0.01,
    "TRANSMISSION_METAL_MAX": 0.0,
}

__all__ = ["CRITICAL_PARAMETERS"]
