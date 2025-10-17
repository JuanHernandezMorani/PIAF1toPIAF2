"""Physiologically grounded human vision simulation utilities.

This module implements simplified yet biologically informed models of human
vision across different lighting regimes (photopic, mesopic, scotopic) and the
magnocellular/parvocellular pathways.  The implementation draws inspiration
from neurophysiology literature describing cone/rod spectral sensitivities and
post-retinal processing (Boynton, *Human Color Vision*, 1996; Stockman &
Sharpe, *Vision Research*, 2000; Dacey, *Annual Review of Neuroscience*, 2004).

The functions exposed here are designed to be computationally lightweight for
integration in training pipelines.  They operate exclusively on ``PIL.Image``
objects and rely only on NumPy and Pillow to avoid additional dependencies.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageFilter


_LMS_FROM_RGB = np.array(
    [
        [0.31399022, 0.63951294, 0.04649755],
        [0.15537241, 0.75789446, 0.08670142],
        [0.01775239, 0.10944209, 0.87256922],
    ],
    dtype=np.float32,
)
"""Matrix converting linear sRGB to Hunt-Pointer-Estevez LMS coordinates."""

_RGB_FROM_LMS = np.linalg.inv(_LMS_FROM_RGB)


@dataclass(frozen=True)
class _SpectralCurves:
    """Photoreceptor luminous efficiency functions (CIE 1931/1951)."""

    photopic: np.ndarray  # V(lambda)
    scotopic: np.ndarray  # V'(lambda)


def _generate_spectral_curves() -> _SpectralCurves:
    wavelengths = np.linspace(380.0, 780.0, 81, dtype=np.float32)
    # Photopic luminous efficiency function approximated with a skewed Gaussian
    photopic = np.exp(-0.5 * ((wavelengths - 555.0) / 50.0) ** 2)
    photopic /= photopic.max()
    # Scotopic luminous efficiency with peak near 507 nm
    scotopic = np.exp(-0.5 * ((wavelengths - 507.0) / 40.0) ** 2)
    scotopic /= scotopic.max()
    return _SpectralCurves(photopic=photopic, scotopic=scotopic)


_CURVES = _generate_spectral_curves()


def _curve_value(curve: np.ndarray, wavelength: float) -> float:
    index = int(np.clip(round((wavelength - 380.0) / 5.0), 0, len(curve) - 1))
    return float(curve[index])


def _ensure_rgba(image: Image.Image) -> Tuple[Image.Image, np.ndarray]:
    rgba = image.convert("RGBA")
    np_image = np.asarray(rgba, dtype=np.float32) / 255.0
    return rgba, np_image


def _merge_rgba(np_rgb: np.ndarray, alpha: np.ndarray) -> Image.Image:
    np_rgb = np.clip(np_rgb, 0.0, 1.0)
    merged = np.concatenate([np_rgb, alpha[..., None]], axis=2)
    return Image.fromarray((merged * 255.0).astype(np.uint8), mode="RGBA")


def _apply_gaussian(np_img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np_img
    pil_img = Image.fromarray(np.clip(np_img * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return np.asarray(blurred, dtype=np.float32) / 255.0


def _apply_unsharp_mask(np_img: np.ndarray, radius: float, amount: float) -> np.ndarray:
    if radius <= 0 or amount <= 0:
        return np_img
    low = _apply_gaussian(np_img, radius)
    detail = np_img - low
    return np.clip(np_img + amount * detail, 0.0, 1.0)


def _apply_high_pass(np_img: np.ndarray, radius: float) -> np.ndarray:
    low = _apply_gaussian(np_img, radius)
    return np.clip(np_img - low + 0.5, 0.0, 1.0)


def _rgb_to_linear(np_rgb: np.ndarray) -> np.ndarray:
    mask = np_rgb <= 0.04045
    linear = np.empty_like(np_rgb)
    linear[mask] = np_rgb[mask] / 12.92
    linear[~mask] = ((np_rgb[~mask] + 0.055) / 1.055) ** 2.4
    return linear


def _linear_to_srgb(np_rgb: np.ndarray) -> np.ndarray:
    mask = np_rgb <= 0.0031308
    srgb = np.empty_like(np_rgb)
    srgb[mask] = 12.92 * np_rgb[mask]
    srgb[~mask] = 1.055 * np_rgb[~mask] ** (1.0 / 2.4) - 0.055
    return srgb


def _clip_alpha(np_rgba: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np_rgba[..., :3], np_rgba[..., 3]


def simulate_photopic_vision(image: Image.Image, luminance_cd_m2: float = 100.0) -> Image.Image:
    """Simulate cone-mediated photopic vision.

    The simulation approximates foveal sharpness and the spectral sensitivity of
    L-, M-, and S-cones under high luminance conditions (>3 cd/m²).

    References
    ----------
    * Boynton, R. M. (1996). *Human Color Vision* (2nd ed.).
    * Stockman, A., & Sharpe, L. T. (2000). The spectral sensitivities of the
      middle- and long-wavelength-sensitive cones derived from measurements in
      observers of known genotype. *Vision Research*.
    """

    rgba, np_rgba = _ensure_rgba(image)
    rgb, alpha = _clip_alpha(np_rgba)

    linear = _rgb_to_linear(rgb)
    lms = linear @ _LMS_FROM_RGB.T
    adaptation_gain = min(max(luminance_cd_m2 / 100.0, 0.25), 4.0)
    lms *= adaptation_gain
    linear_out = lms @ _RGB_FROM_LMS.T
    srgb = _linear_to_srgb(np.clip(linear_out, 0.0, 1.0))
    srgb = _apply_unsharp_mask(srgb, radius=1.2, amount=0.8)
    srgb = np.clip(srgb, 0.0, 1.0)
    return _merge_rgba(srgb, alpha)


def simulate_scotopic_vision(image: Image.Image, luminance_cd_m2: float = 0.001) -> Image.Image:
    """Simulate rod-mediated scotopic vision.

    Under very low luminance (<0.001 cd/m²) rods dominate.  The output image is
    rendered achromatic with reduced spatial resolution and enhanced contrast.

    References
    ----------
    * Aguilar, M., & Stiles, W. S. (1954). Saturation of the rod mechanism of
      the retina at high levels of stimulation. *Optica Acta*.
    * Sharpe, L. T., et al. (1992). Rod spectral sensitivities. *Vision Research*.
    """

    rgba, np_rgba = _ensure_rgba(image)
    rgb, alpha = _clip_alpha(np_rgba)
    linear = _rgb_to_linear(rgb)
    lms = linear @ _LMS_FROM_RGB.T
    rod_response = (0.1 * lms[..., 0] + 0.8 * lms[..., 1] + 0.1 * lms[..., 2])
    rod_response = rod_response[..., None]
    rod_response = np.repeat(rod_response, 3, axis=2)
    rod_response = rod_response ** 0.8  # enhance darker regions
    blur_radius = 6.0
    blurred = _apply_gaussian(np.clip(rod_response, 0.0, 1.0), sigma=blur_radius)
    contrast_gain = min(max(1.0 / max(luminance_cd_m2, 1e-4), 5.0), 50.0)
    blurred = np.clip(0.5 + (blurred - 0.5) * math.log10(contrast_gain + 1.0), 0.0, 1.0)
    return _merge_rgba(blurred, alpha)


def simulate_mesopic_vision(image: Image.Image, luminance_cd_m2: float = 1.0) -> Image.Image:
    """Simulate mesopic vision as a blend between photopic and scotopic states."""

    luminance_cd_m2 = max(luminance_cd_m2, 1e-4)
    blend = (math.log10(luminance_cd_m2 + 1e-4) + 4.0) / 4.0
    blend = float(np.clip(blend, 0.0, 1.0))
    photopic_img = simulate_photopic_vision(image, max(luminance_cd_m2, 3.0))
    scotopic_img = simulate_scotopic_vision(image, luminance_cd_m2)
    photopic = np.asarray(photopic_img, dtype=np.float32) / 255.0
    scotopic = np.asarray(scotopic_img, dtype=np.float32) / 255.0
    mixed = photopic * blend + scotopic * (1.0 - blend)
    rgb, alpha = mixed[..., :3], mixed[..., 3]
    return _merge_rgba(rgb, alpha)


def simulate_magnocellular_pathway(image: Image.Image) -> Image.Image:
    """Simulate the achromatic, low-resolution magnocellular pathway."""

    rgba, np_rgba = _ensure_rgba(image)
    rgb, alpha = _clip_alpha(np_rgba)
    gray = np.dot(rgb, np.array([0.299, 0.587, 0.114], dtype=np.float32))
    gray = gray[..., None]
    low_pass = _apply_gaussian(np.repeat(gray, 3, axis=2), sigma=3.0)
    contrast = np.clip(0.5 + (low_pass - 0.5) * 1.5, 0.0, 1.0)
    return _merge_rgba(contrast, alpha)


def simulate_parvocellular_pathway(image: Image.Image) -> Image.Image:
    """Simulate the high-acuity, chromatic parvocellular pathway."""

    rgba, np_rgba = _ensure_rgba(image)
    rgb, alpha = _clip_alpha(np_rgba)
    linear = _rgb_to_linear(rgb)
    lms = linear @ _LMS_FROM_RGB.T
    l_minus_m = lms[..., 0] - lms[..., 1]
    s_minus_lm = lms[..., 2] - 0.5 * (lms[..., 0] + lms[..., 1])
    opponent = np.stack([l_minus_m, s_minus_lm, linear[..., 1]], axis=2)
    high = _apply_high_pass(opponent, radius=1.0)
    reconstructed = high @ np.array(
        [
            [0.8, 0.2, 0.2],
            [-0.2, 0.5, -0.1],
            [0.2, -0.1, 0.7],
        ],
        dtype=np.float32,
    )
    srgb = np.clip(_linear_to_srgb(np.clip(reconstructed, 0.0, 1.0)), 0.0, 1.0)
    return _merge_rgba(srgb, alpha)


def apply_photopic_adaptation(image: Image.Image, adaptation_time: float = 2.0) -> Image.Image:
    """Apply photopic light adaptation with chromatic recovery."""

    rgba, np_rgba = _ensure_rgba(image)
    rgb, alpha = _clip_alpha(np_rgba)
    factor = np.clip(adaptation_time / 5.0, 0.0, 1.0)
    white_point = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    rgb = rgb * (1.0 - 0.1 * factor) + white_point * 0.1 * factor
    sharpened = _apply_unsharp_mask(rgb, radius=0.8, amount=0.4 * factor)
    srgb = np.clip(sharpened, 0.0, 1.0)
    return _merge_rgba(srgb, alpha)


def apply_scotopic_adaptation(image: Image.Image, adaptation_time: float = 30.0) -> Image.Image:
    """Apply scotopic adaptation emphasising rod sensitivity."""

    rgba, np_rgba = _ensure_rgba(image)
    rgb, alpha = _clip_alpha(np_rgba)
    factor = np.clip(adaptation_time / 30.0, 0.0, 1.0)
    gray = np.dot(rgb, np.array([0.062, 0.678, 0.26], dtype=np.float32))
    gray = gray[..., None]
    boosted = np.clip(gray ** (1.0 - 0.5 * factor), 0.0, 1.0)
    low = _apply_gaussian(np.repeat(boosted, 3, axis=2), sigma=2.5 * factor)
    return _merge_rgba(low, alpha)


def simulate_purkinje_shift(image: Image.Image, mesopic_level: float = 0.5) -> Image.Image:
    """Apply Purkinje shift by reweighting spectral sensitivities."""

    rgba, np_rgba = _ensure_rgba(image)
    rgb, alpha = _clip_alpha(np_rgba)
    mesopic_level = np.clip(mesopic_level, 0.0, 1.0)
    linear = _rgb_to_linear(rgb)
    lms = linear @ _LMS_FROM_RGB.T
    photopic_weight = 1.0 - mesopic_level
    scotopic_weight = mesopic_level
    # Evaluate luminous efficiency curves near canonical RGB primaries.
    red_peak = photopic_weight * _curve_value(_CURVES.photopic, 610.0) + scotopic_weight * _curve_value(_CURVES.scotopic, 610.0)
    green_peak = photopic_weight * _curve_value(_CURVES.photopic, 555.0) + scotopic_weight * _curve_value(_CURVES.scotopic, 535.0)
    blue_peak = photopic_weight * _curve_value(_CURVES.photopic, 445.0) + scotopic_weight * _curve_value(_CURVES.scotopic, 475.0)
    weighting = np.array([red_peak, green_peak, blue_peak], dtype=np.float32)
    weighting /= weighting.max()
    blended = lms * weighting
    out_linear = blended @ _RGB_FROM_LMS.T
    srgb = np.clip(_linear_to_srgb(np.clip(out_linear, 0.0, 1.0)), 0.0, 1.0)
    return _merge_rgba(srgb, alpha)


def _laplacian_of_gaussian(np_img: np.ndarray, sigma: float) -> np.ndarray:
    low = _apply_gaussian(np_img, sigma)
    return np.clip(np_img - low, -1.0, 1.0)


def simulate_retinal_processing(image: Image.Image) -> Image.Image:
    """Approximate retinal preprocessing via lateral inhibition."""

    rgba, np_rgba = _ensure_rgba(image)
    rgb, alpha = _clip_alpha(np_rgba)
    log = _laplacian_of_gaussian(rgb, sigma=1.2)
    normalized = np.clip(rgb + log * 0.6, 0.0, 1.0)
    return _merge_rgba(normalized, alpha)


def _gabor_kernel(size: int, sigma: float, theta: float, frequency: float) -> np.ndarray:
    half = size // 2
    y, x = np.mgrid[-half : half + 1, -half : half + 1]
    rotx = x * math.cos(theta) + y * math.sin(theta)
    roty = -x * math.sin(theta) + y * math.cos(theta)
    g = np.exp(-(rotx**2 + roty**2) / (2.0 * sigma**2)) * np.cos(2.0 * math.pi * frequency * rotx)
    return g.astype(np.float32)


def _apply_convolution(np_img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    padded = np.pad(np_img, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="reflect")
    out = np.zeros_like(np_img)
    for y in range(np_img.shape[0]):
        for x in range(np_img.shape[1]):
            region = padded[y : y + k_h, x : x + k_w]
            out[y, x] = np.tensordot(region, kernel, axes=([0, 1], [0, 1]))
    return out


def simulate_cortical_processing(image: Image.Image) -> Image.Image:
    """Approximate cortical visual processing with Gabor filter bank."""

    rgba, np_rgba = _ensure_rgba(image)
    rgb, alpha = _clip_alpha(np_rgba)
    kernels = [
        _gabor_kernel(9, sigma=2.0, theta=theta, frequency=0.1) for theta in (0.0, math.pi / 4, math.pi / 2)
    ]
    accum = np.zeros_like(rgb)
    for kernel in kernels:
        filtered = _apply_convolution(rgb, kernel)
        accum += np.abs(filtered)
    accum /= max(len(kernels), 1)
    max_val = float(accum.max())
    if max_val > 1e-6:
        accum /= max_val
    enhanced = np.clip(rgb + accum * 0.3, 0.0, 1.0)
    return _merge_rgba(enhanced, alpha)


def calculate_contrast_sensitivity(image: Image.Image, spatial_frequency: float) -> float:
    """Compute contrast sensitivity using a log-parabola CSF model."""

    spatial_frequency = max(spatial_frequency, 0.1)
    f = spatial_frequency
    csf_peak = 4.0
    peak_sensitivity = 1.0 / 0.02
    csf = peak_sensitivity * math.exp(-((math.log2(f / csf_peak)) ** 2) / (2 * (0.4**2)))
    _, np_rgba = _ensure_rgba(image)
    luminance = np.dot(np_rgba[..., :3], np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))
    contrast = float(np.std(luminance) / max(np.mean(luminance), 1e-6))
    return float(csf * (1.0 + contrast))


def validate_photopic_parameters(image: Image.Image) -> Dict[str, float]:
    """Validate photopic metrics such as luminance and contrast."""

    rgba, np_rgba = _ensure_rgba(image)
    rgb, _ = _clip_alpha(np_rgba)
    luminance = np.dot(rgb, np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))
    mean_luminance = float(luminance.mean() * 100.0)
    contrast = float((luminance.max() - luminance.min()) / max(luminance.mean(), 1e-3))
    efficiency = float(np.clip(luminance.mean() / np.sqrt((rgb**2).mean()), 0.0, 1.0))
    return {
        "mean_luminance_cd_m2": mean_luminance,
        "weber_contrast": contrast,
        "visual_efficiency": efficiency,
        "estimated_acuity_20_over": 20.0 / max(1.0, 60.0 * (1.0 - efficiency) + 1.0),
    }


def simulate_visual_acuity(image: Image.Image, acuity_mar: float = 1.0) -> Image.Image:
    """Apply low-pass filtering to approximate acuity limits."""

    rgba, np_rgba = _ensure_rgba(image)
    rgb, alpha = _clip_alpha(np_rgba)
    sigma = np.clip(acuity_mar / 1.5, 0.2, 6.0)
    blurred = _apply_gaussian(rgb, sigma)
    return _merge_rgba(blurred, alpha)


def _lab_from_rgb(np_rgb: np.ndarray) -> np.ndarray:
    rgb = _rgb_to_linear(np_rgb)
    xyz = rgb @ np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float32,
    ).T
    xyz /= np.array([0.95047, 1.0, 1.08883], dtype=np.float32)

    def f(t: np.ndarray) -> np.ndarray:
        delta = 6.0 / 29.0
        return np.where(t > delta**3, t ** (1.0 / 3.0), t / (3 * delta**2) + 4.0 / 29.0)

    fx, fy, fz = f(xyz[..., 0]), f(xyz[..., 1]), f(xyz[..., 2])
    lab = np.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], axis=2)
    return lab


def _rgb_from_lab(lab: np.ndarray) -> np.ndarray:
    l = (lab[..., 0] + 16.0) / 116.0
    a = lab[..., 1] / 500.0
    b = lab[..., 2] / 200.0
    fx = a + l
    fy = l
    fz = fy - b
    delta = 6.0 / 29.0

    def f_inv(t: np.ndarray) -> np.ndarray:
        return np.where(t > delta, t**3, 3 * delta**2 * (t - 4.0 / 29.0))

    x = f_inv(fx) * 0.95047
    y = f_inv(fy)
    z = f_inv(fz) * 1.08883
    xyz = np.stack([x, y, z], axis=2)
    rgb_linear = xyz @ np.array(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ],
        dtype=np.float32,
    ).T
    return np.clip(_linear_to_srgb(np.clip(rgb_linear, 0.0, 1.0)), 0.0, 1.0)


def apply_human_vision_simulation(
    image: Image.Image,
    lighting_condition: str = "photopic",
    adaptation_level: float = 1.0,
    luminance_cd_m2: float | None = None,
) -> Image.Image:
    """Composed simulation of human vision across lighting conditions."""

    lighting_condition = lighting_condition.lower()
    luminance_defaults = {"photopic": 100.0, "mesopic": 1.0, "scotopic": 0.001}
    luminance_cd_m2 = luminance_cd_m2 or luminance_defaults.get(lighting_condition, 10.0)

    retinal = simulate_retinal_processing(image)

    if lighting_condition == "scotopic":
        vision = simulate_scotopic_vision(retinal, luminance_cd_m2)
        adaptation = apply_scotopic_adaptation(vision, adaptation_time=30.0 * adaptation_level)
    elif lighting_condition == "mesopic":
        vision = simulate_mesopic_vision(retinal, luminance_cd_m2)
        adaptation = simulate_purkinje_shift(vision, mesopic_level=adaptation_level)
    else:
        vision = simulate_photopic_vision(retinal, luminance_cd_m2)
        adaptation = apply_photopic_adaptation(vision, adaptation_time=5.0 * adaptation_level)

    cortical = simulate_cortical_processing(adaptation)
    acuity_mar = 1.0 + (1.0 - np.clip(adaptation_level, 0.0, 1.0)) * 2.0
    return simulate_visual_acuity(cortical, acuity_mar=acuity_mar)


def generate_vision_optimized_palette(
    image: Image.Image,
    lighting_condition: str,
    n_colors: int = 8,
) -> List[Tuple[int, int, int]]:
    """Generate palettes optimised for human perception."""

    n_colors = max(2, n_colors)
    rgba, np_rgba = _ensure_rgba(image)
    rgb, _ = _clip_alpha(np_rgba)
    h, w, _ = rgb.shape
    pixels = rgb.reshape((h * w, 3))
    lab = _lab_from_rgb(pixels.reshape((h, w, 3))).reshape((h * w, 3))

    rng = np.random.default_rng(1234)
    sample_size = min(5000, len(lab))
    if sample_size == 0:
        return [(0, 0, 0)] * n_colors
    indices = rng.choice(len(lab), size=sample_size, replace=False)
    samples = lab[indices]
    replace_centers = len(samples) < n_colors
    centers = samples[rng.choice(len(samples), size=n_colors, replace=replace_centers)]

    for _ in range(10):
        distances = np.linalg.norm(samples[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)
        for i in range(n_colors):
            mask = labels == i
            if np.any(mask):
                centers[i] = samples[mask].mean(axis=0)

    centers_rgb = _rgb_from_lab(centers.reshape((n_colors, 1, 3))).reshape((n_colors, 3))
    condition_scale = {"photopic": 1.0, "mesopic": 0.85, "scotopic": 0.6}
    scale = condition_scale.get(lighting_condition.lower(), 1.0)
    centers_rgb = np.clip(centers_rgb * scale, 0.0, 1.0)
    return [tuple((center * 255.0).astype(np.uint8)) for center in centers_rgb]


__all__ = [
    "simulate_photopic_vision",
    "simulate_scotopic_vision",
    "simulate_mesopic_vision",
    "simulate_magnocellular_pathway",
    "simulate_parvocellular_pathway",
    "apply_photopic_adaptation",
    "apply_scotopic_adaptation",
    "simulate_purkinje_shift",
    "simulate_retinal_processing",
    "simulate_cortical_processing",
    "calculate_contrast_sensitivity",
    "validate_photopic_parameters",
    "simulate_visual_acuity",
    "apply_human_vision_simulation",
    "generate_vision_optimized_palette",
]
