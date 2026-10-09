"""
Shared CT intensity conversion, windowing, and export validation.

KiPA22 volumes (as redistributed on Hugging Face) are stored as uint16 NIfTI
with NO scaling metadata (scl_slope/scl_inter are NaN, no description, no header
extension) and no documentation of the intensity encoding. The stored values
are NOT Hounsfield units: they span roughly 0..2500 and cannot hold negative HU.

WORKING HYPOTHESIS (not proven): stored = HU + 1024. Evidence for it, from the
10 locally extracted cases: fat and soft-tissue histogram peaks sit at about
921 and 1074 (separation ~150, i.e. unit slope), implying offsets of ~1021 and
~1024 under typical fat (-100 HU) and soft-tissue (+50 HU) priors; kidney
medians come out at +80..+170 HU after subtracting 1024 (plausible for
contrast-enhanced kidney); and 7/10 volumes contain values floored at 0
(-1024 HU). This only constrains the offset to roughly 1000..1040. No dataset
documentation or header field confirms it. Hence the offset is a parameter, and
`DEFAULT_INTENSITY_OFFSET` is a dataset-specific hypothesis, not a fact.

The original preprocessing applied the HU window directly to stored values, so
every pixel clipped to white. The checks in `assess_export` exist to make that
class of failure loud.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

DEFAULT_INTENSITY_OFFSET = 1024.0  # HYPOTHESIS: stored = HU + 1024 (see module docstring)
DEFAULT_WINDOW_CENTER = 40.0       # HU, standard abdominal soft-tissue window
DEFAULT_WINDOW_WIDTH = 400.0       # HU


class InvalidImageError(ValueError):
    """Raised when an exported CT image fails validation."""


def _as_numeric_array(values, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype.kind not in "iuf":  # rejects bool, complex, object, str
        raise TypeError(f"{name} must be an integer or float array, got dtype {arr.dtype}")
    if arr.dtype.kind == "f" and not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def _finite_scalar(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def raw_to_hu(raw, intensity_offset: float = DEFAULT_INTENSITY_OFFSET) -> np.ndarray:
    """Convert stored intensities to HU as float64: ``hu = raw - intensity_offset``.

    Always promotes to float64 first, so unsigned storage (uint16) cannot wrap
    around when the offset is subtracted.
    """
    arr = _as_numeric_array(raw, "raw")
    offset = _finite_scalar(intensity_offset, "intensity_offset")
    return arr.astype(np.float64) - offset


def apply_window(hu, center: float = DEFAULT_WINDOW_CENTER, width: float = DEFAULT_WINDOW_WIDTH) -> np.ndarray:
    """Clip HU to [center - width/2, center + width/2] and map linearly to uint8 0..255.

    Fractional grey levels are truncated (not rounded), matching the original
    scripts' formula: -160 HU -> 0, 0 HU -> 102, +40 HU -> 127, +240 HU -> 255.
    """
    arr = _as_numeric_array(hu, "hu")
    center = _finite_scalar(center, "center")
    width = _finite_scalar(width, "width")
    if width <= 0:
        raise ValueError(f"width must be > 0, got {width}")
    low = center - width / 2.0
    high = center + width / 2.0
    clipped = np.clip(arr.astype(np.float64), low, high)
    return ((clipped - low) * 255.0 / width).astype(np.uint8)


def window_ct_slice(
    raw_slice,
    intensity_offset: float = DEFAULT_INTENSITY_OFFSET,
    center: float = DEFAULT_WINDOW_CENTER,
    width: float = DEFAULT_WINDOW_WIDTH,
) -> np.ndarray:
    """Stored 2D slice -> HU -> windowed uint8 image (same shape and orientation)."""
    arr = _as_numeric_array(raw_slice, "raw_slice")
    if arr.ndim != 2:
        raise ValueError(f"raw_slice must be 2D, got shape {arr.shape}")
    return apply_window(raw_to_hu(arr, intensity_offset), center, width)


@dataclass(frozen=True)
class QualityThresholds:
    """Failure limits for `assess_export`. Content-aware checks come first; the
    generic grey-level checks are a backstop, not the primary signal."""
    min_window_coverage: float = 0.50     # fraction of slice pixels whose HU is strictly inside the window
    kidney_median_hu_range: tuple = (20.0, 250.0)   # plausible contrast-enhanced kidney, after offset
    tumor_mean_grey_range: tuple = (30.0, 230.0)
    max_tumor_clipped_fraction: float = 0.50        # tumor pixels at exactly 0 or 255
    max_saturated_fraction: float = 0.25            # generic: pixels >= 254, and pixels <= 1
    min_grey_std: float = 20.0                      # generic
    min_unique_levels: int = 128                    # generic
    min_entropy_bits: float = 5.0                   # generic


@dataclass
class QualityReport:
    metrics: dict = field(default_factory=dict)
    failures: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


def _entropy_bits(grey: np.ndarray) -> float:
    p = np.bincount(grey.ravel(), minlength=256) / grey.size
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def assess_export(
    grey,
    raw_slice,
    *,
    intensity_offset: float = DEFAULT_INTENSITY_OFFSET,
    center: float = DEFAULT_WINDOW_CENTER,
    width: float = DEFAULT_WINDOW_WIDTH,
    kidney_mask: Optional[np.ndarray] = None,
    tumor_mask: Optional[np.ndarray] = None,
    thresholds: QualityThresholds = QualityThresholds(),
) -> QualityReport:
    """Measure an exported (windowed, pre-transpose) slice against its source.

    `grey` and `raw_slice` must share shape and orientation. The masks are used
    only for QC here; they are never drawn on or passed to the model.
    """
    t = thresholds
    rep = QualityReport()
    m, fails = rep.metrics, rep.failures

    grey = np.asarray(grey)
    if grey.ndim != 2 or grey.dtype != np.uint8 or grey.size == 0:
        fails.append(f"format: expected non-empty 2D uint8 image, got shape {grey.shape} dtype {grey.dtype}")
        return rep
    raw = _as_numeric_array(raw_slice, "raw_slice")
    if raw.shape != grey.shape:
        fails.append(f"format: raw slice shape {raw.shape} != image shape {grey.shape}")
        return rep

    hu = raw_to_hu(raw, intensity_offset)
    low, high = center - width / 2.0, center + width / 2.0

    # --- content-aware checks (independent of grey-level statistics) ---
    m["window_coverage"] = float(((hu > low) & (hu < high)).mean())
    if m["window_coverage"] < t.min_window_coverage:
        fails.append(
            f"window_coverage {m['window_coverage']:.2f} < {t.min_window_coverage}: most pixels fall outside "
            f"the HU window [{low:g}, {high:g}]; the intensity offset ({intensity_offset:g}) is likely wrong"
        )

    m["matches_reference_window"] = bool(np.array_equal(grey, apply_window(hu, center, width)))
    if not m["matches_reference_window"]:
        fails.append("integrity: image differs from window_ct_slice(raw_slice) with the stated parameters")

    for name, mask in (("kidney_mask", kidney_mask), ("tumor_mask", tumor_mask)):
        if mask is not None and np.asarray(mask).shape != grey.shape:
            fails.append(f"format: {name} shape {np.asarray(mask).shape} != image shape {grey.shape}")
            return rep

    if kidney_mask is not None and np.asarray(kidney_mask).any():
        k = np.asarray(kidney_mask).astype(bool)
        m["kidney_median_hu"] = float(np.median(hu[k]))
        lo, hi = t.kidney_median_hu_range
        if not lo <= m["kidney_median_hu"] <= hi:
            fails.append(f"kidney_median_hu {m['kidney_median_hu']:.0f} outside plausible [{lo:g}, {hi:g}] HU")

    if tumor_mask is not None and np.asarray(tumor_mask).any():
        tm = np.asarray(tumor_mask).astype(bool)
        m["tumor_mean_grey"] = float(grey[tm].mean())
        m["tumor_clipped_fraction"] = float(((grey[tm] == 0) | (grey[tm] == 255)).mean())
        lo, hi = t.tumor_mean_grey_range
        if not lo <= m["tumor_mean_grey"] <= hi:
            fails.append(f"tumor_mean_grey {m['tumor_mean_grey']:.0f} outside [{lo:g}, {hi:g}]")
        if m["tumor_clipped_fraction"] > t.max_tumor_clipped_fraction:
            fails.append(f"tumor_clipped_fraction {m['tumor_clipped_fraction']:.2f} > {t.max_tumor_clipped_fraction}")

    # --- generic grey-level backstop ---
    m["n_unique_levels"] = int(len(np.unique(grey)))
    m["grey_std"] = float(grey.std())
    m["entropy_bits"] = _entropy_bits(grey)
    m["frac_ge_254"] = float((grey >= 254).mean())
    m["frac_le_1"] = float((grey <= 1).mean())
    if m["n_unique_levels"] < t.min_unique_levels:
        fails.append(f"n_unique_levels {m['n_unique_levels']} < {t.min_unique_levels}")
    if m["grey_std"] < t.min_grey_std:
        fails.append(f"grey_std {m['grey_std']:.1f} < {t.min_grey_std}")
    if m["entropy_bits"] < t.min_entropy_bits:
        fails.append(f"entropy_bits {m['entropy_bits']:.2f} < {t.min_entropy_bits}")
    if m["frac_ge_254"] > t.max_saturated_fraction:
        fails.append(f"frac_ge_254 {m['frac_ge_254']:.2f} > {t.max_saturated_fraction} (saturated white)")
    if m["frac_le_1"] > t.max_saturated_fraction:
        fails.append(f"frac_le_1 {m['frac_le_1']:.2f} > {t.max_saturated_fraction} (saturated black)")
    return rep


def validate_export(grey, raw_slice, **kwargs) -> QualityReport:
    """Like `assess_export`, but raises `InvalidImageError` listing every failure."""
    rep = assess_export(grey, raw_slice, **kwargs)
    if not rep.ok:
        raise InvalidImageError("; ".join(rep.failures))
    return rep
