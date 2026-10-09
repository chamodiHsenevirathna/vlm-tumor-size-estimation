"""Focused tests for src/ct_preprocessing.py. Run: python -m unittest discover -s tests -v"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ct_preprocessing import (  # noqa: E402
    DEFAULT_INTENSITY_OFFSET,
    InvalidImageError,
    apply_window,
    assess_export,
    raw_to_hu,
    validate_export,
    window_ct_slice,
)


def synthetic_slice(offset=1024, seed=0, size=64):
    """Plausible stored slice: fat (-100 HU) | muscle (+50) | kidney (+130) | tumor (+70) | air floor, plus noise."""
    rng = np.random.default_rng(seed)
    hu = np.full((size, size), -100.0)
    hu[:, size // 4:size // 2] = 50.0
    kidney = np.zeros((size, size), bool)
    kidney[10:30, 36:56] = True
    tumor = np.zeros((size, size), bool)
    tumor[16:24, 42:50] = True
    hu[kidney] = 130.0
    hu[tumor] = 70.0
    hu += rng.normal(0, 15, hu.shape)
    raw = np.clip(np.round(hu + offset), 0, 65535).astype(np.uint16)
    return raw, kidney, tumor


class WindowFormula(unittest.TestCase):
    def test_known_values(self):
        hu = np.array([-160, -100, 0, 40, 240], dtype=float)
        np.testing.assert_array_equal(apply_window(hu, 40, 400), [0, 38, 102, 127, 255])

    def test_boundaries_and_clipping(self):
        hu = np.array([-1e6, -160.0001, -160, 239.9999, 240, 240.0001, 1e6])
        out = apply_window(hu, 40, 400)
        self.assertEqual(out[0], 0)
        self.assertEqual(out[1], 0)
        self.assertEqual(out[2], 0)
        self.assertEqual(out[3], 254)   # truncation, just under 255
        self.assertEqual(out[4], 255)
        self.assertEqual(out[5], 255)
        self.assertEqual(out[6], 255)

    def test_monotonic_non_decreasing(self):
        out = apply_window(np.linspace(-500, 500, 1001), 40, 400)
        self.assertTrue((np.diff(out.astype(int)) >= 0).all())

    def test_other_window(self):
        # lung-style window: [-1000, 0]
        np.testing.assert_array_equal(apply_window(np.array([-1500, -1000, -500, 0, 500]), -500, 1000), [0, 0, 127, 255, 255])

    def test_invalid_window_params(self):
        for width in (0, -400):
            with self.assertRaises(ValueError):
                apply_window(np.zeros(3), 40, width)
        with self.assertRaises(ValueError):
            apply_window(np.zeros(3), float("nan"), 400)
        with self.assertRaises(TypeError):
            apply_window(np.zeros(3), "40", 400)


class OffsetConversion(unittest.TestCase):
    def test_uint16_does_not_wrap(self):
        raw = np.array([0, 5, 1024, 1064, 65535], dtype=np.uint16)
        np.testing.assert_array_equal(raw_to_hu(raw, 1024), [-1024, -1019, 0, 40, 64511])

    def test_default_offset_is_documented_hypothesis(self):
        self.assertEqual(DEFAULT_INTENSITY_OFFSET, 1024.0)

    def test_offset_is_configurable(self):
        np.testing.assert_array_equal(raw_to_hu(np.array([1000]), 1000), [0])

    def test_zero_offset_reproduces_original_bug(self):
        # Original code windowed stored values directly: soft tissue (~1064) saturated to white.
        raw = np.array([[900, 1064, 1150]], dtype=np.uint16)
        np.testing.assert_array_equal(window_ct_slice(raw, 0), [[255, 255, 255]])
        # 900 -> -124 HU -> 22; 1064 -> +40 HU -> 127; 1150 -> +126 HU -> 182
        np.testing.assert_array_equal(window_ct_slice(raw, 1024), [[22, 127, 182]])


class InputTypesShapeDtype(unittest.TestCase):
    def test_accepts_int_uint_float_and_lists(self):
        for arr in (np.array([[1024, 1064]], dtype=np.uint16), np.array([[1024, 1064]], dtype=np.int32),
                    np.array([[1024.0, 1064.0]], dtype=np.float32), np.array([[1024.0, 1064.0]]), [[1024, 1064]]):
            out = window_ct_slice(arr, 1024)
            np.testing.assert_array_equal(out, [[102, 127]])

    def test_output_shape_and_dtype(self):
        for shape in ((1, 1), (5, 7), (138, 138)):
            out = window_ct_slice(np.full(shape, 1064, dtype=np.uint16))
            self.assertEqual(out.shape, shape)
            self.assertEqual(out.dtype, np.uint8)

    def test_input_not_modified(self):
        raw = np.array([[1000, 1100]], dtype=np.uint16)
        before = raw.copy()
        window_ct_slice(raw)
        np.testing.assert_array_equal(raw, before)

    def test_rejects_bad_types(self):
        for bad in (np.array([True, False]), np.array([1 + 2j]), np.array(["a"]), np.array([None], dtype=object)):
            with self.assertRaises(TypeError):
                raw_to_hu(bad)

    def test_rejects_non_finite(self):
        for bad in (np.nan, np.inf, -np.inf):
            with self.assertRaises(ValueError):
                raw_to_hu(np.array([1.0, bad]))

    def test_rejects_non_2d_slice(self):
        for arr in (np.zeros(5), np.zeros((2, 2, 2))):
            with self.assertRaises(ValueError):
                window_ct_slice(arr)

    def test_rejects_bad_offset(self):
        with self.assertRaises(ValueError):
            raw_to_hu(np.zeros(2), float("nan"))
        with self.assertRaises(TypeError):
            raw_to_hu(np.zeros(2), "1024")
        with self.assertRaises(TypeError):
            raw_to_hu(np.zeros(2), True)


class ExportValidation(unittest.TestCase):
    def setUp(self):
        self.raw, self.kidney, self.tumor = synthetic_slice()

    def _validate(self, grey, **kw):
        return validate_export(grey, self.raw, kidney_mask=self.kidney, tumor_mask=self.tumor, **kw)

    def test_good_export_passes(self):
        rep = self._validate(window_ct_slice(self.raw))
        self.assertTrue(rep.ok)
        self.assertTrue(rep.metrics["matches_reference_window"])
        self.assertGreater(rep.metrics["window_coverage"], 0.9)

    def test_wrong_offset_is_caught_by_content_checks(self):
        # offset 0 reproduces the original bug: everything white
        grey = window_ct_slice(self.raw, 0)
        with self.assertRaises(InvalidImageError) as cm:
            validate_export(grey, self.raw, intensity_offset=0, kidney_mask=self.kidney, tumor_mask=self.tumor)
        msg = str(cm.exception)
        self.assertIn("window_coverage", msg)
        self.assertIn("kidney_median_hu", msg)

    def test_content_check_fires_without_grey_statistics(self):
        # A wrong offset must be flagged by window_coverage alone, even with every
        # generic grey-level threshold disabled.
        from ct_preprocessing import QualityThresholds
        lax = QualityThresholds(min_unique_levels=0, min_grey_std=0, min_entropy_bits=-1,
                                max_saturated_fraction=1.0, tumor_mean_grey_range=(0, 255),
                                max_tumor_clipped_fraction=1.0, kidney_median_hu_range=(-1e9, 1e9))
        grey = window_ct_slice(self.raw, 0)
        rep = assess_export(grey, self.raw, intensity_offset=0, thresholds=lax)
        self.assertEqual([f.split()[0] for f in rep.failures], ["window_coverage"])

    def test_blank_images_rejected(self):
        for value in (0, 255, 128):
            grey = np.full(self.raw.shape, value, dtype=np.uint8)
            with self.assertRaises(InvalidImageError):
                validate_export(grey, self.raw)

    def test_nearly_constant_image_rejected(self):
        grey = np.full(self.raw.shape, 128, dtype=np.uint8)
        grey[0, :4] = (10, 20, 30, 40)
        with self.assertRaises(InvalidImageError):
            validate_export(grey, self.raw)

    def test_export_differing_from_source_rejected(self):
        grey = window_ct_slice(self.raw).copy()
        grey = np.roll(grey, 3, axis=1)  # e.g. a mistaken transform/orientation
        rep = assess_export(grey, self.raw)
        self.assertFalse(rep.metrics["matches_reference_window"])
        self.assertFalse(rep.ok)

    def test_bad_format_rejected(self):
        for grey in (window_ct_slice(self.raw).astype(np.float32), window_ct_slice(self.raw)[None],
                     np.zeros((0, 0), np.uint8), window_ct_slice(self.raw)[:10]):
            with self.assertRaises(InvalidImageError):
                validate_export(grey, self.raw)

    def test_mask_shape_mismatch_rejected(self):
        with self.assertRaises(InvalidImageError):
            validate_export(window_ct_slice(self.raw), self.raw, kidney_mask=np.ones((3, 3), bool))

    def test_raises_all_failures_in_one_error(self):
        grey = np.full(self.raw.shape, 255, dtype=np.uint8)
        with self.assertRaises(InvalidImageError) as cm:
            validate_export(grey, self.raw)
        self.assertGreater(str(cm.exception).count(";"), 2)


if __name__ == "__main__":
    unittest.main()
