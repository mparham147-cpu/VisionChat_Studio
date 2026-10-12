"""
tests/test_operations.py
========================

Unit tests for the pure image functions. No API keys or model files
needed. Run with:

    python -m unittest discover -s tests -v
"""

import unittest

import numpy as np

from image_tools import operations as ops


class OperationTests(unittest.TestCase):
    """Verify the deterministic operations."""

    def test_noise_is_reproducible(self):
        source = np.full((20, 20, 3), 128, np.uint8)
        first = ops.apply_gaussian_noise(source, sigma=20, seed=42)
        second = ops.apply_gaussian_noise(source, sigma=20, seed=42)
        np.testing.assert_array_equal(first, second)

    def test_source_image_is_not_modified(self):
        source = np.full((20, 20, 3), 128, np.uint8)
        original = source.copy()
        ops.apply_gaussian_noise(source, sigma=20, seed=42)
        np.testing.assert_array_equal(source, original)

    def test_normalize_constant_image(self):
        constant = np.full((10, 10, 3), 100, np.uint8)
        result = ops.normalize_to_unit_range(constant)
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(result.max(), 0.0)

    def test_flip_horizontal(self):
        image = np.zeros((2, 3, 3), np.uint8)
        image[0, 0] = [1, 1, 1]
        flipped = ops.flip_image(image, "Horizontal")
        self.assertEqual(int(flipped[0, 2, 0]), 1)


if __name__ == "__main__":
    unittest.main()
