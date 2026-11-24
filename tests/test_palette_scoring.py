"""
Unit tests for CIEDE2000 palette adherence scoring in modules/evaluation.py
"""

import unittest
from PIL import Image
from modules.evaluation import (
    hex_to_rgb,
    rgb_to_hex,
    extract_dominant_colors,
    compute_palette_adherence,
)

class TestPaletteScoring(unittest.TestCase):
    def setUp(self):
        self.brand = {
            "id": "emerald_tea",
            "name": "Emerald Tea",
            "palette": [
                {"name": "Deep Forest Emerald", "hex": "#1B4D3E"},
                {"name": "Wild Sage", "hex": "#87A96B"},
                {"name": "Oat Cream", "hex": "#F5F2EB"}
            ]
        }

    def test_hex_rgb_conversions(self):
        rgb = hex_to_rgb("#1B4D3E")
        self.assertEqual(rgb, (27, 77, 62))
        hex_back = rgb_to_hex(rgb)
        self.assertEqual(hex_back.lower(), "#1b4d3e")

    def test_dominant_color_extraction(self):
        # Create solid emerald image
        img = Image.new("RGB", (100, 100), color=(27, 77, 62))
        colors = extract_dominant_colors(img, k=3)
        self.assertEqual(len(colors), 3)
        # Primary cluster center should match emerald
        r, g, b = colors[0]
        self.assertAlmostEqual(r, 27, delta=5)
        self.assertAlmostEqual(g, 77, delta=5)
        self.assertAlmostEqual(b, 62, delta=5)

    def test_palette_adherence_perfect_match(self):
        # Image colored with exact brand colors should receive high score (> 80)
        img = Image.new("RGB", (100, 100), color=(27, 77, 62))
        score, details = compute_palette_adherence(img, self.brand)
        self.assertGreaterEqual(score, 80.0)
        self.assertTrue(len(details) > 0)
        self.assertLessEqual(details[0]["delta_e"], 5.0)

    def test_palette_adherence_distant_colors(self):
        # Image colored with bright neon magenta should receive much lower score
        img = Image.new("RGB", (100, 100), color=(255, 0, 255))
        score_neon, details_neon = compute_palette_adherence(img, self.brand)
        self.assertLess(score_neon, 50.0)
        self.assertGreater(details_neon[0]["delta_e"], 40.0)

if __name__ == "__main__":
    unittest.main()
