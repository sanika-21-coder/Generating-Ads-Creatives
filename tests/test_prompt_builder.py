"""
Unit tests for modules/prompt_builder.py
"""

import unittest
from modules.prompt_builder import (
    build_prompt,
    build_negative_prompt,
    build_palette_string,
    TWEAK_MODIFIERS,
    DEFAULT_NEGATIVE_TERMS,
)

class TestPromptBuilder(unittest.TestCase):
    def setUp(self):
        self.brand = {
            "id": "test_brand",
            "name": "Botanica Essence",
            "product_name": "Sparkling Herbal Tonic",
            "category": "Beverage",
            "palette": [
                {"name": "Emerald Green", "hex": "#1B4D3E"},
                {"name": "Warm Gold", "hex": "#E4A853"}
            ],
            "mood": ["natural", "premium"],
            "things_to_avoid": ["plastic bottles", "neon glare"],
            "style_block": "soft morning sunlight, dewy condensation, luxury minimalist art direction"
        }

    def test_palette_formatting(self):
        palette_str = build_palette_string(self.brand["palette"])
        self.assertIn("Emerald Green (#1B4D3E)", palette_str)
        self.assertIn("Warm Gold (#E4A853)", palette_str)

    def test_positive_prompt_structure(self):
        prompt = build_prompt(
            self.brand,
            format_name="Instagram Square (1:1)",
            scenario="Hero Product Shot",
            extra_idea="surrounded by organic mint leaves",
            use_style_block=True
        )
        self.assertIn("Botanica Essence", prompt)
        self.assertIn("Sparkling Herbal Tonic", prompt)
        self.assertIn("hero center-stage commercial photography", prompt.lower())
        self.assertIn("surrounded by organic mint leaves", prompt)
        self.assertIn("1:1 square composition", prompt)
        self.assertIn("soft morning sunlight", prompt)

    def test_style_block_ablation(self):
        prompt_with = build_prompt(self.brand, use_style_block=True)
        prompt_without = build_prompt(self.brand, use_style_block=False)
        self.assertIn("Brand Visual DNA:", prompt_with)
        self.assertNotIn("Brand Visual DNA:", prompt_without)

    def test_negative_prompt_construction(self):
        neg = build_negative_prompt(self.brand, use_negative_prompt=True)
        self.assertIn("plastic bottles", neg)
        self.assertIn("neon glare", neg)
        self.assertIn("blurry", neg)
        self.assertIn("distorted text", neg)

    def test_negative_prompt_disabled(self):
        neg_empty = build_negative_prompt(self.brand, use_negative_prompt=False)
        self.assertEqual(neg_empty, "")

    def test_tweak_modifiers(self):
        prompt_vibrant = build_prompt(self.brand, active_tweak="more vibrant")
        self.assertIn(TWEAK_MODIFIERS["more vibrant"], prompt_vibrant)

if __name__ == "__main__":
    unittest.main()
