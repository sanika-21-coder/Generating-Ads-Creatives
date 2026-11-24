"""
Unit tests for Brand Store and Cache sidecars
"""

import unittest
import os
from modules.brands import get_all_brands, get_brand_by_id, save_brand, delete_brand
from modules.cache import compute_cache_key, save_to_cache, get_cached_item
from PIL import Image

class TestCacheAndBrands(unittest.TestCase):
    def test_default_brands_loaded(self):
        brands = get_all_brands()
        self.assertGreaterEqual(len(brands), 3)
        names = [b["name"] for b in brands]
        self.assertIn("Aura Botanicals", names)
        self.assertIn("Velvet Roast", names)
        self.assertIn("Luminaire Audio", names)

    def test_brand_by_id(self):
        brand = get_brand_by_id("aura_botanicals")
        self.assertIsNotNone(brand)
        self.assertEqual(brand["id"], "aura_botanicals")
        self.assertIn("palette", brand)
        self.assertGreaterEqual(len(brand["palette"]), 4)

    def test_brand_crud(self):
        test_brand_data = {
            "name": "Zeta Test Brand",
            "product_name": "Smart Watch",
            "category": "Wearables",
            "palette": [{"name": "Blue", "hex": "#0000FF"}],
            "mood": ["futuristic"]
        }
        brand_id = save_brand(test_brand_data)
        self.assertTrue(brand_id.startswith("zeta"))
        
        loaded = get_brand_by_id(brand_id)
        self.assertEqual(loaded["product_name"], "Smart Watch")
        
        # Cleanup
        deleted = delete_brand(brand_id)
        self.assertTrue(deleted)

    def test_cache_sidecar_roundtrip(self):
        img = Image.new("RGB", (64, 64), color=(100, 150, 200))
        meta = {
            "cache_key": "test_roundtrip_key_123",
            "model": "test_model",
            "scenario": "Hero Shot",
            "seed": 999
        }
        key = save_to_cache(img, meta)
        self.assertEqual(key, "test_roundtrip_key_123")
        
        retrieved = get_cached_item(key)
        self.assertIsNotNone(retrieved)
        ret_img, ret_meta = retrieved
        self.assertEqual(ret_img.size, (64, 64))
        self.assertEqual(ret_meta["seed"], 999)

if __name__ == "__main__":
    unittest.main()
