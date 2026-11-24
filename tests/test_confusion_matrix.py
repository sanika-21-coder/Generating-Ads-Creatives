"""
Unit tests for binary and 3x3 brand confusion matrices in modules/evaluation.py
"""

import unittest
from modules.evaluation import compute_confusion_matrix

class TestConfusionMatrix(unittest.TestCase):
    def setUp(self):
        self.brands = [
            {"id": "brand_a", "name": "Aura Botanicals"},
            {"id": "brand_b", "name": "Velvet Roast"},
            {"id": "brand_c", "name": "Luminaire Audio"}
        ]
        
        # 6 simulated evaluation records (2 per brand)
        self.eval_records = [
            # Brand A records (both correctly classified as Brand A)
            {"cache_key": "rec1", "model_name": "FLUX", "brand_id": "brand_a", "brand_scores": {"brand_a": 88.0, "brand_b": 40.0, "brand_c": 35.0}},
            {"cache_key": "rec2", "model_name": "SDXL", "brand_id": "brand_a", "brand_scores": {"brand_a": 82.0, "brand_b": 45.0, "brand_c": 30.0}},
            # Brand B records (one correctly classified, one misclassified as Brand A)
            {"cache_key": "rec3", "model_name": "FLUX", "brand_id": "brand_b", "brand_scores": {"brand_a": 50.0, "brand_b": 85.0, "brand_c": 38.0}},
            {"cache_key": "rec4", "model_name": "SDXL", "brand_id": "brand_b", "brand_scores": {"brand_a": 72.0, "brand_b": 60.0, "brand_c": 40.0}},
            # Brand C records (both correctly classified as Brand C)
            {"cache_key": "rec5", "model_name": "FLUX", "brand_id": "brand_c", "brand_scores": {"brand_a": 35.0, "brand_b": 38.0, "brand_c": 91.0}},
            {"cache_key": "rec6", "model_name": "SDXL", "brand_id": "brand_c", "brand_scores": {"brand_a": 30.0, "brand_b": 32.0, "brand_c": 86.0}},
        ]

    def test_confusion_matrix_counters(self):
        cm = compute_confusion_matrix(self.eval_records, self.brands, threshold=65.0)
        # Total judgments = 6 records * 3 brands = 18 judgments
        self.assertEqual(cm["total_judgments"], 18)
        self.assertGreater(cm["tp"], 0)
        self.assertGreater(cm["tn"], 0)
        self.assertGreaterEqual(cm["accuracy"], 70.0)
        self.assertGreaterEqual(cm["precision"], 70.0)
        self.assertGreaterEqual(cm["f1"], 70.0)

    def test_brand_attribution_matrix_dimensions(self):
        cm = compute_confusion_matrix(self.eval_records, self.brands, threshold=65.0)
        attr = cm["attribution_matrix"]
        self.assertEqual(set(attr.keys()), {"brand_a", "brand_b", "brand_c"})
        for b_id in ["brand_a", "brand_b", "brand_c"]:
            self.assertEqual(set(attr[b_id].keys()), {"brand_a", "brand_b", "brand_c"})

if __name__ == "__main__":
    unittest.main()
