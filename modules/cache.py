"""
Cache manager: saves and retrieves generated images and metadata JSON sidecars on disk.
Enables Demo Mode with zero API calls.
"""

import os
import json
import hashlib
from typing import Optional, Tuple, Dict, List
from PIL import Image

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cache")

def compute_cache_key(model_id: str, brand_id: str, scenario: str, seed: int, extra_text: str = "") -> str:
    """Generate deterministic MD5 hash key for caching."""
    slug_model = "flux" if "flux" in model_id.lower() else "sdxl"
    raw_str = f"{slug_model}::{brand_id}::{scenario}::{seed}::{extra_text.strip().lower()}"
    digest = hashlib.md5(raw_str.encode("utf-8")).hexdigest()[:12]
    return f"{brand_id}_{slug_model}_{seed}_{digest}"

def get_cached_item(cache_key: str) -> Optional[Tuple[Image.Image, Dict]]:
    """Load image and JSON sidecar if cached on disk."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    img_path = os.path.join(CACHE_DIR, f"{cache_key}.png")
    meta_path = os.path.join(CACHE_DIR, f"{cache_key}.json")
    
    if os.path.exists(img_path) and os.path.exists(meta_path):
        try:
            image = Image.open(img_path).convert("RGB")
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            return image, metadata
        except Exception as e:
            print(f"Error reading cache key {cache_key}: {e}")
            return None
    return None

def save_to_cache(image: Image.Image, metadata: Dict) -> str:
    """Saves image as PNG and metadata as JSON sidecar."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_key = metadata.get("cache_key")
    if not cache_key:
        cache_key = compute_cache_key(
            metadata.get("model", "unknown"),
            metadata.get("brand_id", "brand"),
            metadata.get("scenario", "hero"),
            metadata.get("seed", 42)
        )
        metadata["cache_key"] = cache_key
        
    img_path = os.path.join(CACHE_DIR, f"{cache_key}.png")
    meta_path = os.path.join(CACHE_DIR, f"{cache_key}.json")
    
    image.save(img_path, format="PNG")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    return cache_key

def list_all_cached_metadata() -> List[Dict]:
    """Returns list of all cached metadata sidecars."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    results = []
    for fname in os.listdir(CACHE_DIR):
        if fname.endswith(".json"):
            meta_path = os.path.join(CACHE_DIR, fname)
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    meta["_sidecar_file"] = fname
                    results.append(meta)
            except Exception:
                continue
    return results

def find_best_cached_match(
    brand_id: str,
    model_id: str,
    scenario: Optional[str] = None,
    seed: Optional[int] = None
) -> Optional[Tuple[Image.Image, Dict]]:
    """
    Finds exact or closest cached generation for smooth Demo Mode presentation.
    """
    slug_model = "flux" if "flux" in model_id.lower() else "sdxl"
    all_meta = list_all_cached_metadata()
    
    # Priority 1: Match brand, model, scenario, seed
    for meta in all_meta:
        m_model = "flux" if "flux" in meta.get("model", "").lower() else "sdxl"
        if meta.get("brand_id") == brand_id and m_model == slug_model:
            if scenario and meta.get("scenario") == scenario:
                if seed is not None and meta.get("seed") == seed:
                    key = meta.get("cache_key") or meta["_sidecar_file"][:-5]
                    item = get_cached_item(key)
                    if item:
                        return item
                        
    # Priority 2: Match brand, model, scenario
    for meta in all_meta:
        m_model = "flux" if "flux" in meta.get("model", "").lower() else "sdxl"
        if meta.get("brand_id") == brand_id and m_model == slug_model:
            if scenario and meta.get("scenario") == scenario:
                key = meta.get("cache_key") or meta["_sidecar_file"][:-5]
                item = get_cached_item(key)
                if item:
                    return item

    # Priority 3: Match brand and model
    for meta in all_meta:
        m_model = "flux" if "flux" in meta.get("model", "").lower() else "sdxl"
        if meta.get("brand_id") == brand_id and m_model == slug_model:
            key = meta.get("cache_key") or meta["_sidecar_file"][:-5]
            item = get_cached_item(key)
            if item:
                return item

    return None
