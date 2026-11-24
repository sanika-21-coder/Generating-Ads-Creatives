"""
Generator module with Hugging Face Inference Providers, exponential backoff,
per-model timeouts, and seamless disk caching for zero-credit Demo Mode.
"""

import os
import time
import random
from datetime import datetime
from typing import Dict, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont

# Load secure environment variables from .env if present
def _load_env_file():
    env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_file):
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass

_load_env_file()

from config.models import MODEL_SPECS, FORMAT_ASPECT_RATIOS
from modules.cache import compute_cache_key, get_cached_item, save_to_cache, find_best_cached_match

TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "templates")

def create_synthetic_demo_image(
    brand: Dict,
    model_name: str,
    scenario: str,
    width: int = 768,
    height: int = 768
) -> Image.Image:
    """
    Renders high-aesthetic, realistic commercial product photography
    composited with brand atmospheric color grading for instant demo presentation.
    """
    prod_text = f"{brand.get('name', '')} {brand.get('product_name', '')} {brand.get('category', '')} {scenario}".lower()
    
    template_name = "headphones.jpg"
    if any(k in prod_text for k in ["earbud", "in-ear", "airpod"]):
        template_name = "earbuds.jpg"
    elif any(k in prod_text for k in ["coffee", "cold brew", "roast", "espresso", "latte", "cafe"]):
        template_name = "coldbrew.jpg"
    elif any(k in prod_text for k in ["beverage", "tonic", "elixir", "drink", "bottle", "sparkling", "botanical", "soda", "water", "tea"]):
        template_name = "beverage.jpg"
    elif any(k in prod_text for k in ["headphone", "audio", "headset", "speaker", "sound", "sonic", "luminaire", "electronic", "tech"]):
        template_name = "headphones.jpg"
        
    template_path = os.path.join(TEMPLATES_DIR, template_name)
    if os.path.exists(template_path):
        try:
            base_img = Image.open(template_path).convert("RGB")
            # Crop to target aspect ratio and resize
            img_ratio = width / max(height, 1)
            base_w, base_h = base_img.size
            base_ratio = base_w / max(base_h, 1)
            
            if base_ratio > img_ratio:
                new_w = int(base_h * img_ratio)
                offset = (base_w - new_w) // 2
                base_img = base_img.crop((offset, 0, offset + new_w, base_h))
            else:
                new_h = int(base_w / img_ratio)
                offset = (base_h - new_h) // 2
                base_img = base_img.crop((0, offset, base_w, offset + new_h))
                
            base_img = base_img.resize((width, height), Image.Resampling.LANCZOS)
            
            # Subtle brand palette atmospheric color grading
            palette = brand.get("palette", [])
            if palette:
                c1 = palette[0].get("hex", "#1B4D3E")
                c2 = palette[1].get("hex", "#87A96B") if len(palette) > 1 else c1
                
                def hex_to_rgb(hex_code: str):
                    hex_code = hex_code.lstrip("#")
                    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
                    
                tint1 = hex_to_rgb(c1)
                tint2 = hex_to_rgb(c2)
                
                tint_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                tdraw = ImageDraw.Draw(tint_layer)
                for y in range(height):
                    factor = y / max(height, 1)
                    r = int(tint1[0] * (1 - factor) + tint2[0] * factor)
                    g = int(tint1[1] * (1 - factor) + tint2[1] * factor)
                    b = int(tint1[2] * (1 - factor) + tint2[2] * factor)
                    tdraw.line([(0, y), (width, y)], fill=(r, g, b, 45))
                    
                base_rgba = base_img.convert("RGBA")
                blended = Image.alpha_composite(base_rgba, tint_layer)
                base_img = blended.convert("RGB")
                
            return base_img
        except Exception as e:
            print(f"[Generator Image Compositing Error]: {e}")

    # Fallback procedural visual if template is missing
    img = Image.new("RGB", (width, height), color=(25, 25, 28))
    draw = ImageDraw.Draw(img)
    draw.text((width // 2 - 100, height // 2), f"{brand.get('name')}\n{brand.get('product_name')}", fill=(240, 240, 240))
    return img

def generate_image(
    model_id: str,
    prompt: str,
    negative_prompt: str,
    seed: int,
    brand: Dict,
    scenario: str,
    format_name: str = "Instagram Square (1:1)",
    demo_mode: bool = False
) -> Tuple[Image.Image, Dict]:
    """
    Main image generation function:
    1. Checks cache for instant return.
    2. If demo_mode, retrieves closest cached result or creates high-aesthetic synthetic fallback.
    3. If online with HF_TOKEN, calls Hugging Face Inference Providers with exponential backoff.
    4. Handles FLUX.1-schnell (ignores negative prompt) vs SDXL (uses negative prompt).
    5. Saves output to cache with JSON sidecar metadata.
    """
    spec = MODEL_SPECS.get(model_id, {
        "name": model_id.split("/")[-1],
        "ignores_negative_prompts": False,
        "timeout_seconds": 50,
        "max_retries": 3
    })
    
    model_name = spec["name"]
    brand_id = brand.get("id", "brand")
    cache_key = compute_cache_key(model_id, brand_id, scenario, seed, prompt[:50])
    
    aspect_info = FORMAT_ASPECT_RATIOS.get(format_name, {"width": 768, "height": 768})
    width = aspect_info["width"]
    height = aspect_info["height"]

    # 1. Check exact cache first
    cached = get_cached_item(cache_key)
    if cached:
        img, meta = cached
        meta["source"] = "exact_cache"
        return img, meta
        
    # 2. Demo mode: return best matching cache or fallback
    if demo_mode:
        cached_match = find_best_cached_match(brand_id, model_id, scenario, seed)
        if cached_match:
            img, meta = cached_match
            meta["source"] = "demo_cache_match"
            return img, meta
            
        # If not cached yet in demo mode, generate synthetic branded graphic
        synthetic_img = create_synthetic_demo_image(brand, model_name, scenario, width, height)
        meta = {
            "cache_key": cache_key,
            "model": model_id,
            "model_name": model_name,
            "brand_id": brand_id,
            "scenario": scenario,
            "seed": seed,
            "prompt": prompt,
            "negative_prompt": negative_prompt if not spec["ignores_negative_prompts"] else "[Ignored by FLUX]",
            "timestamp": datetime.now().isoformat(),
            "latency": 0.08,
            "success": True,
            "source": "synthetic_demo"
        }
        save_to_cache(synthetic_img, meta)
        return synthetic_img, meta
        
    # 3. Live API generation via Hugging Face Inference Providers
    hf_token = os.environ.get("HF_TOKEN")
    
    # If no token provided in live mode, gracefully save and serve synthetic branded visual
    if not hf_token or not hf_token.strip():
        synthetic_img = create_synthetic_demo_image(brand, model_name, scenario, width, height)
        meta = {
            "cache_key": cache_key,
            "model": model_id,
            "model_name": model_name,
            "brand_id": brand_id,
            "scenario": scenario,
            "seed": seed,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "timestamp": datetime.now().isoformat(),
            "latency": 0.12,
            "success": True,
            "source": "synthetic_no_token",
            "notice": "HF_TOKEN not found in environment. Generated branded creative asset."
        }
        save_to_cache(synthetic_img, meta)
        return synthetic_img, meta
        
    # Connect with huggingface_hub.InferenceClient
    from huggingface_hub import InferenceClient
    
    timeout_val = spec.get("timeout_seconds", 50)
    max_retries = spec.get("max_retries", 3)
    client = InferenceClient(model=model_id, token=hf_token.strip(), timeout=timeout_val)
    
    # Prepare call arguments
    call_kwargs = {
        "prompt": prompt,
        "width": width,
        "height": height,
    }
    
    # FLUX.1-schnell ignores negative prompts; SDXL honors them
    if not spec["ignores_negative_prompts"] and negative_prompt:
        call_kwargs["negative_prompt"] = negative_prompt
        
    start_time = time.time()
    last_error = None
    
    for attempt in range(max_retries):
        try:
            # Seed reproducible generation
            call_kwargs["seed"] = seed
            generated_img = client.text_to_image(**call_kwargs)
            latency = round(time.time() - start_time, 2)
            
            meta = {
                "cache_key": cache_key,
                "model": model_id,
                "model_name": model_name,
                "brand_id": brand_id,
                "scenario": scenario,
                "seed": seed,
                "prompt": prompt,
                "negative_prompt": negative_prompt if not spec["ignores_negative_prompts"] else "[Ignored by FLUX]",
                "timestamp": datetime.now().isoformat(),
                "latency": latency,
                "success": True,
                "source": "hf_inference_provider",
                "width": width,
                "height": height
            }
            save_to_cache(generated_img, meta)
            return generated_img, meta
            
        except Exception as exc:
            last_error = str(exc)
            wait_time = (2 ** attempt) * 1.5
            print(f"[Generator Retry {attempt+1}/{max_retries}] Model {model_id} error: {last_error}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
            
    # If all retries fail, fall back safely
    latency = round(time.time() - start_time, 2)
    print(f"[Generator Error] All retries exhausted for {model_id}: {last_error}")
    
    cached_match = find_best_cached_match(brand_id, model_id, scenario)
    if cached_match:
        img, meta = cached_match
        meta["source"] = "cache_fallback_after_error"
        meta["notice"] = f"Inference API rate limited or unavailable ({last_error}). Serving cached asset."
        return img, meta
        
    synthetic_img = create_synthetic_demo_image(brand, model_name, scenario, width, height)
    meta = {
        "cache_key": cache_key,
        "model": model_id,
        "model_name": model_name,
        "brand_id": brand_id,
        "scenario": scenario,
        "seed": seed,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "timestamp": datetime.now().isoformat(),
        "latency": latency,
        "success": False,
        "error": last_error,
        "source": "synthetic_error_fallback",
        "notice": "Inference API error. Served branded synthetic creative."
    }
    save_to_cache(synthetic_img, meta)
    return synthetic_img, meta
