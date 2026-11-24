"""
Batch generation script for AdForge.
Fills the evaluation cache with 36 core images (3 brands x 6 scenarios x 2 models).
Resumes automatically if interrupted and skips already cached assets.
"""

import os
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.models import MODEL_A_ID, MODEL_B_ID, SCENARIO_TEMPLATES
from modules.brands import get_all_brands
from modules.prompt_builder import build_prompt, build_negative_prompt
from modules.generator import generate_image
from modules.cache import compute_cache_key, get_cached_item
from app import load_analytics_dashboard

def run_batch_evaluation():
    brands = get_all_brands()
    models = [MODEL_A_ID, MODEL_B_ID]
    scenarios = SCENARIO_TEMPLATES
    
    total_target = len(brands) * len(scenarios) * len(models)
    print("=" * 65)
    print(f"🚀 AdForge Batch Evaluation Generator")
    print(f"Target Matrix: {len(brands)} Brands × {len(scenarios)} Scenarios × {len(models)} Models = {total_target} Creatives")
    print("=" * 65)
    
    seed = 42
    generated_count = 0
    skipped_count = 0
    
    for b_idx, brand in enumerate(brands, 1):
        brand_id = brand["id"]
        brand_name = brand["name"]
        print(f"\n📂 [{b_idx}/{len(brands)}] Brand: {brand_name} ({brand_id})")
        
        for s_idx, scenario in enumerate(scenarios, 1):
            prompt = build_prompt(brand, scenario=scenario, format_name="Instagram Square (1:1)", use_style_block=True)
            neg_prompt = build_negative_prompt(brand, use_negative_prompt=True)
            
            for m_idx, model_id in enumerate(models, 1):
                model_label = "FLUX.1-schnell" if "flux" in model_id.lower() else "SDXL Base 1.0"
                cache_key = compute_cache_key(model_id, brand_id, scenario, seed, prompt[:50])
                
                # Check resume/cache status
                existing = get_cached_item(cache_key)
                if existing:
                    skipped_count += 1
                    print(f"  ⏩ [Cached] {model_label} | {scenario} ({skipped_count + generated_count}/{total_target})")
                    continue
                    
                print(f"  🎨 Generating: {model_label} | {scenario}...", end=" ", flush=True)
                t0 = time.time()
                img, meta = generate_image(
                    model_id=model_id,
                    prompt=prompt,
                    negative_prompt=neg_prompt,
                    seed=seed,
                    brand=brand,
                    scenario=scenario,
                    format_name="Instagram Square (1:1)",
                    demo_mode=False
                )
                dur = round(time.time() - t0, 2)
                generated_count += 1
                src = meta.get("source", "generated")
                print(f"Done ({dur}s, source={src})")
                
    print("\n" + "=" * 65)
    print(f"🎉 Batch Generation Complete!")
    print(f"Summary: {generated_count} newly generated, {skipped_count} loaded from cache. Total: {generated_count + skipped_count}/{total_target}")
    print("=" * 65)
    
    # Compute and persist jury benchmark metrics
    print("\n📊 Computing comprehensive 108-judgment jury benchmark analytics...")
    load_analytics_dashboard()
    print("✅ Benchmark metrics and CSV successfully saved to data/results/model_comparison_metrics.csv")

if __name__ == "__main__":
    run_batch_evaluation()
