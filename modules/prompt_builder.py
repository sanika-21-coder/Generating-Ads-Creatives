"""
Template-based Prompt Builder for Brand Consistency.
Builds positive and negative prompts adhering to brand visual identity guidelines,
with support for aspect ratios, scenarios, fixed brand style blocks, and refinement tweaks.
"""

import os
from typing import Dict, List, Optional
from config.models import TEXT_MODEL_ID, FORMAT_ASPECT_RATIOS

# Baseline photography and artifact rejection terms
DEFAULT_NEGATIVE_TERMS = [
    "blurry",
    "low quality",
    "distorted text",
    "unreadable lettering",
    "watermark",
    "logo artifacts",
    "jpeg compression artifacts",
    "oversaturated noise",
    "extra fingers",
    "deformed anatomy",
    "mutilated",
    "poorly rendered hands",
    "cropped frame"
]

SCENARIO_DESCRIPTIONS = {
    "Hero Product Shot": "hero center-stage commercial photography, high-end studio lighting, sharp focus on primary packaging",
    "Lifestyle in Context": "authentic editorial lifestyle setting, elegant ambient environmental context, aspirational setting",
    "Festive & Celebration": "festive seasonal celebration, sparkling ambient highlights, celebratory holiday atmosphere",
    "Minimal Studio Cyclorama": "minimalist seamless cyclorama studio backdrop, clean architectural shadows, pure elegance",
    "Outdoor Natural Light": "sun-drenched outdoor golden hour lighting, lush organic environmental depth, soft breeze",
    "Macro Detail Close-Up": "extreme macro product photography, tactile material texture, condensation and embossed details, shallow depth of field"
}

FORMAT_COMPOSITION_HINTS = {
    "Instagram Square (1:1)": "balanced 1:1 square composition, centered focal subject, generous negative space for typography overlay",
    "Story / Reel (9:16)": "vertical 9:16 portrait composition, dynamic top-to-bottom visual hierarchy, ample vertical clearance",
    "Landscape Banner (16:9)": "cinematic 16:9 wide banner composition, off-center third rule placement, clear copy negative space",
    "Portrait Poster (4:5)": "editorial 4:5 vertical portrait composition, striking magazine cover art direction, high visual impact",
    "Festive Leaderboard (12:5)": "ultra-wide 12:5 banner layout, panoramic spatial staging, seamless horizontal gradient"
}

TWEAK_MODIFIERS = {
    "more vibrant": "intensely vibrant colors, rich saturated tones, radiant chromatic depth, punchy dynamic lighting",
    "simpler background": "ultra-clean monochromatic cyclorama background, zero distractions, pure minimalist negative space",
    "more premium": "ultra-luxury prestige aesthetic, brushed metallic and velvet specular highlights, elite high-fashion art direction",
    "brighter": "high-key bright commercial illumination, luminous airy atmosphere, crystal-clear soft shadows",
    "closer product shot": "extreme macro close-up view, microscopic tactile surface detail, beautiful bokeh blur"
}

def build_palette_string(palette: List[Dict]) -> str:
    """Formats palette list into descriptive color terms and hex codes."""
    if not palette:
        return "neutral brand colors"
    color_phrases = []
    for c in palette:
        name = c.get("name", "Color")
        hex_val = c.get("hex", "")
        if hex_val:
            color_phrases.append(f"{name} ({hex_val})")
        else:
            color_phrases.append(name)
    return ", ".join(color_phrases)

def build_prompt(
    brand: Dict,
    format_name: str = "Instagram Square (1:1)",
    scenario: str = "Hero Product Shot",
    extra_idea: str = "",
    use_style_block: bool = True,
    active_tweak: Optional[str] = None
) -> str:
    """
    Assembles a brand-consistent positive prompt using structured templates.
    No LLM required; entirely deterministic and reproducible.
    """
    product = brand.get("product_name", "product")
    brand_name = brand.get("name", "Brand")
    mood_list = brand.get("mood", [])
    mood_str = ", ".join(mood_list) if mood_list else "premium"
    palette_str = build_palette_string(brand.get("palette", []))
    
    scenario_desc = SCENARIO_DESCRIPTIONS.get(scenario, scenario)
    format_hint = FORMAT_COMPOSITION_HINTS.get(format_name, "commercial advertising composition")
    
    # 1. Subject & Product
    prompt_parts = [
        f"Commercial advertisement photography for {brand_name}'s {product}",
        scenario_desc
    ]
    
    # Extra user creative idea
    if extra_idea and extra_idea.strip():
        prompt_parts.append(f"featuring {extra_idea.strip()}")
        
    # Mood & Palette
    prompt_parts.append(f"evoking a {mood_str} mood")
    prompt_parts.append(f"adhering to the official brand color palette of {palette_str}")
    
    # Camera, composition, format
    prompt_parts.append(f"{format_hint}, Hasselblad 100mm f/2.8 macro lens, volumetric ray-traced lighting, 8k resolution commercial print quality")
    
    # Style block (Brand Visual DNA consistency technique)
    if use_style_block:
        style_block = brand.get("style_block", "")
        if style_block:
            prompt_parts.append(f"Brand Visual DNA: {style_block}")
            
    # One-click tweak modifier
    if active_tweak and active_tweak.lower() in TWEAK_MODIFIERS:
        prompt_parts.append(TWEAK_MODIFIERS[active_tweak.lower()])
        
    prompt_parts.append("No text, no watermarks, no distorted typography")
    
    return ". ".join(prompt_parts) + "."

def build_negative_prompt(brand: Dict, use_negative_prompt: bool = True) -> str:
    """
    Constructs a robust negative prompt combining brand 'things to avoid'
    and baseline photography defect suppressions.
    """
    if not use_negative_prompt:
        return ""
        
    negatives = list(DEFAULT_NEGATIVE_TERMS)
    
    # Add brand-specific avoidances
    avoid_items = brand.get("things_to_avoid", [])
    for item in avoid_items:
        clean_item = item.strip().lower()
        if clean_item and clean_item not in negatives:
            negatives.insert(0, clean_item)
            
    return ", ".join(negatives)

def improve_prompt_with_ai(original_prompt: str, brand: Dict) -> str:
    """
    Optional prompt polisher using hosted Hugging Face Inference API.
    Silently falls back to original prompt on failure or missing token.
    """
    token = os.environ.get("HF_TOKEN")
    if not token or not token.strip():
        return original_prompt
        
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(token=token.strip())
        
        system_instruction = (
            "You are an elite creative advertising director. Polish the following diffusion prompt "
            "to make it more photorealistic, visually captivating, and brand-coherent. "
            "Keep the exact brand colors, product, and style cues intact. Do not add conversational text. "
            "Output ONLY the revised prompt text."
        )
        
        response = client.chat.completions.create(
            model=TEXT_MODEL_ID,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Polish this ad prompt:\n{original_prompt}"}
            ],
            max_tokens=250,
            temperature=0.7
        )
        
        polished = response.choices[0].message.content.strip()
        if len(polished) > 30:
            return polished
    except Exception as e:
        print(f"[Prompt AI Polisher] Fallback to template due to: {e}")
        
    return original_prompt
