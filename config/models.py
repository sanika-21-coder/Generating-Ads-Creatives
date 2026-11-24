"""
Configuration for AdForge models.
Change model IDs here in one place if API limits or availability changes.
"""

# Primary comparison models
MODEL_A_ID = "black-forest-labs/FLUX.1-schnell"
MODEL_B_ID = "stabilityai/stable-diffusion-xl-base-1.0"

# Optional prompt improvement model (hosted free inference provider)
TEXT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# Local CPU scoring model for brand alignment and multi-modal consistency
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

# Detailed model capabilities and constraints
MODEL_SPECS = {
    MODEL_A_ID: {
        "name": "FLUX.1-schnell",
        "key": "flux",
        "description": "Black Forest Labs 12B rectified flow transformer distilled for rapid generation.",
        "ignores_negative_prompts": True,
        "default_width": 768,
        "default_height": 768,
        "timeout_seconds": 50,
        "max_retries": 3,
        "free_tier_credit_efficient": True,
    },
    MODEL_B_ID: {
        "name": "SDXL Base 1.0",
        "key": "sdxl",
        "description": "Stability AI 3.5B parameter latent diffusion model with full negative prompt guidance.",
        "ignores_negative_prompts": False,
        "default_width": 768,
        "default_height": 768,
        "timeout_seconds": 60,
        "max_retries": 3,
        "free_tier_credit_efficient": True,
    }
}

# Aspect ratios mapped to formats
FORMAT_ASPECT_RATIOS = {
    "Instagram Square (1:1)": {"width": 768, "height": 768, "ratio_label": "1:1"},
    "Story / Reel (9:16)": {"width": 576, "height": 1024, "ratio_label": "9:16"},
    "Landscape Banner (16:9)": {"width": 1024, "height": 576, "ratio_label": "16:9"},
    "Portrait Poster (4:5)": {"width": 640, "height": 800, "ratio_label": "4:5"},
    "Festive Leaderboard (12:5)": {"width": 960, "height": 400, "ratio_label": "12:5"},
}

# Standard scenarios
SCENARIO_TEMPLATES = [
    "Hero Product Shot",
    "Lifestyle in Context",
    "Festive & Celebration",
    "Minimal Studio Cyclorama",
    "Outdoor Natural Light",
    "Macro Detail Close-Up"
]
