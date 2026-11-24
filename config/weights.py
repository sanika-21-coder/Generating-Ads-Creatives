"""
Configuration for evaluation scoring and composite Brand Fit Score.
"""

# Scoring weights (must sum to 1.0)
WEIGHTS = {
    "prompt_alignment": 0.35,      # CLIP image-text similarity
    "palette_adherence": 0.35,     # CIEDE2000 Delta E in CIELAB
    "forbidden_check": 0.15,       # CLIP penalty for elements to avoid
    "visual_consistency": 0.15,    # Style block compliance and embedding similarity
}

# Traffic light badge thresholds (0-100 scale)
THRESHOLDS = {
    "great": 75.0,
    "good": 50.0,
    "needs_work": 0.0
}

def get_traffic_light_badge(score: float) -> tuple[str, str]:
    """Returns (label, color_hex) based on score."""
    if score >= THRESHOLDS["great"]:
        return "Great", "#22c55e"       # Green
    elif score >= THRESHOLDS["good"]:
        return "Good", "#eab308"        # Yellow / Amber
    else:
        return "Needs Work", "#ef4444"  # Red
