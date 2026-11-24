"""
Labeling, Watermarking, and Portfolio module for AdForge.
Burns a compliant 'AI Generated' disclosure badge onto exported creatives,
manages portfolio gallery storage, and provides regulatory disclosure documentation.
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Tuple
from PIL import Image, ImageDraw, ImageFont

PORTFOLIO_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "results", "portfolio.json")
PORTFOLIO_IMG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "results", "portfolio_images")

def apply_ai_disclosure_badge(
    image: Image.Image,
    brand_name: str = "Brand",
    model_name: str = "Diffusion Model"
) -> Image.Image:
    """
    Burns a permanent, elegant semi-transparent disclosure badge into the image
    complying with EU AI Act Art. 50 and ASCI transparency directives.
    """
    watermarked = image.copy().convert("RGBA")
    w, h = watermarked.size
    
    # Create overlay
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    badge_text = f"✨ AI GENERATED • {brand_name} • AdForge"
    
    # Calculate badge dimensions
    badge_w = min(int(w * 0.45), 320)
    badge_h = 32
    padding = 16
    x1 = w - badge_w - padding
    y1 = h - badge_h - padding
    x2 = w - padding
    y2 = h - padding
    
    # Pill background with semi-transparent dark backdrop
    draw.rounded_rectangle(
        [x1, y1, x2, y2],
        radius=14,
        fill=(15, 23, 42, 190),  # Slate 900 with ~75% opacity
        outline=(255, 255, 255, 120),
        width=1
    )
    
    # Text placement
    draw.text((x1 + 14, y1 + 8), badge_text, fill=(255, 255, 255, 240))
    
    # Alpha composite
    combined = Image.alpha_composite(watermarked, overlay)
    return combined.convert("RGB")

def save_to_portfolio(
    image: Image.Image,
    brand: Dict,
    model_name: str,
    prompt: str,
    compliance_score: float,
    scenario: str
) -> Tuple[bool, str]:
    """Saves watermarked creative and metadata to persistent portfolio."""
    os.makedirs(PORTFOLIO_IMG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(PORTFOLIO_FILE), exist_ok=True)
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    item_id = f"creative_{timestamp_str}_{brand.get('id', 'brand')}"
    
    watermarked = apply_ai_disclosure_badge(image, brand.get("name", "Brand"), model_name)
    img_filename = f"{item_id}.png"
    img_filepath = os.path.join(PORTFOLIO_IMG_DIR, img_filename)
    watermarked.save(img_filepath, "PNG")
    
    # Load existing portfolio
    portfolio = []
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
                portfolio = json.load(f)
        except Exception:
            portfolio = []
            
    record = {
        "id": item_id,
        "image_path": img_filepath,
        "filename": img_filename,
        "brand_name": brand.get("name", "Brand"),
        "brand_id": brand.get("id", "brand"),
        "product": brand.get("product_name", "Product"),
        "model_name": model_name,
        "scenario": scenario,
        "prompt": prompt,
        "compliance_score": compliance_score,
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    portfolio.insert(0, record)
    with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
        json.dump(portfolio, f, indent=2)
        
    return True, f"Saved to portfolio! Saved as {img_filename} with mandatory disclosure badge."

def get_portfolio_items() -> List[Dict]:
    """Retrieves all saved portfolio creatives."""
    if not os.path.exists(PORTFOLIO_FILE):
        return []
    try:
        with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
            items = json.load(f)
            # Filter for files that exist
            return [it for it in items if os.path.exists(it.get("image_path", ""))]
    except Exception:
        return []

def get_ethics_regulatory_note() -> str:
    """Returns authoritative legal and regulatory synthesis for AI in advertising."""
    return """
### ⚖️ Regulatory Compliance & Ethics in Generative Advertising

Advertising generated through diffusion models is subject to emerging global transparency mandates:

1. **European Union AI Act (Article 50 - Transparency Obligations)**:
   - Providers and deployers of AI systems that generate synthetic audio, image, video or text content must ensure that outputs are clearly marked in a machine-detectable and visibly human-perceptible format as **artificially generated or manipulated**.
   - AdForge natively burns a verifiable `✨ AI GENERATED • AdForge` disclosure pill directly into the raster image buffer upon export to prevent deceptive commercial representation.

2. **Advertising Standards Council of India (ASCI Guidelines on Virtual/AI Endorsers & Ads)**:
   - ASCI requires advertisers to prominently disclose that content is artificially generated (e.g., `#AIGenerated` or conspicuous visual disclosures) so consumers are not misled regarding the genuine physical properties, packaging, or claims of the advertised product.
   - Consumers must never be led to believe an unverified synthetic rendering represents genuine photographic evidence of real-world product performance.

3. **Brand Safety & Intellectual Property Precautions**:
   - Diffusion models can inadvertently reproduce trademarked motifs or copyrighted artist styles. AdForge utilizes parameterized **Style Blocks** and strict **Avoidance Negatives** to confine generation to original brand design assets.
"""
