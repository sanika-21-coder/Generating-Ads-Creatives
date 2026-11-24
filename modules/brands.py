"""
Brand management module: loads, saves, updates, and deletes brand briefs.
"""

import os
import json
import re
from typing import Dict, List, Optional

BRANDS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "brands")

def slugify(text: str) -> str:
    """Generate a clean URL/filename-safe slug from brand name."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"[-\s]+", "_", text) or "custom_brand"

def get_all_brands() -> List[Dict]:
    """Retrieve all saved brand definitions sorted by name."""
    os.makedirs(BRANDS_DIR, exist_ok=True)
    brands = []
    for filename in os.listdir(BRANDS_DIR):
        if filename.endswith(".json"):
            path = os.path.join(BRANDS_DIR, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    brand_data = json.load(f)
                    if "id" not in brand_data:
                        brand_data["id"] = filename[:-5]
                    brands.append(brand_data)
            except Exception as e:
                print(f"Error loading brand from {filename}: {e}")
    return sorted(brands, key=lambda b: b.get("name", "").lower())

def get_brand_by_id(brand_id: str) -> Optional[Dict]:
    """Get single brand dictionary by its ID."""
    brands = get_all_brands()
    for b in brands:
        if b.get("id") == brand_id:
            return b
    return brands[0] if brands else None

def save_brand(brand_data: Dict) -> str:
    """Save or update brand in JSON store, returning brand ID."""
    os.makedirs(BRANDS_DIR, exist_ok=True)
    brand_id = brand_data.get("id") or slugify(brand_data.get("name", "custom_brand"))
    brand_data["id"] = brand_id
    
    # Ensure palette has properly structured dicts
    if "palette" in brand_data and isinstance(brand_data["palette"], list):
        clean_palette = []
        for c in brand_data["palette"]:
            if isinstance(c, str):
                clean_palette.append({"name": "Color", "hex": c})
            elif isinstance(c, dict):
                clean_palette.append({
                    "name": c.get("name", "Color"),
                    "hex": c.get("hex", "#000000")
                })
        brand_data["palette"] = clean_palette

    filepath = os.path.join(BRANDS_DIR, f"{brand_id}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(brand_data, f, indent=2)
    return brand_id

def delete_brand(brand_id: str) -> bool:
    """Delete a brand file if more than one brand exists."""
    brands = get_all_brands()
    if len(brands) <= 1:
        return False  # Do not delete the last remaining brand
    filepath = os.path.join(BRANDS_DIR, f"{brand_id}.json")
    if os.path.exists(filepath):
        os.remove(filepath)
        return True
    return False

def render_brand_card_html(brand: Dict, is_selected: bool = False) -> str:
    """Generate modern responsive HTML card showing swatches, mood, and description."""
    palette = brand.get("palette", [])
    swatches_html = "".join([
        f'<div style="background-color: {c.get("hex", "#999")}; width: 28px; height: 28px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.2); title: \'{c.get("name", "")} ({c.get("hex", "")})\';" title="{c.get("name", "")}: {c.get("hex", "")}"></div>'
        for c in palette[:4]
    ])
    
    mood_tags = "".join([
        f'<span style="background: rgba(99, 102, 241, 0.12); color: #4338ca; border-radius: 9999px; padding: 2px 8px; font-size: 11px; font-weight: 500; margin-right: 4px; display: inline-block; margin-bottom: 3px;">{m}</span>'
        for m in brand.get("mood", [])[:4]
    ])

    border_style = "2px solid #4f46e5; background: #f5f3ff;" if is_selected else "1px solid #e2e8f0; background: #ffffff;"

    return f"""
    <div style="{border_style} border-radius: 12px; padding: 14px; margin-bottom: 10px; transition: all 0.2s ease; box-shadow: 0 2px 4px rgba(0,0,0,0.04);">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 6px;">
            <div>
                <h4 style="margin: 0; font-size: 15px; font-weight: 700; color: #1e293b;">{brand.get('name', 'Brand')}</h4>
                <span style="font-size: 11px; color: #64748b; font-weight: 500;">{brand.get('category', 'Category')} &bull; {brand.get('product_name', 'Product')}</span>
            </div>
            <div style="display: flex; gap: 4px;">
                {swatches_html}
            </div>
        </div>
        <p style="margin: 4px 0 8px 0; font-size: 12px; color: #475569; line-height: 1.4;">{brand.get('description', '')}</p>
        <div style="display: flex; flex-wrap: wrap;">
            {mood_tags}
        </div>
    </div>
    """
