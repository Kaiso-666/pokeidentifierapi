import os
import re
import io
import asyncio
import difflib
import urllib.request
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image, ImageEnhance, ImageDraw, ImageOps, ImageFilter
import pytesseract

app = FastAPI(title="Pokémon Detection API")

# --- Global State & Initialization ---
pokemon_dict = {}
pokemon_normalized_names = []

# Pre-compile regex patterns
alnum_regex = re.compile(r"[^a-zA-Z0-9\-\']")
normalization_regex = re.compile(r"[^a-z0-9]")

def sanitize_text(text: str) -> str:
    """
    Cleans incoming text by throwing away weird whitespaces and non-whitelisted characters.
    Only allows: letters, numbers, spaces, single quotes, and hyphens.
    """
    if not text:
        return ""
    # Convert all odd whitespaces, tabs, or newlines into standard spaces
    text = re.sub(r"\s+", " ", text)
    # Strip everything except A-Z, a-z, 0-9, hyphen, single quote, and normal spaces
    text = re.sub(r"[^a-zA-Z0-9\-\' ]", "", text)
    # Collapse multiple spaces down to a single space and strip borders
    return re.sub(r" +", " ", text).strip()

def normalize_name(text: str) -> str:
    return normalization_regex.sub("", text.lower())

def download_image_from_url(url: str) -> bytes:
    """
    Synchronously fetches raw image data from a URL with a fallback User-Agent header.
    """
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    )
    with urllib.request.urlopen(req, timeout=10) as response:
        return response.read()

@app.on_event("startup")
def load_pokemon_database():
    global pokemon_dict, pokemon_normalized_names
    file_path = "pokenames.txt"
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                name = sanitize_text(line)
                if name:
                    normalized = normalize_name(name)
                    pokemon_dict[normalized] = name
        pokemon_normalized_names = list(pokemon_dict.keys())
        print(f"[*] Loaded {len(pokemon_dict)} Pokémon names into memory.")
    else:
        print(f"[!] Warning: {file_path} not found. Matching will fail until added.")

# --- Core Detection Logic ---
def get_best_match(extracted_text: str) -> Optional[str]:
    if not extracted_text or not pokemon_dict:
        return None
        
    entire_block_norm = normalize_name(extracted_text)
    if entire_block_norm in pokemon_dict:
        return pokemon_dict[entire_block_norm]
        
    words = extracted_text.split()
    for word in words:
        word_norm = normalize_name(word)
        if word_norm in pokemon_dict:
            return pokemon_dict[word_norm]
            
    matches = difflib.get_close_matches(entire_block_norm, pokemon_normalized_names, n=1, cutoff=0.75)
    if matches:
        return pokemon_dict[matches[0]]
        
    return None

def process_image_ocr(image_bytes: bytes) -> Optional[str]:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
            bg = Image.new('RGB', image.size, (255, 255, 255))
            bg.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
            image = bg
        
        draw = ImageDraw.Draw(image)
        draw.rectangle([image.width - 80, 0, image.width, 75], fill=(255, 255, 255))
        
        if image.width < 900:
            image = image.resize((image.width * 3, image.height * 3), Image.Resampling.LANCZOS)
            
        image = image.convert('L') 
        image = ImageOps.autocontrast(image)
        image = image.filter(ImageFilter.SHARPEN)
        
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(3.0) 
        
        whitelist = r"-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-"
        
        config_pass_1 = f"{whitelist} --psm 7"
        raw_ocr_text = pytesseract.image_to_string(image, config=config_pass_1)
        
        if not raw_ocr_text.strip():
            config_pass_2 = f"{whitelist} --psm 11"
            raw_ocr_text = pytesseract.image_to_string(image, config=config_pass_2)
            
        # Sanitize raw OCR characters before processing chunks
        sanitized_ocr = sanitize_text(raw_ocr_text)
        words = sanitized_ocr.split()
        
        cleaned_parts = []
        for word in words:
            if (word.startswith('<') and word.endswith('>')) or (word.startswith(':') and word.endswith(':')):
                continue
            
            clean = alnum_regex.sub('', word)
            if len(clean) >= 3:
                cleaned_parts.append(clean)
                
        return " ".join(cleaned_parts[:3]) if cleaned_parts else None
    except Exception as e:
        print(f"[!] Error processing image: {e}")
        return None

# --- API Endpoint ---
@app.post("/predict")
async def predict_pokemon(
    text: Optional[str] = Form(None), 
    image: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None)
):
    # Determine if standard 'text' argument is carrying an explicit URL string instead
    text_has_link = False
    if text and ("http://" in text or "https://" in text):
        text_has_link = True
        if not image_url:
            image_url = text.strip()

    # 1. Pipeline Priority 1: Text-based explicit match (Skip if flagged as a URL)
    if text and not text_has_link:
        content = sanitize_text(text)
        text_valid = True
        
        if len(content) > 72:
            text_valid = False

        if text_valid and content:
            best_match = get_best_match(content)
            if best_match:
                return {"pokemon": best_match, "source": "text"}

    # 2. Pipeline Priority 2: OCR Extraction via File Upload or Downloadable Link
    image_bytes = None

    if image:
        image_bytes = await image.read()
    elif image_url:
        try:
            # Safely hand down blocking download logic into low-level thread worker pools
            image_bytes = await asyncio.to_thread(download_image_from_url, image_url)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch image link resource: {str(e)}")

    if image_bytes:
        extracted_word = await asyncio.to_thread(process_image_ocr, image_bytes)
        
        if extracted_word:
            extracted_word = sanitize_text(extracted_word)
            if extracted_word:
                best_match = get_best_match(extracted_word)
                if best_match:
                    return {"pokemon": best_match, "source": "ocr"}

    raise HTTPException(status_code=404, detail="Pokémon could not be identified.")
