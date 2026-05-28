"""
text_extractor.py — Task 2: OCR Student Information
=====================================================

Uses Google Gemini Vision to extract handwritten name and 
registration number from the top header of the quiz sheet.
"""

import json
from typing import Dict
import cv2
import numpy as np
from PIL import Image
import google.generativeai as genai

from src.utils.logger import get_logger
from config import GEMINI_API_KEY, GEMINI_MODEL_NAME

logger = get_logger(__name__)

if GEMINI_API_KEY:
    logger.info("Initializing Gemini API for OCR...")
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logger.warning("GEMINI_API_KEY is not set. OCR will fail.")

def _cv2_to_pil(image: np.ndarray) -> Image.Image:
    """Converts an OpenCV image (numpy array) to a PIL Image for Gemini."""
    if len(image.shape) == 2:
        return Image.fromarray(image)
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def extract_student_info(image: np.ndarray) -> Dict[str, str]:
    """
    Extracts Name and Registration # from the quiz sheet header using Gemini.

    Strategy:
    1. Crop the top 20% of the image (the header row).
    2. Convert to PIL Image.
    3. Send to Gemini with a prompt expecting a JSON response.
    4. Parse and return the JSON.
    """
    logger.info("Starting Gemini OCR extraction for student info...")
    
    result = {"name": "Unknown", "reg_no": "Unknown"}

    if not GEMINI_API_KEY:
        logger.error("Cannot run OCR without GEMINI_API_KEY.")
        return result

    # Crop the top 20% of the image to just get the header
    h, w = image.shape[:2]
    header_crop = image[0:int(h * 0.20), 0:w]
    
    pil_image = _cv2_to_pil(header_crop)
    
    prompt = (
        "You are an expert at reading handwritten student information from exam forms. "
        "Extract the student's Name and Registration Number (or Reg #) from the provided image crop. "
        "The registration number usually looks like 'FA24-ISSE-016', 'SP26-BSE-005', etc. "
        "Return ONLY a valid JSON object with the keys 'name' and 'reg_no'. "
        "If you cannot read a field clearly, or it is missing, set its value to 'Unknown'. "
        "Do not include any markdown formatting, just the raw JSON object."
    )

    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        response = model.generate_content([prompt, pil_image])
        
        # Clean up the response in case Gemini includes markdown blocks like ```json ... ```
        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
            
        response_text = response_text.strip()
        
        extracted_data = json.loads(response_text)
        
        # Merge with default result
        result["name"] = extracted_data.get("name", "Unknown")
        result["reg_no"] = extracted_data.get("reg_no", "Unknown")
        
        logger.info(f"Gemini OCR result → Name: {result['name']},  Reg#: {result['reg_no']}")
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from Gemini response: {response.text if 'response' in locals() else 'No response'}")
    except Exception as e:
        logger.error(f"Gemini OCR failed: {str(e)}")

    return result