"""
text_extractor.py — Task 2: OCR Student Information
=====================================================

Uses Google Gemini Vision to extract handwritten name and 
registration number from the top header of the quiz sheet.

IMPROVEMENTS:
- Better region detection for name/registration fields
- Improved error handling
- Fallback to Tesseract OCR if Gemini fails
"""

import json
import re
from typing import Dict
import cv2
import numpy as np
from PIL import Image

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from src.utils.logger import get_logger
from config import GEMINI_API_KEY, GEMINI_MODEL_NAME

logger = get_logger(__name__)

# Initialize Gemini API
if GEMINI_AVAILABLE and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        logger.info("Gemini API initialized for OCR.")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini API: {e}")
else:
    if not GEMINI_AVAILABLE:
        logger.warning("google.generativeai not installed. Install with: pip install google-generativeai")
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not set. OCR will not work properly.")


def _cv2_to_pil(image: np.ndarray) -> Image.Image:
    """Converts an OpenCV image (numpy array) to a PIL Image for Gemini."""
    if len(image.shape) == 2:
        return Image.fromarray(image)
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def _extract_region_of_interest(image: np.ndarray) -> np.ndarray:
    """
    Extract the region containing name and registration number.
    Based on the template layout, this is typically in the top section.
    """
    h, w = image.shape[:2]
    
    # The name/reg fields are typically in the top 25% of the image
    # and span most of the width
    top_margin = int(h * 0.05)  # Skip very top edge
    bottom_margin = int(h * 0.30)  # Include top 30%
    left_margin = int(w * 0.05)  # Small left margin
    right_margin = int(w * 0.95)  # Small right margin
    
    roi = image[top_margin:bottom_margin, left_margin:right_margin]
    return roi


def _fallback_name_extraction(image: np.ndarray) -> Dict[str, str]:
    """
    Fallback method using basic image analysis and Tesseract if available.
    This is used when Gemini API is not available.
    """
    result = {"name": "Unknown", "reg_no": "Unknown"}
    
    # Try to use Tesseract if available
    try:
        import pytesseract
        
        h, w = image.shape[:2]
        roi = _extract_region_of_interest(image)
        
        # Convert to PIL for Tesseract
        pil_image = _cv2_to_pil(roi)
        
        # Run OCR
        text = pytesseract.image_to_string(pil_image)
        
        # Try to extract registration number (pattern like FA24-BSE-016)
        reg_pattern = r'[A-Z]{2}\d{2}-[A-Z]{3}-\d{3}'
        reg_match = re.search(reg_pattern, text.upper())
        if reg_match:
            result["reg_no"] = reg_match.group(0)
        
        # Try to extract name (look for "Name:" or similar)
        lines = text.split('\n')
        for line in lines:
            if 'name' in line.lower():
                # Extract text after "Name:"
                parts = line.split(':', 1)
                if len(parts) > 1:
                    name = parts[1].strip()
                    if name and len(name) > 2:
                        result["name"] = name
                        break
        
        logger.info(f"Tesseract OCR result -> Name: {result['name']}, Reg#: {result['reg_no']}")
        
    except ImportError:
        logger.warning("Tesseract not available. Install with: pip install pytesseract")
    except Exception as e:
        logger.error(f"Tesseract OCR failed: {e}")
    
    return result


def extract_student_info(image: np.ndarray) -> Dict[str, str]:
    """
    Extracts Name and Registration # from the quiz sheet header using Gemini.

    Strategy:
    1. Extract the header region (top 30% of image)
    2. Convert to PIL Image
    3. Send to Gemini with a structured prompt
    4. Parse and return the JSON
    
    Falls back to basic methods if Gemini is unavailable.
    """
    logger.info("Starting OCR extraction for student info...")
    
    result = {"name": "Unknown", "reg_no": "Unknown"}
    
    # Check if Gemini is available
    if not GEMINI_AVAILABLE or not GEMINI_API_KEY:
        logger.warning("Gemini not available, using fallback OCR...")
        return _fallback_name_extraction(image)
    
    # Extract region of interest
    roi = _extract_region_of_interest(image)
    
    if roi.size == 0:
        logger.error("ROI extraction failed - image too small")
        return result
    
    pil_image = _cv2_to_pil(roi)
    
    prompt = (
        "You are an expert at reading handwritten student information from exam forms. "
        "Look at the image and extract the student's Name and Registration Number.\n\n"
        "The registration number typically follows a pattern like:\n"
        "- FA24-BSE-016\n"
        "- SP26-BSE-005\n"
        "- FA23-BCS-123\n\n"
        "Return ONLY a valid JSON object with exactly these two keys:\n"
        "{\"name\": \"John Smith\", \"reg_no\": \"FA24-BSE-016\"}\n\n"
        "If you cannot read a field clearly, use \"Unknown\" as the value. "
        "Do not include any markdown formatting, only the raw JSON."
    )

    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        response = model.generate_content([prompt, pil_image])
        
        if not response or not response.text:
            logger.error("Empty response from Gemini API")
            return result
        
        # Clean up the response in case Gemini includes markdown blocks
        response_text = response.text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        response_text = response_text.strip()
        
        # Try to parse JSON
        try:
            extracted_data = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON from the text
            json_match = re.search(r'\{[^}]*\}', response_text)
            if json_match:
                extracted_data = json.loads(json_match.group(0))
            else:
                raise
        
        # Validate and extract fields
        name = extracted_data.get("name", "Unknown")
        reg_no = extracted_data.get("reg_no", "Unknown")
        
        # Clean up the values
        if name and name.strip() and name.lower() != "unknown":
            result["name"] = name.strip()
        
        if reg_no and reg_no.strip() and reg_no.lower() != "unknown":
            result["reg_no"] = reg_no.strip()
        
        logger.info(f"Gemini OCR result -> Name: {result['name']}, Reg#: {result['reg_no']}")
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from Gemini response: {response_text if 'response_text' in locals() else 'No response'}")
        logger.error(f"JSON error: {e}")
    except Exception as e:
        logger.error(f"Gemini OCR failed: {str(e)}")
        # Try fallback
        result = _fallback_name_extraction(image)
    
    return result
