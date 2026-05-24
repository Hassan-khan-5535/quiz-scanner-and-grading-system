"""
qr_reader.py — Task 1: QR Code Decoding
=========================================

PURPOSE:
    Detects the QR code in the quiz image, decodes the text payload,
    and parses it into a structured dictionary (AnswerKey).

ASSIGNMENT REQUIREMENTS MET:
    - Locate QR code anywhere on the page
    - Decode the payload
    - Parse into: Quiz set, Part-I answers, Part-II answers

IMPROVEMENTS:
    - Multiple preprocessing strategies to handle real-world photos
      (angled shots, shadows, partial crops, low contrast)
    - Crops & upscales the top-right region where QR is expected
    - Falls back gracefully with clear error messages
"""

from typing import Optional, Dict, Any, List
from pyzbar.pyzbar import decode
import numpy as np
import cv2

from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_qr_payload(payload: str) -> Optional[Dict[str, Any]]:
    """
    Parses the raw text from the QR code into a structured dictionary.
    
    EXPECTED FORMAT (from assignment):
    "AI Quiz SP2026 Set-C | Part-I: Q1=D Q2=A Q3=B Q4=A Q5=D Q6=A Q7=A Q8=B | Part-II: Q1=C Q2=D Q3=D Q4=D Q5=C Q6=C Q7=C Q8=B"
    
    Args:
        payload: The raw string decoded from the QR code.
        
    Returns:
        Dictionary representing the AnswerKey, or None if parsing fails.
    """
    try:
        # Split the payload into its 3 main sections using the "|" separator
        sections = [s.strip() for s in payload.split("|")]
        
        if len(sections) != 3:
            logger.error(f"QR payload does not have 3 sections. Found: {len(sections)}")
            return None
            
        header_section = sections[0]
        part1_section = sections[1]
        part2_section = sections[2]
        
        # 1. Parse Header (e.g., "AI Quiz SP2026 Set-C")
        header_parts = header_section.split()
        quiz_title = " ".join(header_parts[:-1])
        
        set_string = header_parts[-1]  # "Set-C"
        set_id = set_string.split("-")[1] if "-" in set_string else set_string
        
        def parse_answers(section_text: str) -> Dict[str, str]:
            """Converts 'Part-I: Q1=D Q2=A' -> {'Q01': 'D', 'Q02': 'A'}"""
            content = section_text.split(":", 1)[1].strip()
            pairs = content.split()
            answers = {}
            for pair in pairs:
                if "=" in pair:
                    q, a = pair.split("=")
                    q_num = q.replace("Q", "")
                    standardized_q = f"Q{q_num.zfill(2)}"
                    answers[standardized_q] = a
            return answers
            
        part1_answers = parse_answers(part1_section)
        part2_answers = parse_answers(part2_section)
        
        answer_key = {
            "quiz_title": quiz_title,
            "set": set_id,
            "part1": part1_answers,
            "part2": part2_answers
        }
        
        logger.info(f"Successfully parsed answer key for Set {set_id}")
        return answer_key
        
    except Exception as e:
        logger.error(f"Failed to parse QR payload: {str(e)}")
        logger.error(f"Raw payload was: {payload}")
        return None


def _try_decode(image: np.ndarray) -> Optional[str]:
    """
    Attempt to decode a QR code from a single image variant.
    Returns the raw payload string if found, else None.
    """
    results = decode(image)
    if results:
        return results[0].data.decode('utf-8')
    return None


def _get_preprocessed_variants(image: np.ndarray) -> List[np.ndarray]:
    """
    Generates multiple preprocessed versions of the image to maximise
    the chance of pyzbar finding the QR code in a real-world photo.
    
    Strategies:
      1. Original grayscale (baseline)
      2. Upscaled 2x (helps with small/far QR codes)
      3. Adaptive threshold (handles uneven lighting/shadows)
      4. Top-right crop + upscale (QR is usually in that corner)
      5. Top-right crop + adaptive threshold
      6. Sharpened image
      7. Inverted image (handles dark backgrounds)
    """
    variants = []

    # Ensure grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape

    # 1. Original grayscale
    variants.append(("original_gray", gray))

    # 2. Upscaled 2x
    upscaled = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    variants.append(("upscaled_2x", upscaled))

    # 3. Adaptive threshold on full image
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    variants.append(("adaptive_thresh", adaptive))

    # 4. Top-right crop (QR code location on this quiz sheet)
    #    Take top 25% height, right 25% width — with a little padding
    crop_top = gray[0: int(h * 0.28), int(w * 0.68):]
    if crop_top.size > 0:
        crop_top_big = cv2.resize(
            crop_top,
            (crop_top.shape[1] * 4, crop_top.shape[0] * 4),
            interpolation=cv2.INTER_CUBIC
        )
        variants.append(("top_right_crop_4x", crop_top_big))

        # 5. Top-right crop + adaptive threshold
        crop_adaptive = cv2.adaptiveThreshold(
            crop_top_big, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        variants.append(("top_right_crop_adaptive", crop_adaptive))

        # Extra: Otsu threshold on crop
        _, crop_otsu = cv2.threshold(
            crop_top_big, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        variants.append(("top_right_crop_otsu", crop_otsu))

    # 6. Sharpen full image
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    variants.append(("sharpened", sharpened))

    # 7. Inverted (dark background QR codes)
    inverted = cv2.bitwise_not(gray)
    variants.append(("inverted", inverted))

    # 8. Otsu threshold on full image
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("otsu_thresh", otsu))

    return variants


def decode_answer_key(image: np.ndarray) -> Optional[Dict[str, Any]]:
    """
    Main function for Task 1. Detects and decodes the QR code.

    Uses multiple preprocessing strategies so it works on real-world
    photos (angled, shadowed, partially cropped QR codes).
    
    Args:
        image: The image of the quiz sheet (grayscale or BGR).
        
    Returns:
        The structured AnswerKey dictionary, or None if no QR code is found.
    """
    logger.info("Scanning for QR code...")

    variants = _get_preprocessed_variants(image)

    for name, variant in variants:
        logger.info(f"Trying QR decode strategy: {name}")
        raw_payload = _try_decode(variant)
        if raw_payload:
            logger.info(f"QR code found using strategy: '{name}'")
            logger.info(f"Raw payload: {raw_payload[:60]}...")
            return parse_qr_payload(raw_payload)

    logger.error(
        "No QR code found after trying all preprocessing strategies. "
        "Tips: ensure the QR code is fully visible and not cut off at the edges, "
        "use better lighting, and hold the camera flat above the sheet."
    )
    return None