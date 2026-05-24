"""
text_extractor.py — Task 2: OCR Student Information
=====================================================

Calibrated for: 1240x1755 scanned quiz sheet
Name field:  y=13-17% of height, x=80-400px  (left of centre)
Reg# field:  y=13-17% of height, x=490-870px (right of centre, before QR)
"""

from typing import Dict
import cv2
import numpy as np
import re
import easyocr

from src.utils.logger import get_logger
from config import OCR_LANGUAGES, OCR_USE_GPU

logger = get_logger(__name__)

logger.info("Initializing EasyOCR Model...")
READER = easyocr.Reader(OCR_LANGUAGES, gpu=OCR_USE_GPU)
logger.info("EasyOCR Model loaded.")

# Template dimensions the crops are calibrated for
TEMPLATE_W = 1240
TEMPLATE_H = 1755

# Crop regions as fractions of image size (measured from blank scan)
NAME_Y1, NAME_Y2 = 0.155, 0.180   # row containing "Name: ___"
NAME_X1, NAME_X2 = 0.190, 0.330   # just the handwritten name area (after "Name:" label)

REG_Y1,  REG_Y2  = 0.155, 0.180   # same row
REG_X1,  REG_X2  = 0.530, 0.750   # just the handwritten reg# area (after "Registration #" label)


def _upscale(crop: np.ndarray) -> np.ndarray:
    """2× upscale + sharpen for better OCR on small handwriting."""
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()
    h, w = gray.shape
    up = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(up, -1, kernel)


def extract_student_info(image: np.ndarray) -> Dict[str, str]:
    """
    Extracts Name and Registration # from the quiz sheet header.

    Strategy:
    1. Crop the EXACT region where the handwritten name lives.
    2. Crop the EXACT region where the handwritten reg# lives.
    3. Run EasyOCR on each crop independently.
    4. Clean up the result with regex.
    """
    logger.info("Starting OCR extraction for student info...")

    h, w = image.shape[:2]
    result = {"name": "Unknown", "reg_no": "Unknown"}

    # ── NAME crop ──────────────────────────────────────────────
    ny1, ny2 = int(h * NAME_Y1), int(h * NAME_Y2)
    nx1, nx2 = int(w * NAME_X1), int(w * NAME_X2)
    name_crop = _upscale(image[ny1:ny2, nx1:nx2])

    name_texts = READER.readtext(name_crop, detail=0, paragraph=True)
    raw_name = " ".join(name_texts).strip()
    logger.debug(f"Name crop OCR raw: '{raw_name}'")

    # Remove any leftover "Name:" prefix OCR might have caught
    raw_name = re.sub(r'^[Nn]ame\s*[:#]?\s*', '', raw_name).strip()
    if raw_name and len(raw_name) > 1:
        result["name"] = raw_name

    # ── REG# crop ──────────────────────────────────────────────
    ry1, ry2 = int(h * REG_Y1),  int(h * REG_Y2)
    rx1, rx2 = int(w * REG_X1),  int(w * REG_X2)
    reg_crop = _upscale(image[ry1:ry2, rx1:rx2])

    reg_texts = READER.readtext(reg_crop, detail=0, paragraph=True)
    raw_reg = " ".join(reg_texts).strip()
    logger.debug(f"Reg crop OCR raw: '{raw_reg}'")

    # Try strict pattern first: FA24-ISSE-016 style
    m = re.search(r'([A-Z]{2}\d{2}[-_][A-Z]{2,6}[-_][A-Za-z0-9]{2,6})', raw_reg)
    if m:
        result["reg_no"] = m.group(1)
    else:
        # Looser: anything with letters-digits-dash that looks like an ID
        raw_reg_clean = re.sub(r'^[Rr]eg.*?[#:]\s*', '', raw_reg).strip()
        if raw_reg_clean and len(raw_reg_clean) > 3:
            result["reg_no"] = raw_reg_clean

    # ── Fallback: scan the full header row ─────────────────────
    if result["name"] == "Unknown" or result["reg_no"] == "Unknown":
        logger.info("Targeted crops failed, scanning full header row...")
        full_y1, full_y2 = int(h * 0.155), int(h * 0.185)
        full_crop = _upscale(image[full_y1:full_y2, 0:int(w * 0.75)])
        full_results = READER.readtext(full_crop, detail=1)

        for _, text, _ in full_results:
            t = text.strip()
            if result["name"] == "Unknown":
                nm = re.search(r'[Nn]ame\s*[:#]?\s*([A-Za-z][A-Za-z .]{2,})', t)
                if nm:
                    result["name"] = nm.group(1).strip()

            if result["reg_no"] == "Unknown":
                rm = re.search(r'([A-Z]{2}\d{2}[-_][A-Z]{2,6}[-_][A-Za-z0-9]{2,6})', t)
                if rm:
                    result["reg_no"] = rm.group(1)

    logger.info(f"OCR result → Name: {result['name']},  Reg#: {result['reg_no']}")
    return result