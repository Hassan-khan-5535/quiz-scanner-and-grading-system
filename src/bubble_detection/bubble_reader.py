"""
bubble_reader.py — Task 3: Bubble Detection & Answer Extraction
================================================================

APPROACH: Coordinate-based (not contour-based)
Instead of trying to "find" bubbles via contour detection (which fails on
real photos with noise), we use KNOWN pixel coordinates from the blank
scanned template. This is robust, fast, and accurate.

Calibrated for: 1240x1755 scanned quiz sheet (CamScanner output)
Grid: x=177-1072, y=340-662
Part-I  bubble X centres (A,B,C,D): 289, 369, 449, 535  (in image coords)
Part-II bubble X centres (A,B,C,D): 721, 799, 877, 961  (in image coords)
Row Y centres (Q01-Q08):            398, 432, 466, 500, 534, 568, 604, 638
"""

from typing import Dict, Any, Tuple
import cv2
import numpy as np

from src.utils.logger import get_logger
from src.utils.constants import OPTION_LABELS, QUESTION_LABELS, UNATTEMPTED, INVALID
from config import BUBBLE_FILL_THRESHOLD, QUESTIONS_PER_PART, OPTIONS_PER_QUESTION

logger = get_logger(__name__)

# ─────────────────────────────────────────────
# TEMPLATE COORDINATES (pixels, 1240×1755 base)
# Measured from the blank scanned sheet.
# ─────────────────────────────────────────────
TEMPLATE_W = 1240
TEMPLATE_H = 1755
BUBBLE_RADIUS = 15   # px at template resolution

# Absolute X centres for each option column
PART1_X = [289, 369, 449, 535]   # A, B, C, D
PART2_X = [721, 799, 877, 961]   # A, B, C, D

# Absolute Y centres for each question row (Q01–Q08)
ROW_Y = [398, 432, 466, 500, 534, 568, 604, 638]


def _scale_coords(image: np.ndarray):
    """Returns (sx, sy) scale factors from template to actual image."""
    h, w = image.shape[:2]
    return w / TEMPLATE_W, h / TEMPLATE_H


def _read_bubble_fill(gray: np.ndarray, cx: int, cy: int, r: int) -> float:
    """
    Returns the fill ratio of a single bubble centred at (cx, cy) with
    radius r. 

    We use an inverted threshold: dark ink → high fill ratio.
    """
    h, w = gray.shape
    x1, y1 = max(cx - r, 0), max(cy - r, 0)
    x2, y2 = min(cx + r, w), min(cy + r, h)

    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0

    # Threshold: pixels darker than 128 are "ink"
    _, binary = cv2.threshold(roi, 128, 255, cv2.THRESH_BINARY_INV)

    # Circular mask
    mask = np.zeros_like(binary)
    local_cx = cx - x1
    local_cy = cy - y1
    cv2.circle(mask, (local_cx, local_cy), r, 255, -1)

    ink_pixels = cv2.countNonZero(cv2.bitwise_and(binary, binary, mask=mask))
    total_pixels = cv2.countNonZero(mask)

    return ink_pixels / total_pixels if total_pixels > 0 else 0.0


def _read_part(gray: np.ndarray, x_cols, sx: float, sy: float,
               threshold: float) -> Dict[str, str]:
    """Read all 8 questions for one part (Part-I or Part-II)."""
    answers = {}
    r = max(8, int(BUBBLE_RADIUS * min(sx, sy)))

    for q_idx, base_y in enumerate(ROW_Y):
        cy = int(base_y * sy)
        fills = []

        for base_x in x_cols:
            cx = int(base_x * sx)
            fill = _read_bubble_fill(gray, cx, cy, r)
            fills.append(fill)

        logger.debug(f"  Q{q_idx+1:02d} fills: "
                     + " ".join(f"{OPTION_LABELS[i]}={f:.2f}"
                                for i, f in enumerate(fills)))

        filled = [i for i, f in enumerate(fills) if f >= threshold]

        q_label = f"Q{str(q_idx + 1).zfill(2)}"
        if len(filled) == 0:
            answers[q_label] = UNATTEMPTED
        elif len(filled) > 1:
            answers[q_label] = INVALID
        else:
            answers[q_label] = OPTION_LABELS[filled[0]]

    return answers


def read_bubble_sheet(color_image: np.ndarray,
                      thresh_image: np.ndarray) -> Dict[str, Dict[str, str]]:
    """
    Main entry point for Task 3.
    Uses coordinate-based reading against the known template layout.
    """
    logger.info("Starting Bubble Sheet Reading...")

    # Work on grayscale for fill measurement
    if len(color_image.shape) == 3:
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = color_image.copy()

    sx, sy = _scale_coords(gray)
    logger.info(f"Scale factors: sx={sx:.3f} sy={sy:.3f}")

    # Use a fixed threshold slightly lower than config to handle
    # real-world ink variation
    threshold = BUBBLE_FILL_THRESHOLD

    logger.info("Processing Part-I bubbles...")
    part1 = _read_part(gray, PART1_X, sx, sy, threshold)

    logger.info("Processing Part-II bubbles...")
    part2 = _read_part(gray, PART2_X, sx, sy, threshold)

    logger.info("Bubble sheet reading complete.")
    logger.info(f"Part-I:  {part1}")
    logger.info(f"Part-II: {part2}")

    return {"part1": part1, "part2": part2}