"""
bubble_reader.py — Task 3: Bubble Detection & Answer Extraction
================================================================

APPROACH: Coordinate-based with RELATIVE fill detection.
The image is warped/resized to exactly 1240x1755 by the preprocessor,
so we use the known template coordinates directly without scaling.

Instead of a single absolute threshold, we use relative detection:
for each question row, if one bubble's fill ratio stands out significantly
above the others, it's considered filled. This handles variable lighting,
camera quality, and ink darkness much better than a flat threshold.

Calibrated for: 1240x1755 scanned quiz sheet (CamScanner output)
"""

from typing import Dict, Any, Tuple
import cv2
import numpy as np

from src.utils.logger import get_logger
from src.utils.constants import OPTION_LABELS, QUESTION_LABELS, UNATTEMPTED, INVALID
from config import BUBBLE_FILL_THRESHOLD, QUESTIONS_PER_PART, OPTIONS_PER_QUESTION

logger = get_logger(__name__)

# ─────────────────────────────────────────────
# TEMPLATE COORDINATES (pixels, 1240x1755 base)
# ─────────────────────────────────────────────
TEMPLATE_W = 1240
TEMPLATE_H = 1755
BUBBLE_RADIUS = 15   # px at template resolution

# Absolute X centres for each option column
PART1_X = [289, 369, 449, 535]   # A, B, C, D
PART2_X = [721, 799, 877, 961]   # A, B, C, D

# Absolute Y centres for each question row (Q01-Q08)
ROW_Y = [398, 432, 466, 500, 534, 568, 604, 638]

# ─────────────────────────────────────────────
# RELATIVE DETECTION PARAMETERS
# ─────────────────────────────────────────────
# Minimum absolute fill ratio to even consider a bubble as potentially filled.
# This prevents noise from being detected as a filled bubble.
MIN_FILL_ABSOLUTE = 0.12

# The dominant bubble's fill must be at least this many times higher than
# the average of the OTHER bubbles in the same row to count as "filled".
DOMINANCE_RATIO = 1.8


def _read_bubble_fill(gray: np.ndarray, cx: int, cy: int, r: int) -> float:
    """
    Returns the fill ratio of a single bubble centred at (cx, cy) with
    radius r.

    We use an inverted threshold: dark ink -> high fill ratio.
    Uses the grayscale image directly with a threshold of 140.
    """
    h, w = gray.shape
    x1, y1 = max(cx - r, 0), max(cy - r, 0)
    x2, y2 = min(cx + r, w), min(cy + r, h)

    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0

    # Threshold: pixels darker than 140 are "ink"
    _, binary = cv2.threshold(roi, 140, 255, cv2.THRESH_BINARY_INV)

    # Circular mask
    mask = np.zeros_like(binary)
    local_cx = cx - x1
    local_cy = cy - y1
    cv2.circle(mask, (local_cx, local_cy), r, 255, -1)

    ink_pixels = cv2.countNonZero(cv2.bitwise_and(binary, binary, mask=mask))
    total_pixels = cv2.countNonZero(mask)

    return ink_pixels / total_pixels if total_pixels > 0 else 0.0


def _detect_answer(fills: list, q_label: str) -> str:
    """
    Uses relative detection to determine which bubble (if any) is filled.
    
    Logic:
    1. Find the option with the highest fill ratio.
    2. If it's below MIN_FILL_ABSOLUTE, mark as unattempted.
    3. Compare it to the average of the OTHER options.
    4. If it's DOMINANCE_RATIO times higher, it's the answer.
    5. If multiple options pass, mark as INVALID.
    """
    max_fill = max(fills)
    
    # If no bubble has meaningful ink, it's unattempted
    if max_fill < MIN_FILL_ABSOLUTE:
        return UNATTEMPTED
    
    # Find all options that could be "filled"
    # An option is a candidate if:
    #   (a) it's above MIN_FILL_ABSOLUTE, AND
    #   (b) it's significantly higher than the other bubbles
    candidates = []
    
    for i, fill in enumerate(fills):
        if fill < MIN_FILL_ABSOLUTE:
            continue
            
        # Calculate the average of the OTHER options
        other_fills = [f for j, f in enumerate(fills) if j != i]
        other_avg = sum(other_fills) / len(other_fills) if other_fills else 0
        
        # The bubble must dominate the others
        if other_avg < 0.01:
            # All others are basically zero, this one clearly stands out
            candidates.append(i)
        elif fill / other_avg >= DOMINANCE_RATIO:
            candidates.append(i)
    
    if len(candidates) == 0:
        # No single bubble dominates — could be all slightly dark (unfilled printed circles)
        return UNATTEMPTED
    elif len(candidates) == 1:
        return OPTION_LABELS[candidates[0]]
    else:
        # Multiple bubbles are filled
        return INVALID


def _read_part(gray: np.ndarray, x_cols, threshold: float) -> Dict[str, str]:
    """Read all 8 questions for one part (Part-I or Part-II)."""
    answers = {}
    r = BUBBLE_RADIUS  # No scaling needed - image is already 1240x1755

    for q_idx, cy in enumerate(ROW_Y):
        fills = []

        for cx in x_cols:
            fill = _read_bubble_fill(gray, cx, cy, r)
            fills.append(fill)

        logger.debug(f"  Q{q_idx+1:02d} fills: "
                     + " ".join(f"{OPTION_LABELS[i]}={f:.2f}"
                                for i, f in enumerate(fills)))

        q_label = f"Q{str(q_idx + 1).zfill(2)}"
        answer = _detect_answer(fills, q_label)
        answers[q_label] = answer

    return answers


def read_bubble_sheet(color_image: np.ndarray,
                      thresh_image: np.ndarray) -> Dict[str, Dict[str, str]]:
    """
    Main entry point for Task 3.
    Uses coordinate-based reading against the known template layout.
    Image is guaranteed to be 1240x1755 after preprocessing.
    """
    logger.info("Starting Bubble Sheet Reading...")

    # Work on grayscale for fill measurement
    if len(color_image.shape) == 3:
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = color_image.copy()

    threshold = BUBBLE_FILL_THRESHOLD

    logger.info("Processing Part-I bubbles...")
    part1 = _read_part(gray, PART1_X, threshold)

    logger.info("Processing Part-II bubbles...")
    part2 = _read_part(gray, PART2_X, threshold)

    logger.info("Bubble sheet reading complete.")
    logger.info(f"Part-I:  {part1}")
    logger.info(f"Part-II: {part2}")

    return {"part1": part1, "part2": part2}