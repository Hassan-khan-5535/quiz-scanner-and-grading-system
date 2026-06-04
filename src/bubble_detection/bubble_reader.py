"""
bubble_reader.py — Task 3: Bubble Detection & Answer Extraction
================================================================

APPROACH: Dynamic grid detection using HoughCircles.
The image is warped/resized to approximately 1240x1755 by the preprocessor.
Instead of using fixed template coordinates (which break with different
scans/photos), we dynamically detect the bubble circles using HoughCircles
and cluster them into the 8-row × 4-column × 2-part grid.

Falls back to template coordinates if dynamic detection fails.

Calibrated for: 1240x1755 preprocessed quiz sheet
"""

from typing import Dict, Any, List, Optional
import cv2
import numpy as np

from src.utils.logger import get_logger
from src.utils.constants import OPTION_LABELS, QUESTION_LABELS, UNATTEMPTED, INVALID
from config import BUBBLE_FILL_THRESHOLD, QUESTIONS_PER_PART, OPTIONS_PER_QUESTION

logger = get_logger(__name__)

# ─────────────────────────────────────────────
# TEMPLATE COORDINATES (fallback, for 1240x1755)
# Used ONLY when dynamic detection fails.
# ─────────────────────────────────────────────
TEMPLATE_W = 1240
TEMPLATE_H = 1755
BUBBLE_RADIUS = 15

FALLBACK_PART1_X = [289, 369, 449, 535]   # A, B, C, D
FALLBACK_PART2_X = [721, 799, 877, 961]   # A, B, C, D
FALLBACK_ROW_Y = [398, 432, 466, 500, 534, 568, 604, 638]

# ─────────────────────────────────────────────
# RELATIVE DETECTION PARAMETERS
# ─────────────────────────────────────────────
# Minimum absolute fill ratio to even consider a bubble as potentially filled.
# This prevents noise from being detected as a filled bubble.
MIN_FILL_ABSOLUTE = 0.12

# The dominant bubble's fill must be at least this many times higher than
# the average of the OTHER bubbles in the same row to count as "filled".
DOMINANCE_RATIO = 1.5


# ─────────────────────────────────────────────
# DYNAMIC GRID DETECTION
# ─────────────────────────────────────────────

def _cluster_values(values: List[int], threshold: int = 15) -> List[List[int]]:
    """
    Groups nearby integer values into clusters.
    Values within `threshold` pixels of each other are merged.
    
    Example: [100, 102, 105, 200, 203] with threshold=10
         --> [[100, 102, 105], [200, 203]]
    """
    if not values:
        return []
    sorted_vals = sorted(values)
    clusters = [[sorted_vals[0]]]
    for v in sorted_vals[1:]:
        if v - clusters[-1][-1] <= threshold:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return clusters


def _detect_grid(gray: np.ndarray) -> Optional[Dict]:
    """
    Dynamically detect bubble grid positions using HoughCircles.
    
    Algorithm:
    1. Detect circles in the upper portion of the image
    2. Cluster X positions into columns, filter noise
    3. Split columns at the largest gap (Part-I | Part-II boundary)
    4. Take rightmost 4 columns from left group = Part-I (A,B,C,D)
       Take rightmost 4 columns from right group = Part-II (A,B,C,D)
       (This skips any Q-label columns detected on the left edge)
    5. Cluster Y positions into rows, extrapolate to 8 if needed
    
    Returns:
        Dict with 'part1_x', 'part2_x', 'row_y', 'radius' or None on failure.
    """
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Try multiple HoughCircles parameter sets for robustness
    param_sets = [
        {'dp': 1.2, 'minDist': 25, 'param1': 50, 'param2': 30, 'minRadius': 10, 'maxRadius': 25},
        {'dp': 1.5, 'minDist': 20, 'param1': 80, 'param2': 25, 'minRadius': 8,  'maxRadius': 30},
        {'dp': 1.2, 'minDist': 20, 'param1': 40, 'param2': 25, 'minRadius': 8,  'maxRadius': 28},
    ]

    bubble_circles = []
    for params in param_sets:
        detected = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT,
            dp=params['dp'], minDist=params['minDist'],
            param1=params['param1'], param2=params['param2'],
            minRadius=params['minRadius'], maxRadius=params['maxRadius']
        )
        if detected is not None:
            circles_arr = np.round(detected[0]).astype(int)
            # Filter to the bubble region: upper 20-50% of height, inner 10-90% width
            filtered = [(int(x), int(y), int(r)) for x, y, r in circles_arr
                        if h * 0.20 < y < h * 0.50 and w * 0.10 < x < w * 0.90]
            if len(filtered) >= 30:
                bubble_circles = filtered
                logger.info(f"HoughCircles: found {len(filtered)} circles with param set {params}")
                break

    if len(bubble_circles) < 30:
        logger.warning(f"Dynamic grid detection failed: only {len(bubble_circles)} circles found (need 30+)")
        return None

    avg_radius = int(np.mean([r for _, _, r in bubble_circles]))

    # ── Step 1: Cluster X values into columns ──
    x_vals = [x for x, _, _ in bubble_circles]
    x_clusters = _cluster_values(x_vals, threshold=20)
    # Keep only columns with at least 4 circles (filters out noise and Q-label text)
    x_clusters = [c for c in x_clusters if len(c) >= 4]
    x_col_centers = sorted([int(np.mean(c)) for c in x_clusters])

    logger.debug(f"Detected column centers (count>=4): {x_col_centers}")

    if len(x_col_centers) < 8:
        # Relax threshold to count >= 3
        x_clusters = _cluster_values(x_vals, threshold=20)
        x_clusters = [c for c in x_clusters if len(c) >= 3]
        x_col_centers = sorted([int(np.mean(c)) for c in x_clusters])
        logger.debug(f"Relaxed column centers (count>=3): {x_col_centers}")

    if len(x_col_centers) < 8:
        logger.warning(f"Only {len(x_col_centers)} columns detected, need at least 8")
        return None

    # ── Step 2: Split at the largest gap (Part-I | Part-II boundary) ──
    gaps = [(x_col_centers[i + 1] - x_col_centers[i], i)
            for i in range(len(x_col_centers) - 1)]
    _, max_gap_idx = max(gaps, key=lambda g: g[0])

    left_cols = x_col_centers[:max_gap_idx + 1]
    right_cols = x_col_centers[max_gap_idx + 1:]

    # Take rightmost 4 from each group to skip label columns
    part1_x = left_cols[-4:] if len(left_cols) >= 4 else left_cols
    part2_x = right_cols[-4:] if len(right_cols) >= 4 else right_cols

    if len(part1_x) != 4 or len(part2_x) != 4:
        logger.warning(f"Column split failed: Part-I has {len(part1_x)}, Part-II has {len(part2_x)} (need 4 each)")
        return None

    # ── Step 3: Cluster Y values into rows ──
    y_vals = [y for _, y, _ in bubble_circles]
    y_clusters = _cluster_values(y_vals, threshold=15)
    # Sort by count descending to prioritize well-populated rows
    y_clusters_by_count = sorted(y_clusters, key=lambda c: -len(c))

    # Take up to 8 most populated row clusters
    top_rows = y_clusters_by_count[:min(8, len(y_clusters_by_count))]
    row_y = sorted([int(np.mean(c)) for c in top_rows])

    if len(row_y) < 4:
        logger.warning(f"Only {len(row_y)} rows detected, need at least 4 to extrapolate")
        return None

    # If we have fewer than 8 rows, extrapolate using average spacing
    if len(row_y) < 8:
        spacings = [row_y[i + 1] - row_y[i] for i in range(len(row_y) - 1)]
        avg_spacing = int(np.mean(spacings))
        logger.info(f"Only {len(row_y)} rows detected (avg spacing={avg_spacing}px). Extrapolating to 8.")
        while len(row_y) < 8:
            row_y.append(row_y[-1] + avg_spacing)

    row_y = row_y[:8]

    logger.info(f"Dynamic grid detected successfully:")
    logger.info(f"  Part-I  X (A,B,C,D): {part1_x}")
    logger.info(f"  Part-II X (A,B,C,D): {part2_x}")
    logger.info(f"  Row Y (Q01-Q08):     {row_y}")
    logger.info(f"  Bubble radius:       {avg_radius}px")

    return {
        'part1_x': part1_x,
        'part2_x': part2_x,
        'row_y': row_y,
        'radius': avg_radius
    }


# ─────────────────────────────────────────────
# BUBBLE FILL MEASUREMENT
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# GRID READING
# ─────────────────────────────────────────────

def _read_part(thresh: np.ndarray, x_cols: list, row_y: list, radius: int) -> Dict[str, str]:
    """Read all 8 questions for one part (Part-I or Part-II)."""
    answers = {}

    for q_idx, cy in enumerate(row_y):
        fills = []
        for cx in x_cols:
            fill = _read_bubble_fill(thresh, cx, cy, radius)
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
    
    Uses dynamic grid detection (HoughCircles) to find actual bubble
    positions in the image, then reads fill ratios at those positions.
    Falls back to template coordinates if dynamic detection fails.
    
    Image is expected to be approximately 1240x1755 after preprocessing.
    """
    logger.info("Starting Bubble Sheet Reading...")

    # Work on grayscale for fill measurement
    if len(color_image.shape) == 3:
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = color_image.copy()

    # ── Dynamic grid detection ──
    grid = _detect_grid(gray)

    if grid is not None:
        part1_x = grid['part1_x']
        part2_x = grid['part2_x']
        row_y = grid['row_y']
        radius = grid['radius']
        logger.info("Using dynamically detected grid coordinates.")
    else:
        # Fallback to hardcoded template coordinates
        part1_x = FALLBACK_PART1_X
        part2_x = FALLBACK_PART2_X
        row_y = FALLBACK_ROW_Y
        radius = BUBBLE_RADIUS
        logger.warning("Dynamic detection failed. Using fallback template coordinates.")

    logger.info("Processing Part-I bubbles...")
    part1 = _read_part(gray, part1_x, row_y, radius)

    logger.info("Processing Part-II bubbles...")
    part2 = _read_part(gray, part2_x, row_y, radius)

    logger.info("Bubble sheet reading complete.")
    logger.info(f"Part-I:  {part1}")
    logger.info(f"Part-II: {part2}")

    return {"part1": part1, "part2": part2}