"""
bubble_reader.py — Task 3: Bubble Detection & Answer Extraction
================================================================

Robust bubble detection using template-guided local search.
Uses expected template coordinates as starting points, then
searches locally for actual bubble centers using circle detection.

APPROACH:
1. Start with expected template coordinates (scaled to image)
2. Search in small ROI around each expected position for actual bubble
3. Use circle detection to find precise bubble center
4. Check center pixel intensity to determine if filled
5. Select answer based on darkest center with dominance check
"""

from typing import Dict, Any, List, Tuple
import cv2
import numpy as np

from src.utils.logger import get_logger
from src.utils.constants import OPTION_LABELS, UNATTEMPTED, INVALID
from config import (
    BUBBLE_FILL_THRESHOLD,
    DOMINANCE_RATIO,
    QUESTIONS_PER_PART,
    OPTIONS_PER_QUESTION,
)

logger = get_logger(__name__)

# Template coordinates for 1240x1755 reference image
TEMPLATE_W = 1240
TEMPLATE_H = 1755

# Part 1 X coordinates (A, B, C, D columns) - CALIBRATED from actual image
PART1_X_TEMPLATE = [92, 194, 310, 434]

# Part 2 X coordinates (A, B, C, D columns) - CALIBRATED from actual image
# Adjusted based on actual bubble positions
PART2_X_TEMPLATE = [730, 837, 944, 1078]

# Part 2 Y coordinates - different from Part-I
PART2_Y_TEMPLATE = [325, 445, 550, 670, 700, 800, 900, 950]

# Y coordinates for Q01-Q08 rows - CALIBRATED from actual image
# Based on analysis of filled bubble positions
ROW_Y_TEMPLATE = [310, 445, 600, 670, 745, 800, 955, 970]

# Search radius around expected position
SEARCH_RADIUS = 25


def _find_bubble_center(gray: np.ndarray, expected_x: int, expected_y: int, 
                        search_radius: int = SEARCH_RADIUS) -> Tuple[int, int]:
    """
    Find actual bubble center by searching around expected position.
    
    Returns:
        (actual_x, actual_y) - refined bubble center coordinates
    """
    h, w = gray.shape
    
    # Define search region
    x1 = max(expected_x - search_radius, 0)
    x2 = min(expected_x + search_radius, w)
    y1 = max(expected_y - search_radius, 0)
    y2 = min(expected_y + search_radius, h)
    
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return expected_x, expected_y
    
    # Detect circles in ROI
    blurred = cv2.GaussianBlur(roi, (5, 5), 0)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=15,
        param1=50,
        param2=15,
        minRadius=10,
        maxRadius=25
    )
    
    if circles is not None and len(circles[0]) > 0:
        # Use the circle closest to center of ROI
        cx_roi = roi.shape[1] // 2
        cy_roi = roi.shape[0] // 2
        
        circles = circles[0, :]
        # Find circle closest to center
        best_circle = min(circles, key=lambda c: abs(c[0] - cx_roi) + abs(c[1] - cy_roi))
        
        # Convert back to absolute coordinates
        actual_x = int(best_circle[0]) + x1
        actual_y = int(best_circle[1]) + y1
        return actual_x, actual_y
    
    # No circle found, return expected position
    return expected_x, expected_y


def _check_bubble_filled(gray: np.ndarray, cx: int, cy: int, radius: int = 8) -> float:
    """
    Check if a bubble is filled by examining the center region.
    
    Strategy: Look at the center 5x5 pixel region. 
    - Filled bubble: center is dark (low intensity)
    - Empty bubble: center is white (high intensity)
    
    Returns fill score from 0.0 (empty/white) to 1.0 (filled/black)
    """
    h, w = gray.shape
    
    # Define small center region (5x5 pixels around center)
    r = 2
    x1 = max(cx - r, 0)
    y1 = max(cy - r, 0)
    x2 = min(cx + r + 1, w)
    y2 = min(cy + r + 1, h)
    
    # Extract center region
    center_roi = gray[y1:y2, x1:x2]
    if center_roi.size == 0:
        return 0.0
    
    # Calculate mean intensity (0-255)
    mean_intensity = np.mean(center_roi)
    
    # Convert to fill score: white (255) -> 0.0, black (0) -> 1.0
    # Use threshold at 180: above = empty, below = potentially filled
    if mean_intensity > 180:
        return 0.0  # Definitely empty (white paper)
    elif mean_intensity < 100:
        return 1.0  # Definitely filled (dark ink)
    else:
        # Linear interpolation between 100-180
        return (180 - mean_intensity) / 80.0


def _detect_answer(fill_scores: List[float], q_label: str) -> str:
    """
    Determine which bubble is filled based on fill scores.
    
    Logic:
    1. Find bubble with highest fill score
    2. Check if it's dark enough to be considered filled
    3. Check if it's clearly darker than other bubbles
    4. Return answer label, UNATTEMPTED, or INVALID
    """
    # Log scores for debugging
    debug_info = " | ".join([
        f"{OPTION_LABELS[i]}: {fill_scores[i]:.2f}"
        for i in range(len(OPTION_LABELS))
    ])
    logger.info(f"  {q_label} scores: {debug_info}")
    
    max_fill = max(fill_scores)
    max_idx = fill_scores.index(max_fill)
    
    # If best bubble is essentially empty, question is unattempted
    if max_fill < 0.15:
        logger.info(f"  {q_label}: UNATTEMPTED (max fill {max_fill:.2f} < 0.15)")
        return UNATTEMPTED
    
    # Calculate second-best fill
    other_fills = [fill_scores[i] for i in range(len(fill_scores)) if i != max_idx]
    second_best = max(other_fills) if other_fills else 0
    
    # Check if this bubble clearly dominates
    fill_gap = max_fill - second_best
    
    # Strong fill with clear separation -> valid answer
    if max_fill >= 0.5 and fill_gap >= 0.20:
        selected = OPTION_LABELS[max_idx]
        logger.info(f"  {q_label}: {selected} (strong fill={max_fill:.2f}, gap={fill_gap:.2f})")
        return selected
    
    # Moderate fill with good separation -> valid answer
    if max_fill >= 0.35 and fill_gap >= 0.15:
        selected = OPTION_LABELS[max_idx]
        logger.info(f"  {q_label}: {selected} (moderate fill={max_fill:.2f}, gap={fill_gap:.2f})")
        return selected
    
    # Light fill but clearly better than others -> valid answer
    if max_fill >= 0.20 and fill_gap >= 0.10:
        selected = OPTION_LABELS[max_idx]
        logger.info(f"  {q_label}: {selected} (light fill={max_fill:.2f}, gap={fill_gap:.2f})")
        return selected
    
    # Multiple bubbles have similar fill -> INVALID
    # Require at least 2 bubbles with significant fill (>= 0.35) to be considered invalid
    significant_fills = [f for f in fill_scores if f >= 0.35]
    if len(significant_fills) >= 2:
        logger.info(f"  {q_label}: INVALID ({len(significant_fills)} bubbles with fill >= 0.35)")
        return INVALID
    
    # Single bubble filled but not dominant enough
    if max_fill >= 0.25:
        selected = OPTION_LABELS[max_idx]
        logger.info(f"  {q_label}: {selected} (single fill={max_fill:.2f})")
        return selected
    
    logger.info(f"  {q_label}: UNATTEMPTED")
    return UNATTEMPTED


def _read_part(gray: np.ndarray, x_cols: List[int], y_rows: List[int], 
               part_name: str) -> Dict[str, str]:
    """
    Read all 8 questions for one part using template coordinates.
    Checks fill directly at template positions (no local search).
    """
    answers = {}
    
    logger.info(f"Processing {part_name}...")
    
    for q_idx, cy in enumerate(y_rows):
        fill_scores = []
        
        for cx in x_cols:
            # Check if bubble at this position is filled
            fill_score = _check_bubble_filled(gray, cx, cy)
            fill_scores.append(fill_score)
        
        q_label = f"Q{str(q_idx + 1).zfill(2)}"
        answer = _detect_answer(fill_scores, q_label)
        answers[q_label] = answer
    
    return answers


def read_bubble_sheet(color_image: np.ndarray,
                      thresh_image: np.ndarray = None,
                      use_hough: bool = None) -> Dict[str, Dict[str, str]]:
    """
    Main entry point for bubble detection.
    
    Args:
        color_image: The warped color image (BGR format)
        thresh_image: Optional pre-thresholded image (not used)
        use_hough: Ignored - always uses template-based detection
    
    Returns:
        Dictionary with 'part1' and 'part2' containing answer dictionaries
    """
    logger.info("=" * 50)
    logger.info("Starting Bubble Sheet Reading...")
    logger.info("Using template-guided detection")
    logger.info("=" * 50)
    
    # Convert to grayscale
    if len(color_image.shape) == 3:
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = color_image.copy()
    
    # Get actual image dimensions
    h, w = gray.shape
    
    # Calculate scale factors
    scale_x = w / TEMPLATE_W
    scale_y = h / TEMPLATE_H
    
    logger.info(f"Image dimensions: {w}x{h}, Template: {TEMPLATE_W}x{TEMPLATE_H}")
    logger.info(f"Scale factors: x={scale_x:.3f}, y={scale_y:.3f}")
    
    # Scale template coordinates to match image
    part1_x = [int(x * scale_x) for x in PART1_X_TEMPLATE]
    part2_x = [int(x * scale_x) for x in PART2_X_TEMPLATE]
    part1_y = [int(y * scale_y) for y in ROW_Y_TEMPLATE]
    part2_y = [int(y * scale_y) for y in PART2_Y_TEMPLATE]
    
    logger.info(f"Template Part-I X: {part1_x}")
    logger.info(f"Template Part-II X: {part2_x}")
    logger.info(f"Template Part-I Y: {part1_y}")
    logger.info(f"Template Part-II Y: {part2_y}")
    
    # Process both parts with their respective Y coordinates
    part1 = _read_part(gray, part1_x, part1_y, "Part-I")
    part2 = _read_part(gray, part2_x, part2_y, "Part-II")
    
    logger.info("=" * 50)
    logger.info("Bubble sheet reading complete.")
    logger.info(f"Part-I: {part1}")
    logger.info(f"Part-II: {part2}")
    logger.info("=" * 50)
    
    return {"part1": part1, "part2": part2}


def visualize_detection(color_image: np.ndarray, answers: Dict[str, Dict[str, str]], 
                       output_path: str = None) -> np.ndarray:
    """
    Create a visualization of detected bubbles with their answers.
    """
    vis_image = color_image.copy()
    
    # Get image dimensions for scaling
    h, w = vis_image.shape[:2]
    scale_x = w / TEMPLATE_W
    scale_y = h / TEMPLATE_H
    
    # Scale coordinates
    part1_x = [int(x * scale_x) for x in PART1_X_TEMPLATE]
    part2_x = [int(x * scale_x) for x in PART2_X_TEMPLATE]
    row_y = [int(y * scale_y) for y in ROW_Y_TEMPLATE]
    
    # Colors
    COLOR_FILLED = (0, 255, 0)        # Green
    COLOR_UNATTEMPTED = (128, 128, 128)  # Gray
    COLOR_INVALID = (0, 165, 255)     # Orange
    
    for part_idx, (part_name, x_cols) in enumerate([("part1", part1_x), ("part2", part2_x)]):
        part_answers = answers.get(part_name, {})
        
        for q_idx, cy in enumerate(row_y):
            q_label = f"Q{str(q_idx + 1).zfill(2)}"
            answer = part_answers.get(q_label, UNATTEMPTED)
            
            for opt_idx, cx in enumerate(x_cols):
                option = OPTION_LABELS[opt_idx]
                
                # Determine color based on answer
                if answer == option:
                    color = COLOR_FILLED
                    thickness = 3
                    # Draw option label
                    cv2.putText(vis_image, option, (cx - 5, cy + 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
                elif answer == UNATTEMPTED:
                    color = COLOR_UNATTEMPTED
                    thickness = 1
                elif answer == INVALID:
                    color = COLOR_INVALID
                    thickness = 2
                else:
                    color = (255, 255, 255)  # White
                    thickness = 1
                
                # Draw circle
                radius = int(8 * min(scale_x, scale_y))
                cv2.circle(vis_image, (cx, cy), radius, color, thickness)
    
    if output_path:
        cv2.imwrite(output_path, vis_image)
    
    return vis_image
