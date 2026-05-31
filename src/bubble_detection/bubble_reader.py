"""
bubble_reader.py — Task 3: Bubble Detection & Answer Extraction
================================================================

APPROACH: Hybrid detection combining coordinate-based ROI with adaptive circle detection.
Uses computer vision to dynamically locate bubbles within expected regions,
then applies adaptive thresholding for robust fill detection across varying
lighting conditions and image qualities.

Key Improvements:
1. Dynamic bubble localization using circle detection within ROIs
2. Adaptive thresholding (Otsu) instead of fixed threshold
3. Morphological operations to clean noise
4. Multi-metric fill analysis (darkness + contour analysis)
5. Smart confidence-based answer detection
"""

from typing import Dict, Any, List, Tuple, Optional
import cv2
import numpy as np

from src.utils.logger import get_logger
from src.utils.constants import OPTION_LABELS, UNATTEMPTED, INVALID
from config import BUBBLE_FILL_THRESHOLD, QUESTIONS_PER_PART, OPTIONS_PER_QUESTION

logger = get_logger(__name__)

# ─────────────────────────────────────────────
# TEMPLATE COORDINATES (pixels, 1240x1755 base)
# These are EXPECTED positions - actual bubbles are searched nearby
# ─────────────────────────────────────────────
TEMPLATE_W = 1240
TEMPLATE_H = 1755

# Search radius around expected coordinates to find actual bubble
BUBBLE_SEARCH_RADIUS = 25  # pixels to search around expected center
BUBBLE_RADIUS = 15  # expected bubble radius

# Absolute X centres for each option column (A, B, C, D)
PART1_X = [289, 369, 449, 535]   # Left side
PART2_X = [721, 799, 877, 961]   # Right side

# Absolute Y centres for each question row (Q01-Q08)
ROW_Y = [398, 432, 466, 500, 534, 568, 604, 638]

# ─────────────────────────────────────────────
# DETECTION PARAMETERS
# ─────────────────────────────────────────────
# Minimum confidence to consider a bubble as filled (0.0 to 1.0)
MIN_CONFIDENCE = 0.35

# Ratio a filled bubble must exceed others by to be considered selected
DOMINANCE_RATIO = 1.5

# Minimum darkness ratio (filled vs empty bubble reference)
MIN_DARKNESS_RATIO = 1.3


def _preprocess_roi(roi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess a bubble ROI for analysis.
    Returns both the original ROI and a binary version using adaptive thresholding.
    """
    if roi.size == 0:
        return roi, np.zeros_like(roi)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(roi, (3, 3), 0)
    
    # Use Otsu's thresholding for automatic threshold selection
    # This adapts to the local lighting conditions
    if len(blurred.shape) == 3:
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    else:
        gray = blurred
    
    # Otsu's thresholding - automatically finds optimal threshold
    _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Also create a slightly more sensitive binary version for comparison
    mean_val = np.mean(gray)
    threshold = max(mean_val * 0.7, 80)  # Adaptive threshold based on local mean
    _, binary_adaptive = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Combine both approaches
    binary = cv2.bitwise_or(binary_otsu, binary_adaptive)
    
    # Morphological operations to clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return gray, binary


def _find_bubble_center(gray: np.ndarray, expected_cx: int, expected_cy: int, 
                        search_radius: int) -> Tuple[int, int]:
    """
    Dynamically find the actual bubble center by looking for circular shapes
    near the expected coordinates.
    """
    h, w = gray.shape
    
    # Define search region
    x1 = max(expected_cx - search_radius, 0)
    y1 = max(expected_cy - search_radius, 0)
    x2 = min(expected_cx + search_radius, w)
    y2 = min(expected_cy + search_radius, h)
    
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return expected_cx, expected_cy
    
    # Try to find circles using Hough transform
    try:
        # Normalize ROI for circle detection
        roi_normalized = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX)
        
        # Detect circles
        circles = cv2.HoughCircles(
            roi_normalized,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=10,
            param1=50,
            param2=20,
            minRadius=8,
            maxRadius=22
        )
        
        if circles is not None:
            # Find the circle closest to the expected center
            circles = np.uint16(np.around(circles))
            best_circle = None
            min_dist = float('inf')
            
            for circle in circles[0, :]:
                cx, cy, radius = circle
                # Distance from expected center (relative to ROI)
                dist = np.sqrt((cx - (expected_cx - x1))**2 + (cy - (expected_cy - y1))**2)
                if dist < min_dist:
                    min_dist = dist
                    best_circle = circle
            
            if best_circle is not None and min_dist < search_radius:
                actual_cx = x1 + best_circle[0]
                actual_cy = y1 + best_circle[1]
                return actual_cx, actual_cy
                
    except Exception as e:
        logger.debug(f"Circle detection failed: {e}")
    
    # Fallback: return expected coordinates
    return expected_cx, expected_cy


def _analyze_bubble_fill(gray: np.ndarray, cx: int, cy: int, r: int) -> Dict[str, float]:
    """
    Comprehensive bubble fill analysis using multiple metrics.
    Returns a dictionary with various fill measurements.
    """
    h, w = gray.shape
    x1, y1 = max(cx - r, 0), max(cy - r, 0)
    x2, y2 = min(cx + r, w), min(cy + r, h)
    
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return {"fill_ratio": 0.0, "darkness": 0.0, "confidence": 0.0}
    
    # Preprocess ROI
    gray_roi, binary = _preprocess_roi(roi)
    
    # Create circular mask
    mask = np.zeros_like(binary)
    local_cx = cx - x1
    local_cy = cy - y1
    cv2.circle(mask, (local_cx, local_cy), r, 255, -1)
    
    # Calculate fill metrics
    # 1. Fill ratio: percentage of dark pixels inside bubble
    ink_pixels = cv2.countNonZero(cv2.bitwise_and(binary, binary, mask=mask))
    total_pixels = cv2.countNonZero(mask)
    fill_ratio = ink_pixels / total_pixels if total_pixels > 0 else 0.0
    
    # 2. Darkness: average intensity of pixels inside bubble (lower = darker)
    bubble_region = cv2.bitwise_and(gray_roi, gray_roi, mask=mask)
    non_zero_pixels = bubble_region[bubble_region > 0]
    if len(non_zero_pixels) > 0:
        avg_intensity = np.mean(non_zero_pixels)
        darkness = (255 - avg_intensity) / 255.0  # Normalize to 0-1
    else:
        darkness = 0.0
    
    # 3. Contour analysis: look for filled shape
    contours, _ = cv2.findContours(
        cv2.bitwise_and(binary, binary, mask=mask),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    contour_fill = 0.0
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)
        contour_fill = contour_area / total_pixels if total_pixels > 0 else 0.0
    
    # Combined confidence score (weighted combination)
    # Fill ratio is most important, darkness helps confirm, contour provides validation
    confidence = (
        fill_ratio * 0.5 +
        darkness * 0.3 +
        min(contour_fill * 2, 1.0) * 0.2  # Scale contour fill up a bit
    )
    
    return {
        "fill_ratio": fill_ratio,
        "darkness": darkness,
        "contour_fill": contour_fill,
        "confidence": confidence
    }


def _detect_answer(bubble_data: List[Dict[str, float]], q_label: str) -> str:
    """
    Determine which bubble (if any) is filled based on confidence scores.
    
    Logic:
    1. Calculate confidence scores for all options
    2. Find the option with highest confidence
    3. Check if it dominates the others sufficiently
    4. Return answer label, UNATTEMPTED, or INVALID
    """
    # Extract confidence scores
    confidences = [data["confidence"] for data in bubble_data]
    fill_ratios = [data["fill_ratio"] for data in bubble_data]
    darkness_values = [data["darkness"] for data in bubble_data]
    
    # Log detailed analysis
    debug_info = " | ".join([
        f"{OPTION_LABELS[i]}: c={confidences[i]:.2f}, f={fill_ratios[i]:.2f}, d={darkness_values[i]:.2f}"
        for i in range(len(OPTION_LABELS))
    ])
    logger.debug(f"  {q_label}: {debug_info}")
    
    max_conf = max(confidences)
    
    # If no bubble has meaningful confidence, it's unattempted
    if max_conf < MIN_CONFIDENCE:
        logger.debug(f"  -> UNATTEMPTED (max confidence {max_conf:.2f} < {MIN_CONFIDENCE})")
        return UNATTEMPTED
    
    # Find all candidates that pass the minimum threshold
    candidates = []
    for i, conf in enumerate(confidences):
        if conf < MIN_CONFIDENCE:
            continue
        
        # Calculate average of OTHER options
        other_confs = [c for j, c in enumerate(confidences) if j != i]
        other_avg = sum(other_confs) / len(other_confs) if other_confs else 0
        
        # Check if this option dominates others
        if other_avg < 0.05:  # Others are essentially empty
            if conf > MIN_CONFIDENCE:
                candidates.append(i)
        elif conf / other_avg >= DOMINANCE_RATIO:
            candidates.append(i)
        elif conf > 0.5 and conf - other_avg > 0.2:  # Strong absolute confidence with gap
            candidates.append(i)
    
    if len(candidates) == 0:
        logger.debug(f"  -> UNATTEMPTED (no dominant candidate)")
        return UNATTEMPTED
    elif len(candidates) == 1:
        selected = OPTION_LABELS[candidates[0]]
        logger.debug(f"  -> {selected}")
        return selected
    else:
        # Multiple candidates - check if one is clearly stronger
        candidate_confs = [confidences[i] for i in candidates]
        max_cand_conf = max(candidate_confs)
        second_max = sorted(candidate_confs, reverse=True)[1] if len(candidate_confs) > 1 else 0
        
        # If the top candidate is significantly stronger, use it
        if max_cand_conf / second_max >= 1.3:
            best_idx = candidates[candidate_confs.index(max_cand_conf)]
            selected = OPTION_LABELS[best_idx]
            logger.debug(f"  -> {selected} (best of multiple)")
            return selected
        
        logger.debug(f"  -> INVALID (multiple: {[OPTION_LABELS[i] for i in candidates]})")
        return INVALID


def _read_part(gray: np.ndarray, x_cols: List[int], part_name: str) -> Dict[str, str]:
    """
    Read all 8 questions for one part (Part-I or Part-II).
    """
    answers = {}
    
    logger.info(f"Processing {part_name}...")
    
    for q_idx, expected_cy in enumerate(ROW_Y):
        bubble_data = []
        
        for expected_cx in x_cols:
            # Dynamically find bubble center
            actual_cx, actual_cy = _find_bubble_center(
                gray, expected_cx, expected_cy, BUBBLE_SEARCH_RADIUS
            )
            
            # Analyze bubble fill
            analysis = _analyze_bubble_fill(gray, actual_cx, actual_cy, BUBBLE_RADIUS)
            bubble_data.append(analysis)
        
        q_label = f"Q{str(q_idx + 1).zfill(2)}"
        answer = _detect_answer(bubble_data, q_label)
        answers[q_label] = answer
    
    return answers


def read_bubble_sheet(color_image: np.ndarray,
                      thresh_image: np.ndarray = None) -> Dict[str, Dict[str, str]]:
    """
    Main entry point for Task 3.
    
    Args:
        color_image: The warped color image (BGR format)
        thresh_image: Optional pre-thresholded image (not used in new implementation)
    
    Returns:
        Dictionary with 'part1' and 'part2' containing answer dictionaries
    """
    logger.info("=" * 50)
    logger.info("Starting Bubble Sheet Reading...")
    logger.info("=" * 50)
    
    # Convert to grayscale for analysis
    if len(color_image.shape) == 3:
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = color_image.copy()
    
    # Validate image dimensions
    h, w = gray.shape
    if h != TEMPLATE_H or w != TEMPLATE_W:
        logger.warning(f"Image dimensions ({w}x{h}) differ from template ({TEMPLATE_W}x{TEMPLATE_H})")
        logger.warning("Resizing to match template...")
        gray = cv2.resize(gray, (TEMPLATE_W, TEMPLATE_H))
    
    # Process both parts
    part1 = _read_part(gray, PART1_X, "Part-I")
    part2 = _read_part(gray, PART2_X, "Part-II")
    
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
    Useful for debugging and verification.
    """
    vis_image = color_image.copy()
    
    # Colors for different states
    COLOR_CORRECT = (0, 255, 0)      # Green
    COLOR_INCORRECT = (0, 0, 255)    # Red
    COLOR_UNATTEMPTED = (128, 128, 128)  # Gray
    COLOR_INVALID = (0, 165, 255)     # Orange
    COLOR_EMPTY = (255, 255, 255)     # White
    
    for part_idx, (part_name, x_cols) in enumerate([("part1", PART1_X), ("part2", PART2_X)]):
        part_answers = answers.get(part_name, {})
        
        for q_idx, cy in enumerate(ROW_Y):
            q_label = f"Q{str(q_idx + 1).zfill(2)}"
            answer = part_answers.get(q_label, UNATTEMPTED)
            
            for opt_idx, cx in enumerate(x_cols):
                option = OPTION_LABELS[opt_idx]
                
                # Determine color based on answer
                if answer == option:
                    color = COLOR_CORRECT
                    thickness = 3
                elif answer == UNATTEMPTED:
                    color = COLOR_UNATTEMPTED
                    thickness = 1
                elif answer == INVALID:
                    color = COLOR_INVALID
                    thickness = 2
                else:
                    color = COLOR_EMPTY
                    thickness = 1
                
                # Draw circle
                cv2.circle(vis_image, (cx, cy), BUBBLE_RADIUS, color, thickness)
                
                # Draw option label if selected
                if answer == option:
                    cv2.putText(vis_image, option, (cx - 5, cy + 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
    
    if output_path:
        cv2.imwrite(output_path, vis_image)
    
    return vis_image
