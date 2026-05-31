"""
bubble_reader.py — Task 3: Bubble Detection & Answer Extraction
================================================================

Simplified and robust bubble detection using direct ROI analysis.
Analyzes the expected bubble locations directly without complex
circle detection that can fail on real-world images.

APPROACH:
1. Use expected template coordinates (calibrated for 1240x1755)
2. Extract ROI around each expected bubble location
3. Calculate fill percentage by counting dark pixels
4. Apply threshold to determine if bubble is filled
5. Select answer based on highest fill ratio with dominance check
"""

from typing import Dict, Any, List, Tuple
import cv2
import numpy as np

from src.utils.logger import get_logger
from src.utils.constants import OPTION_LABELS, UNATTEMPTED, INVALID
from config import (
    BUBBLE_FILL_THRESHOLD, 
    BUBBLE_CONFIDENCE_THRESHOLD,
    DOMINANCE_RATIO,
    QUESTIONS_PER_PART, 
    OPTIONS_PER_QUESTION,
    USE_HOUGH_CIRCLES,
    HOUGH_BLUR_KERNEL,
    HOUGH_DP,
    HOUGH_MIN_DIST,
    HOUGH_PARAM1,
    HOUGH_PARAM2,
    HOUGH_MIN_RADIUS,
    HOUGH_MAX_RADIUS,
    HOUGH_FILL_THRESHOLD
)

logger = get_logger(__name__)

# ─────────────────────────────────────────────
# TEMPLATE COORDINATES (pixels, 1240x1755 base)
# These are calibrated positions for the bubble centers
# ─────────────────────────────────────────────
TEMPLATE_W = 1240
TEMPLATE_H = 1755

# Absolute X centres for each option column (A, B, C, D)
PART1_X = [289, 369, 449, 535]   # Left side (Part-I)
PART2_X = [721, 799, 877, 961]   # Right side (Part-II)

# Absolute Y centres for each question row (Q01-Q08)
ROW_Y = [398, 432, 466, 500, 534, 568, 604, 638]

# Bubble radius for ROI extraction
BUBBLE_RADIUS = 18


def _calculate_bubble_fill(gray: np.ndarray, cx: int, cy: int, radius: int) -> float:
    """
    Calculate the fill ratio of a bubble at the given coordinates.
    
    Returns a value from 0.0 (empty) to 1.0 (completely filled).
    Uses adaptive thresholding to handle different lighting/ink conditions.
    """
    h, w = gray.shape
    
    # Define ROI bounds
    x1 = max(cx - radius, 0)
    y1 = max(cy - radius, 0)
    x2 = min(cx + radius, w)
    y2 = min(cy + radius, h)
    
    # Extract ROI
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    
    # Create circular mask
    mask = np.zeros(roi.shape, dtype=np.uint8)
    local_cx = cx - x1
    local_cy = cy - y1
    cv2.circle(mask, (local_cx, local_cy), radius, 255, -1)
    
    # Count pixels inside the circle
    total_pixels = cv2.countNonZero(mask)
    if total_pixels == 0:
        return 0.0
    
    # Calculate average intensity inside the bubble (only masked area)
    masked_roi = cv2.bitwise_and(roi, roi, mask=mask)
    
    # Calculate mean intensity of the bubble area (non-zero masked pixels)
    # This helps us determine an adaptive threshold
    roi_nonzero = masked_roi[mask == 255]
    if roi_nonzero.size == 0:
        return 0.0
    
    mean_intensity = np.mean(roi_nonzero)
    std_intensity = np.std(roi_nonzero)
    
    # Adaptive threshold: use mean - 0.5*std as the cutoff
    # This is more robust than fixed 128 threshold
    # Empty bubbles (white paper) have high mean intensity
    # Filled bubbles have lower mean intensity
    adaptive_threshold = max(mean_intensity - 0.5 * std_intensity, 100)
    adaptive_threshold = min(adaptive_threshold, 180)  # Cap at 180 to avoid being too strict
    
    # Count dark pixels (ink) - pixels below adaptive threshold
    _, dark_mask = cv2.threshold(masked_roi, int(adaptive_threshold), 255, cv2.THRESH_BINARY_INV)
    dark_pixels = cv2.countNonZero(cv2.bitwise_and(dark_mask, mask))
    
    fill_ratio = dark_pixels / total_pixels
    
    logger.debug(f"Bubble at ({cx}, {cy}): mean={mean_intensity:.1f}, threshold={adaptive_threshold:.1f}, fill={fill_ratio:.2f}")
    
    return fill_ratio


def _detect_answer(fill_ratios: List[float], q_label: str) -> str:
    """
    Determine which bubble (if any) is filled based on fill ratios.
    
    Smart detection that handles:
    - Printed circle outlines (low fill, ~0.10-0.30)
    - Lightly filled bubbles (medium fill, ~0.25-0.50)  
    - Dark filled bubbles (high fill, ~0.50-1.00)
    - Multiple filled bubbles (INVALID)
    """
    # Log fill ratios for debugging
    debug_info = " | ".join([
        f"{OPTION_LABELS[i]}: {fill_ratios[i]:.2f}"
        for i in range(len(OPTION_LABELS))
    ])
    logger.debug(f"  {q_label}: {debug_info}")
    
    max_fill = max(fill_ratios)
    max_idx = fill_ratios.index(max_fill)
    
    # Calculate statistics of all bubbles
    other_fills = [fill_ratios[i] for i in range(len(fill_ratios)) if i != max_idx]
    other_avg = sum(other_fills) / len(other_fills) if other_fills else 0
    other_max = max(other_fills) if other_fills else 0
    
    # Strategy 1: Strong dominance - filled bubble is much darker than others
    # This catches dark fills (0.70+) against printed circles (0.30-)
    if max_fill >= 0.50 and max_fill / (other_avg + 0.001) >= 1.5:
        selected = OPTION_LABELS[max_idx]
        logger.info(f"  {q_label}: {selected} (strong dominance, fill={max_fill:.2f}, others={other_avg:.2f})")
        return selected
    
    # Strategy 2: Moderate fill with clear separation from second-best
    # This catches lighter fills (0.30-0.70) when they're clearly the darkest
    fill_gap = max_fill - other_max
    if max_fill >= 0.25 and fill_gap >= 0.15:
        selected = OPTION_LABELS[max_idx]
        logger.info(f"  {q_label}: {selected} (clear winner by {fill_gap:.2f}, fill={max_fill:.2f})")
        return selected
    
    # Strategy 3: Check for multiple legitimate fills (INVALID)
    # Count bubbles that are both above threshold AND reasonably filled
    significant_fills = [f for f in fill_ratios if f >= 0.25]
    if len(significant_fills) > 1:
        logger.info(f"  {q_label}: INVALID ({len(significant_fills)} significant fills)")
        return INVALID
    
    # Strategy 4: Single bubble above minimum threshold
    if max_fill >= BUBBLE_FILL_THRESHOLD:
        # Check if it's at least somewhat darker than others
        if max_fill > other_avg + 0.10:
            selected = OPTION_LABELS[max_idx]
            logger.info(f"  {q_label}: {selected} (single fill, fill={max_fill:.2f})")
            return selected
    
    # Strategy 5: All bubbles are essentially empty (just printed circles)
    if max_fill < 0.25:
        logger.info(f"  {q_label}: UNATTEMPTED (max fill {max_fill:.2f}, likely printed circles only)")
        return UNATTEMPTED
    
    # Ambiguous case - not clearly filled but above threshold
    logger.info(f"  {q_label}: UNATTEMPTED (ambiguous, max={max_fill:.2f}, others={other_avg:.2f})")
    return UNATTEMPTED


def _detect_bubbles_hough(gray: np.ndarray) -> List[Tuple[int, int, int]]:
    """
    Detect bubbles using Hough Circle Transform.
    
    Returns list of (x, y, radius) tuples for detected circles.
    This method is robust against checkmarks intersecting bubbles.
    """
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (HOUGH_BLUR_KERNEL, HOUGH_BLUR_KERNEL), 0)
    
    # Detect circles using Hough Transform
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP,
        minDist=HOUGH_MIN_DIST,
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=HOUGH_MIN_RADIUS,
        maxRadius=HOUGH_MAX_RADIUS
    )
    
    if circles is None:
        logger.warning("No circles detected by Hough Transform")
        return []
    
    # Convert to integer tuples
    circles = np.round(circles[0, :]).astype("int")
    result = [(int(x), int(y), int(r)) for x, y, r in circles]
    
    logger.info(f"Hough Circle detection found {len(result)} bubbles")
    return result


def _calculate_hough_fill(gray: np.ndarray, cx: int, cy: int, radius: int) -> float:
    """
    Calculate fill ratio by examining the CENTER of the bubble only.
    
    Key insight: A filled bubble has dark pixels in the CENTER.
    An empty bubble (just printed outline) has white center with dark ring around edge.
    
    By looking at only the inner 50% of the bubble radius, we distinguish:
    - Filled bubble: center is dark (high fill ratio)
    - Empty bubble: center is white (low fill ratio)
    
    Returns fill ratio from 0.0 (empty) to 1.0 (completely filled).
    """
    h, w = gray.shape
    
    # Only examine the center 50% of the bubble (ignore outer ring where printed circle is)
    inner_radius = int(radius * 0.5)
    if inner_radius < 3:
        inner_radius = max(3, radius // 2)
    
    # Ensure coordinates are within bounds
    r = inner_radius + 2
    x1 = max(cx - r, 0)
    y1 = max(cy - r, 0)
    x2 = min(cx + r, w)
    y2 = min(cy + r, h)
    
    # Extract ROI
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    
    # Create circular mask for INNER region only
    mask = np.zeros(roi.shape, dtype=np.uint8)
    local_cx = cx - x1
    local_cy = cy - y1
    cv2.circle(mask, (local_cx, local_cy), inner_radius, 255, -1)
    
    # Count pixels inside the inner circle
    total_pixels = cv2.countNonZero(mask)
    if total_pixels == 0:
        return 0.0
    
    # Get masked ROI (center region only)
    masked_roi = cv2.bitwise_and(roi, roi, mask=mask)
    
    # Calculate mean intensity of the center
    roi_pixels = masked_roi[mask == 255]
    if roi_pixels.size == 0:
        return 0.0
    
    mean_intensity = np.mean(roi_pixels)
    
    # Simple threshold: dark center = filled, bright center = empty
    # Paper white ~200-255, pencil/ink filled ~0-100
    dark_threshold = 140
    
    # Count dark pixels in center
    _, dark_mask = cv2.threshold(masked_roi, dark_threshold, 255, cv2.THRESH_BINARY_INV)
    dark_pixels = cv2.countNonZero(cv2.bitwise_and(dark_mask, mask))
    
    fill_ratio = dark_pixels / total_pixels
    
    # Debug logging for high fill ratios
    if fill_ratio > 0.08:
        logger.debug(f"Bubble at ({cx},{cy}): mean={mean_intensity:.1f}, thresh={dark_threshold:.1f}, fill={fill_ratio:.2f}")
    
    return fill_ratio


def _cluster_bubbles_into_grid(bubbles: List[Tuple[int, int, int]], 
                                num_questions: int = QUESTIONS_PER_PART,
                                num_options: int = OPTIONS_PER_QUESTION) -> Dict[str, Dict[str, Tuple[int, int, int]]]:
    """
    Cluster detected bubbles into a question-option grid.
    
    Organizes bubbles by their Y position (questions) and X position (options).
    Returns a nested dict: {question_idx: {option_idx: (x, y, r)}}
    """
    if not bubbles:
        return {}
    
    # Sort bubbles by Y coordinate (rows)
    bubbles_by_y = sorted(bubbles, key=lambda b: b[1])
    
    # Group into rows based on Y proximity
    rows = []
    current_row = [bubbles_by_y[0]]
    
    for bubble in bubbles_by_y[1:]:
        # If Y is close to current row, add to row
        if abs(bubble[1] - current_row[0][1]) < HOUGH_MIN_DIST:
            current_row.append(bubble)
        else:
            # Sort current row by X and store
            rows.append(sorted(current_row, key=lambda b: b[0]))
            current_row = [bubble]
    
    # Don't forget the last row
    if current_row:
        rows.append(sorted(current_row, key=lambda b: b[0]))
    
    # Organize into grid structure
    grid = {}
    for q_idx, row in enumerate(rows[:num_questions]):
        grid[q_idx] = {}
        for opt_idx, bubble in enumerate(row[:num_options]):
            grid[q_idx][opt_idx] = bubble
    
    return grid


def _find_bubble_near(bubbles: List[Tuple[int, int, int]], 
                       target_x: int, target_y: int, 
                       search_radius: int = 30) -> Tuple[int, int, int]:
    """
    Find a detected bubble near the expected template position.
    Returns the bubble (x, y, r) closest to target, or None if none within search_radius.
    """
    best_bubble = None
    best_dist = float('inf')
    
    for (x, y, r) in bubbles:
        dist = np.sqrt((x - target_x)**2 + (y - target_y)**2)
        if dist < search_radius and dist < best_dist:
            best_dist = dist
            best_bubble = (x, y, r)
    
    return best_bubble


def _read_part_clustered(gray: np.ndarray, bubbles: List[Tuple[int, int, int]], 
                          part_name: str) -> Dict[str, str]:
    """
    Read answers by clustering detected bubbles into a grid.
    
    Uses k-means clustering to group bubbles into 8 rows (questions) 
    and 4 columns (options A-D) per row.
    """
    answers = {}
    
    logger.info(f"Processing {part_name} with bubble clustering...")
    
    # Cluster bubbles into grid
    grid = _cluster_bubbles_to_grid(bubbles, num_rows=8, num_cols=4)
    
    if not grid:
        logger.error(f"{part_name}: Failed to cluster bubbles into grid")
        return {f"Q{str(i+1).zfill(2)}": UNATTEMPTED for i in range(8)}
    
    for q_idx in range(8):
        q_label = f"Q{str(q_idx + 1).zfill(2)}"
        
        fill_ratios = []
        for opt_idx in range(4):
            if q_idx in grid and opt_idx in grid[q_idx]:
                bx, by, br = grid[q_idx][opt_idx]
                fill_ratio = _calculate_hough_fill(gray, bx, by, br)
            else:
                # Missing bubble - assume empty
                fill_ratio = 0.0
            
            fill_ratios.append(fill_ratio)
        
        answer = _detect_answer(fill_ratios, q_label)
        answers[q_label] = answer
    
    return answers


def _read_part(gray: np.ndarray, x_cols: List[int], part_name: str, 
               scale_x: float = 1.0, scale_y: float = 1.0) -> Dict[str, str]:
    """
    Read all 8 questions for one part (Part-I or Part-II).
    Scales template coordinates to match actual image dimensions.
    """
    answers = {}
    
    logger.info(f"Processing {part_name}...")
    
    # Scale the bubble radius based on image size
    scaled_radius = int(BUBBLE_RADIUS * min(scale_x, scale_y))
    
    for q_idx, cy_template in enumerate(ROW_Y):
        fill_ratios = []
        
        # Scale Y coordinate to actual image size
        cy = int(cy_template * scale_y)
        
        for cx_template in x_cols:
            # Scale X coordinate to actual image size
            cx = int(cx_template * scale_x)
            
            # Calculate fill ratio for this bubble using scaled radius
            fill_ratio = _calculate_bubble_fill(gray, cx, cy, scaled_radius)
            fill_ratios.append(fill_ratio)
        
        q_label = f"Q{str(q_idx + 1).zfill(2)}"
        answer = _detect_answer(fill_ratios, q_label)
        answers[q_label] = answer
    
    return answers


def _cluster_bubbles_to_grid(bubbles: List[Tuple[int, int, int]], 
                              num_rows: int = 8,
                              num_cols: int = 4) -> Dict[int, Dict[int, Tuple[int, int, int]]]:
    """
    Cluster detected bubbles into a grid of rows and columns.
    
    Uses k-means clustering on Y coordinates to find rows,
    then sorts each row by X to assign columns (A, B, C, D).
    
    Returns: {row_idx: {col_idx: (x, y, r)}}
    """
    if len(bubbles) < num_rows * num_cols // 2:
        logger.warning(f"Not enough bubbles ({len(bubbles)}) for grid clustering")
        return {}
    
    # Extract Y coordinates for row clustering
    y_coords = np.array([b[1] for b in bubbles])
    
    # Simple k-means-like clustering for rows
    # Sort Y and find clusters
    y_sorted = np.sort(y_coords)
    
    # Find row centers by looking for groups of bubbles
    # Expect roughly equal spacing between rows
    min_y, max_y = y_sorted[0], y_sorted[-1]
    expected_spacing = (max_y - min_y) / (num_rows - 1) if num_rows > 1 else 30
    
    # Initialize row centers evenly distributed
    row_centers = np.linspace(min_y + expected_spacing/2, max_y - expected_spacing/2, num_rows)
    
    # Iteratively refine row centers
    for _ in range(5):  # 5 iterations should converge
        row_assignments = [[] for _ in range(num_rows)]
        
        for y in y_sorted:
            # Find closest row center
            distances = [abs(y - rc) for rc in row_centers]
            closest = distances.index(min(distances))
            row_assignments[closest].append(y)
        
        # Update row centers to median of assigned bubbles
        for i in range(num_rows):
            if row_assignments[i]:
                row_centers[i] = np.median(row_assignments[i])
    
    # Now assign each bubble to a row and column
    grid = {}
    for i in range(num_rows):
        grid[i] = {}
    
    # For each bubble, find closest row
    for bubble in bubbles:
        bx, by, br = bubble
        distances = [abs(by - rc) for rc in row_centers]
        row_idx = distances.index(min(distances))
        
        if row_idx not in grid:
            grid[row_idx] = {}
        
        # Store bubble in this row (will sort by X later)
        if 'bubbles' not in grid[row_idx]:
            grid[row_idx]['bubbles'] = []
        grid[row_idx]['bubbles'].append(bubble)
    
    # Sort each row by X and assign columns
    for row_idx in grid:
        if 'bubbles' in grid[row_idx]:
            row_bubbles = sorted(grid[row_idx]['bubbles'], key=lambda b: b[0])
            # Assign to columns 0-3 (A-D)
            for col_idx, bubble in enumerate(row_bubbles[:num_cols]):
                grid[row_idx][col_idx] = bubble
            del grid[row_idx]['bubbles']
    
    return grid


def read_bubble_sheet(color_image: np.ndarray,
                      thresh_image: np.ndarray = None,
                      use_hough: bool = None) -> Dict[str, Dict[str, str]]:
    """
    Main entry point for Task 3.
    
    Args:
        color_image: The warped color image (BGR format)
        thresh_image: Optional pre-thresholded image (not used in this implementation)
        use_hough: Override USE_HOUGH_CIRCLES config. If None, uses config value.
    
    Returns:
        Dictionary with 'part1' and 'part2' containing answer dictionaries
    """
    logger.info("=" * 50)
    logger.info("Starting Bubble Sheet Reading...")
    logger.info("=" * 50)
    
    # Determine detection method
    hough_mode = USE_HOUGH_CIRCLES if use_hough is None else use_hough
    
    if hough_mode:
        logger.info("Using Hough Circle Transform with auto-calibration")
    else:
        logger.info("Using template-based bubble detection with coordinate scaling")
    
    # Convert to grayscale for analysis
    if len(color_image.shape) == 3:
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = color_image.copy()
    
    # Get actual image dimensions
    h, w = gray.shape
    
    # Calculate scale factors to map template coordinates to actual image
    scale_x = w / TEMPLATE_W
    scale_y = h / TEMPLATE_H
    
    if h != TEMPLATE_H or w != TEMPLATE_W:
        logger.info(f"Image dimensions ({w}x{h}) differ from template ({TEMPLATE_W}x{TEMPLATE_H})")
        logger.info(f"Base scale factors: x={scale_x:.3f}, y={scale_y:.3f}")
    
    # Process both parts using selected method
    if hough_mode:
        # Detect all bubbles using Hough Transform
        all_bubbles = _detect_bubbles_hough(gray)
        
        # Define X ranges for each part (scaled)
        mid_x_scaled = int((TEMPLATE_W // 2) * scale_x)
        margin = int(50 * scale_x)
        
        # Filter bubbles by part
        part1_bubbles = [b for b in all_bubbles if b[0] < mid_x_scaled - margin]
        part2_bubbles = [b for b in all_bubbles if b[0] > mid_x_scaled + margin]
        
        logger.info(f"Part-I: {len(part1_bubbles)} bubbles, Part-II: {len(part2_bubbles)} bubbles")
        
        # Cluster bubbles into grid and read answers
        part1 = _read_part_clustered(gray, part1_bubbles, "Part-I")
        part2 = _read_part_clustered(gray, part2_bubbles, "Part-II")
    else:
        # Use template-based detection with scaled coordinates
        part1 = _read_part(gray, PART1_X, "Part-I", scale_x, scale_y)
        part2 = _read_part(gray, PART2_X, "Part-II", scale_x, scale_y)
    
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
    COLOR_FILLED = (0, 255, 0)        # Green for detected
    COLOR_UNATTEMPTED = (128, 128, 128)  # Gray for empty
    COLOR_INVALID = (0, 165, 255)     # Orange for invalid
    
    for part_idx, (part_name, x_cols) in enumerate([("part1", PART1_X), ("part2", PART2_X)]):
        part_answers = answers.get(part_name, {})
        
        for q_idx, cy in enumerate(ROW_Y):
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
                    color = (255, 255, 255)  # White for other options
                    thickness = 1
                
                # Draw circle
                cv2.circle(vis_image, (cx, cy), BUBBLE_RADIUS, color, thickness)
    
    if output_path:
        cv2.imwrite(output_path, vis_image)
    
    return vis_image


def visualize_hough_detection(color_image: np.ndarray, 
                              bubbles: List[Tuple[int, int, int]],
                              filled_indices: List[int] = None,
                              output_path: str = None) -> np.ndarray:
    """
    Visualize Hough Circle detection results.
    
    Args:
        color_image: The input color image
        bubbles: List of (x, y, radius) tuples from Hough detection
        filled_indices: Indices of bubbles that are filled
        output_path: Optional path to save visualization
    
    Returns:
        Visualization image with detected circles
    """
    vis_image = color_image.copy()
    filled_indices = filled_indices or []
    
    # Colors
    COLOR_DETECTED = (0, 255, 0)      # Green outline for detected
    COLOR_FILLED = (0, 0, 255)        # Red center for filled
    
    for idx, (x, y, r) in enumerate(bubbles):
        # Draw circle outline
        cv2.circle(vis_image, (x, y), r, COLOR_DETECTED, 2)
        
        # If filled, draw red center dot
        if idx in filled_indices:
            cv2.circle(vis_image, (x, y), 5, COLOR_FILLED, -1)
        
        # Draw circle index for debugging
        cv2.putText(vis_image, str(idx), (x - 10, y - r - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_DETECTED, 1)
    
    # Add summary text
    summary = f"Detected: {len(bubbles)}, Filled: {len(filled_indices)}"
    cv2.putText(vis_image, summary, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    if output_path:
        cv2.imwrite(output_path, vis_image)
    
    return vis_image


def calibrate_bubble_positions(image_path: str) -> Dict[str, List[Tuple[int, int]]]:
    """
    Calibrate bubble positions using Hough Circle detection.
    
    Run this on a reference image with all bubbles filled to discover
    the correct coordinates for your template.
    
    Returns:
        Dictionary with 'part1_x', 'part2_x', 'row_y' coordinates
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect all bubbles
    bubbles = _detect_bubbles_hough(gray)
    
    if not bubbles:
        logger.error("No bubbles detected! Check Hough parameters in config.py")
        return {}
    
    # Sort by Y coordinate
    bubbles_sorted = sorted(bubbles, key=lambda b: b[1])
    
    # Group into rows
    rows = []
    current_row = [bubbles_sorted[0]]
    
    for bubble in bubbles_sorted[1:]:
        if abs(bubble[1] - current_row[0][1]) < HOUGH_MIN_DIST:
            current_row.append(bubble)
        else:
            rows.append(sorted(current_row, key=lambda b: b[0]))
            current_row = [bubble]
    
    if current_row:
        rows.append(sorted(current_row, key=lambda b: b[0]))
    
    # Find midpoint to separate parts
    all_x = [b[0] for b in bubbles]
    mid_x = (min(all_x) + max(all_x)) // 2
    
    # Extract coordinates
    part1_x = []
    part2_x = []
    row_y = []
    
    for row in rows[:QUESTIONS_PER_PART]:
        if row:
            row_y.append(row[0][1])
            
            # Split into parts based on X position
            for bubble in row:
                x = bubble[0]
                # Take only first occurrence of each X position
                if x < mid_x and x not in part1_x and len(part1_x) < OPTIONS_PER_QUESTION:
                    part1_x.append(x)
                elif x >= mid_x and x not in part2_x and len(part2_x) < OPTIONS_PER_QUESTION:
                    part2_x.append(x)
    
    result = {
        'part1_x': sorted(part1_x),
        'part2_x': sorted(part2_x),
        'row_y': row_y
    }
    
    logger.info("Calibration results:")
    logger.info(f"  Part 1 X: {result['part1_x']}")
    logger.info(f"  Part 2 X: {result['part2_x']}")
    logger.info(f"  Row Y: {result['row_y']}")
    
    return result
