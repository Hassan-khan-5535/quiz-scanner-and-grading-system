"""
image_processor.py — Image Cleanup Pipeline
=============================================

PURPOSE:
    Takes a raw image from the camera/scanner and "cleans it up"
    so the AI modules (QR, OCR, Bubbles) can read it easily.

KEY FIXES:
    1. Rotation (forces portrait mode)
    2. Document Alignment (Smart crop + resize to 1240x1755)
    3. Grayscale (removes color complexity)
    4. Blurring (removes noise)
    5. Adaptive Thresholding (fixes shadows/uneven lighting)
"""

import cv2
import numpy as np

from src.utils.logger import get_logger
from src.utils.helpers import save_debug_image
from config import BLUR_KERNEL_SIZE, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_CONSTANT

logger = get_logger(__name__)

# Template dimensions (the reference scan resolution)
TEMPLATE_W = 1240
TEMPLATE_H = 1755


def fix_rotation(image: np.ndarray) -> np.ndarray:
    """
    Checks if the image is in landscape mode (width > height).
    If it is, rotates it 90 degrees to make it portrait.
    """
    height, width = image.shape[:2]
    
    if width > height:
        logger.info("Image is landscape (width > height). Rotating 90 degrees clockwise.")
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        return rotated
    
    return image


def order_points(pts):
    """Orders 4 points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left has smallest sum
    rect[2] = pts[np.argmax(s)]   # bottom-right has largest sum
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # top-right has smallest difference
    rect[3] = pts[np.argmax(diff)] # bottom-left has largest difference
    return rect


def find_document_contour(image: np.ndarray):
    """
    Tries to find the quiz paper boundary using contour detection.
    Returns the 4 corner points if found, else None.
    
    Uses multiple edge-detection strategies to handle various backgrounds.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    image_area = h * w
    
    strategies = []
    
    # Strategy 1: Standard Canny
    blurred1 = cv2.GaussianBlur(gray, (5, 5), 0)
    edged1 = cv2.Canny(blurred1, 50, 150)
    strategies.append(("canny_standard", edged1))
    
    # Strategy 2: Stronger blur + wider Canny range
    blurred2 = cv2.GaussianBlur(gray, (7, 7), 0)
    edged2 = cv2.Canny(blurred2, 30, 120)
    strategies.append(("canny_wide", edged2))
    
    # Strategy 3: Morphological closing to connect broken edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edged1, cv2.MORPH_CLOSE, kernel, iterations=2)
    strategies.append(("canny_closed", closed))
    
    # Strategy 4: Adaptive threshold based edge detection
    adaptive = cv2.adaptiveThreshold(blurred1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
    strategies.append(("adaptive", adaptive))
    
    for name, edge_img in strategies:
        contours, _ = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue
            
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        
        for c in contours:
            area = cv2.contourArea(c)
            # The paper should be at least 40% of the image area
            if area < image_area * 0.40:
                continue
                
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            if len(approx) == 4:
                pts = order_points(approx.reshape(4, 2))
                # Validate: the warped rectangle should have reasonable aspect ratio
                w_rect = np.linalg.norm(pts[1] - pts[0])
                h_rect = np.linalg.norm(pts[3] - pts[0])
                aspect = h_rect / w_rect if w_rect > 0 else 0
                # A4 paper aspect ratio is ~1.414. Allow range 1.1 to 1.8
                if 1.1 < aspect < 1.8:
                    logger.info(f"Document contour found using strategy: '{name}' (area={area/image_area:.1%})")
                    return pts
    
    return None


def warp_document(image: np.ndarray, width=TEMPLATE_W, height=TEMPLATE_H) -> np.ndarray:
    """
    Aligns the document to the template dimensions.
    
    Strategy:
    1. Try to find the paper contour and do a perspective warp
    2. If that fails, simply resize the image (works when paper fills the frame)
    """
    pts = find_document_contour(image)
    
    if pts is not None:
        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]], dtype="float32")
            
        M = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(image, M, (width, height))
        logger.info("Document perspective warp successful.")
        return warped
    else:
        # Fallback: the paper likely fills the frame already.
        # Just resize to template dimensions.
        logger.info("No document contour found. Using direct resize to template dimensions.")
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def preprocess_image(image: np.ndarray, debug_dir: str = None, filename: str = "doc") -> dict:
    """
    The main preprocessing pipeline.
    """
    logger.info("Starting image preprocessing pipeline...")
    
    # 1. Fix Rotation
    fixed_image = fix_rotation(image)
    
    # Save the original (un-warped) grayscale for QR decoding
    original_gray = cv2.cvtColor(fixed_image, cv2.COLOR_BGR2GRAY)
    
    # 2. Warp/Resize Document to standard 1240x1755 template size
    warped_image = warp_document(fixed_image, width=TEMPLATE_W, height=TEMPLATE_H)
    
    if debug_dir:
        save_debug_image(warped_image, f"{filename}_01_warped", debug_dir)

    # 3. Grayscale (of the warped image, for bubble reading)
    gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    if debug_dir:
        save_debug_image(gray, f"{filename}_02_grayscale", debug_dir)

    # 4. Blur
    blurred = cv2.GaussianBlur(gray, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), 0)
    if debug_dir:
        save_debug_image(blurred, f"{filename}_03_blurred", debug_dir)

    # 5. Adaptive Threshold
    thresh = cv2.adaptiveThreshold(
        blurred, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        ADAPTIVE_THRESH_BLOCK_SIZE, 
        ADAPTIVE_THRESH_CONSTANT
    )
    
    if debug_dir:
        save_debug_image(thresh, f"{filename}_04_thresholded", debug_dir)
        
    logger.info("Preprocessing complete.")
    
    return {
        "color": warped_image,
        "gray": gray,
        "thresh": thresh,
        "original_gray": original_gray
    }
