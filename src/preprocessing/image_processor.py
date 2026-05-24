"""
image_processor.py — Image Cleanup Pipeline
=============================================

PURPOSE:
    Takes a raw image from the camera/scanner and "cleans it up"
    so the AI modules (QR, OCR, Bubbles) can read it easily.

KEY FIXES:
    1. Rotation (forces portrait mode)
    2. Grayscale (removes color complexity)
    3. Blurring (removes noise)
    4. Adaptive Thresholding (fixes shadows/uneven lighting)
"""

import cv2
import numpy as np

from src.utils.logger import get_logger
from src.utils.helpers import save_debug_image
from config import BLUR_KERNEL_SIZE, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_CONSTANT

logger = get_logger(__name__)

def fix_rotation(image: np.ndarray) -> np.ndarray:
    """
    Checks if the image is in landscape mode (width > height).
    If it is, rotates it 90 degrees to make it portrait.
    
    WHY: Quiz sheets are printed on A4 paper (portrait). If someone
    takes a photo sideways, our coordinates will be completely wrong.
    """
    height, width = image.shape[:2]
    
    if width > height:
        logger.info("Image is landscape (width > height). Rotating 90 degrees clockwise.")
        # cv2.ROTATE_90_CLOCKWISE is a built-in OpenCV constant
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        return rotated
    
    return image

def preprocess_image(image: np.ndarray, debug_dir: str = None, filename: str = "doc") -> dict:
    """
    The main preprocessing pipeline.
    
    Takes a raw color image and returns a dictionary containing multiple
    versions of the image (color, grayscale, thresholded) because different
    tasks need different versions. (e.g., QR reader prefers grayscale, 
    bubble reader prefers thresholded).
    
    Args:
        image: The raw input image (NumPy array)
        debug_dir: If provided, saves intermediate steps here
        filename: Base name for debug images (e.g., "quiz1")
        
    Returns:
        dict containing:
            - 'original': The rotation-fixed color image
            - 'gray': The grayscale version
            - 'thresh': The black-and-white thresholded version
    """
    logger.info("Starting image preprocessing pipeline...")
    
    # ---------------------------------------------------------
    # STEP 1: Fix Rotation
    # ---------------------------------------------------------
    fixed_image = fix_rotation(image)
    if debug_dir:
        save_debug_image(fixed_image, f"{filename}_01_rotated", debug_dir)

    # ---------------------------------------------------------
    # STEP 2: Grayscale Conversion
    # ---------------------------------------------------------
    # cv2.cvtColor changes color spaces. We go from BGR (OpenCV's default) to GRAY
    gray = cv2.cvtColor(fixed_image, cv2.COLOR_BGR2GRAY)
    if debug_dir:
        save_debug_image(gray, f"{filename}_02_grayscale", debug_dir)

    # ---------------------------------------------------------
    # STEP 3: Blur (Noise Reduction)
    # ---------------------------------------------------------
    # GaussianBlur averages pixels with their neighbors.
    # The kernel size (e.g., 5x5) determines how much to blur.
    # We do this so tiny specs of dust don't become black dots in step 4.
    blurred = cv2.GaussianBlur(gray, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), 0)
    if debug_dir:
        save_debug_image(blurred, f"{filename}_03_blurred", debug_dir)

    # ---------------------------------------------------------
    # STEP 4: Adaptive Thresholding
    # ---------------------------------------------------------
    # This is the most important step for handling shadows!
    # Instead of a global threshold (e.g., pixel < 127 = black), it looks at a 
    # small block (e.g., 11x11) and calculates the threshold for just that block.
    # 
    # Parameters:
    # 1. 255: The maximum value (white)
    # 2. ADAPTIVE_THRESH_GAUSSIAN_C: Uses a weighted sum of neighborhood values
    # 3. THRESH_BINARY_INV: Inverts the result (black background, white text/lines).
    #    *We use inverted because OpenCV contour detection looks for WHITE objects
    #    on a BLACK background.*
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
        "color": fixed_image,
        "gray": gray,
        "thresh": thresh
    }
