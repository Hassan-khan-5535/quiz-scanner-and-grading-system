"""
helpers.py — Common Helper Functions
======================================

PURPOSE:
    Small utility functions used by multiple modules.
    Avoids duplicating the same code across different files.

FUNCTIONS:
    load_image(path)         → Loads an image from file path
    save_debug_image(...)    → Saves intermediate processing images for debugging
    ensure_directory(path)   → Creates a directory if it doesn't exist
    get_image_files(folder)  → Lists all image files in a folder
"""

import os
from typing import Optional

import cv2
import numpy as np

from src.utils.logger import get_logger
from src.utils.constants import SUPPORTED_EXTENSIONS

# Create a logger for this module
logger = get_logger(__name__)


def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load an image from the given file path.

    WHY A SEPARATE FUNCTION?
        - Centralizes error handling (what if the file doesn't exist?)
        - Adds logging so we can see what's being loaded
        - Returns None instead of crashing if the file is bad

    Args:
        image_path: Full path to the image file (e.g., "samples/single/quiz1.jpg")

    Returns:
        The image as a NumPy array (OpenCV format: BGR color), or None if loading failed.

    ABOUT OPENCV IMAGE FORMAT:
        OpenCV loads images as NumPy arrays with shape (height, width, channels).
        Colors are in BGR order (Blue, Green, Red), NOT RGB.
        Example: A 1920x1080 color image has shape (1080, 1920, 3).
    """
    # Check if the file actually exists before trying to load it
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        return None

    # cv2.imread() reads the image file and converts it to a NumPy array
    # It returns None if the file format is unsupported or corrupted
    image = cv2.imread(image_path)

    if image is None:
        logger.error(f"Failed to load image (corrupted or unsupported format): {image_path}")
        return None

    logger.info(f"Loaded image: {os.path.basename(image_path)} "
                f"(size: {image.shape[1]}x{image.shape[0]} pixels)")
    return image


def save_debug_image(image: np.ndarray, stage_name: str, debug_dir: str) -> str:
    """
    Save an intermediate processing image for debugging/visualization.

    WHY THIS IS USEFUL:
        When something goes wrong (e.g., bubbles not detected), we can look at
        the debug images to see what happened at each processing stage.
        Example: "threshold.png" shows the black/white version — if it looks
        wrong, we know to adjust the threshold settings.

    Args:
        image:      The image to save (NumPy array)
        stage_name: Descriptive name like "01_grayscale" or "03_threshold"
        debug_dir:  Directory to save into (e.g., "output/debug/")

    Returns:
        The full path where the image was saved.
    """
    # Make sure the debug directory exists
    ensure_directory(debug_dir)

    # Build the full file path
    file_path = os.path.join(debug_dir, f"{stage_name}.png")

    # Save the image
    cv2.imwrite(file_path, image)
    logger.debug(f"Saved debug image: {stage_name}.png")

    return file_path


def ensure_directory(dir_path: str) -> None:
    """
    Create a directory (and all parent directories) if it doesn't already exist.

    WHY:
        We save output files to directories that might not exist yet.
        os.makedirs() creates them. exist_ok=True means "don't crash if it already exists".

    Args:
        dir_path: Path to the directory to create.
    """
    os.makedirs(dir_path, exist_ok=True)


def get_image_files(folder_path: str) -> list[str]:
    """
    Get a sorted list of all image file paths in a folder.

    WHY:
        Used by the batch processor to find all quiz images in a folder.
        Filters to only supported formats (JPG, PNG) so we don't accidentally
        try to process a .txt file.

    Args:
        folder_path: Path to the folder to scan.

    Returns:
        List of full file paths to image files, sorted alphabetically.
    """
    if not os.path.isdir(folder_path):
        logger.error(f"Directory not found: {folder_path}")
        return []

    image_files = []
    for filename in sorted(os.listdir(folder_path)):
        # Get the file extension and check if it's a supported image format
        # os.path.splitext("quiz1.jpg") returns ("quiz1", ".jpg")
        _, extension = os.path.splitext(filename)

        if extension.lower() in SUPPORTED_EXTENSIONS:
            full_path = os.path.join(folder_path, filename)
            image_files.append(full_path)

    logger.info(f"Found {len(image_files)} image(s) in {folder_path}")
    return image_files
