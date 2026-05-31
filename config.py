"""
config.py — Global Configuration for Quiz Scanner
===================================================

PURPOSE:
    Central place for ALL configurable values.
    Change settings HERE, not scattered across code files.

WHY:
    - Avoids "magic numbers" buried in code
    - Easy to tune parameters during testing
    - Easy to explain in viva: "All settings are in config.py"

TUNING GUIDE:
    If bubble detection is not working:
    1. Enable SAVE_DEBUG_IMAGES = True
    2. Check debug images in output/debug/
    3. Adjust BUBBLE_FILL_THRESHOLD up/down if bubbles are missed/falsely detected
    4. Adjust DOMINANCE_RATIO if multiple bubbles are being detected
"""

import os

# =============================================================
# PROJECT PATHS
# =============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Input/Output directories
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "samples")
SAMPLES_SINGLE_DIR = os.path.join(SAMPLES_DIR, "single")
SAMPLES_BATCH_DIR = os.path.join(SAMPLES_DIR, "batch")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
DEBUG_DIR = os.path.join(OUTPUT_DIR, "debug")

# =============================================================
# IMAGE PREPROCESSING SETTINGS
# =============================================================

# Gaussian blur kernel size (must be odd number)
# WHY: Blur reduces noise before thresholding. Larger = more blur.
# 5 is a good balance — removes noise without losing bubble edges.
BLUR_KERNEL_SIZE = 5

# Adaptive threshold block size (must be odd number)
# WHY: Adaptive thresholding looks at a local neighborhood of pixels
# to decide if each pixel is "black" or "white".
# 11 means it looks at an 11x11 pixel neighborhood.
ADAPTIVE_THRESH_BLOCK_SIZE = 11

# Adaptive threshold constant (subtracted from the mean)
# WHY: This fine-tunes the threshold. Higher = more pixels become white.
ADAPTIVE_THRESH_CONSTANT = 2

# =============================================================
# BUBBLE DETECTION SETTINGS
# =============================================================

# Minimum fill ratio to consider a bubble as "filled"
# Range: 0.0 to 1.0 (0% to 100% of bubble area)
# 
# TUNING:
# - INCREASE if empty bubbles are being detected as filled (false positives)
# - DECREASE if filled bubbles are being missed (false negatives)
# - Typical range: 0.25 to 0.50
# - Default: 0.35 (35% of bubble area must be dark)
BUBBLE_FILL_THRESHOLD = 0.35

# Minimum confidence score for a bubble to be considered filled
# This is a weighted combination of fill ratio, darkness, and contour analysis
# Range: 0.0 to 1.0
# 
# TUNING:
# - INCREASE for stricter detection (fewer false positives)
# - DECREASE for more sensitive detection (fewer false negatives)
# - Default: 0.35
BUBBLE_CONFIDENCE_THRESHOLD = 0.35

# Dominance ratio - filled bubble must be this many times darker than others
# 
# TUNING:
# - INCREASE if partially filled bubbles are being detected
# - DECREASE if lightly filled bubbles are being missed
# - Default: 1.5 (filled bubble is 1.5x darker than average of others)
DOMINANCE_RATIO = 1.5

# Number of questions per part
QUESTIONS_PER_PART = 8

# Number of options per question (A, B, C, D)
OPTIONS_PER_QUESTION = 4

# =============================================================
# GRADING SETTINGS
# =============================================================

# Marks per correct answer
MARKS_PER_CORRECT = 1

# Marks deducted per wrong answer (0 = no negative marking)
NEGATIVE_MARKING = 0

# Marks for unattempted questions
MARKS_UNATTEMPTED = 0

# Letter grade boundaries (percentage)
GRADE_BOUNDARIES = {
    "A": 90,   # 90% and above
    "B": 80,   # 80-89%
    "C": 70,   # 70-79%
    "D": 60,   # 60-69%
    "F": 0,    # Below 60%
}

# =============================================================
# OCR SETTINGS
# =============================================================

from dotenv import load_dotenv
# Load environment variables from .env file for the Gemini API Key
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# Google Gemini model to use for handwriting recognition
GEMINI_MODEL_NAME = "gemini-2.5-flash"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# =============================================================
# LOGGING SETTINGS
# =============================================================

# Log level: "DEBUG", "INFO", "WARNING", "ERROR"
# DEBUG: Shows detailed bubble fill values and detection logic
# INFO: Shows processing progress and final results
# WARNING: Only shows warnings and errors
LOG_LEVEL = "INFO"

# Whether to save debug images at each processing step
# Set to True when tuning or troubleshooting
SAVE_DEBUG_IMAGES = True

# =============================================================
# TEMPLATE COORDINATES (for reference)
# =============================================================
# These are calibrated for 1240x1755 scanned sheets
# Only change if your quiz sheet template is different

TEMPLATE_WIDTH = 1240
TEMPLATE_HEIGHT = 1755

# Part 1 bubble X coordinates (left side) - columns A, B, C, D
PART1_X_COORDS = [289, 369, 449, 535]

# Part 2 bubble X coordinates (right side) - columns A, B, C, D  
PART2_X_COORDS = [721, 799, 877, 961]

# Question row Y coordinates (Q01-Q08)
ROW_Y_COORDS = [398, 432, 466, 500, 534, 568, 604, 638]
