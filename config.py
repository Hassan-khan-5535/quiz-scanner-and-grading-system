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
"""

import os

# =============================================================
# PROJECT PATHS
# =============================================================
# os.path.dirname(__file__) gives us the folder where THIS file lives
# This makes paths work regardless of where you run the script from

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
# WHY: We count dark pixels inside each bubble circle.
#       filled_ratio = dark_pixels / total_pixels_in_bubble
#       If ratio > threshold, the bubble is "filled".
# 0.4 means 40% of the bubble area must be dark ink.
BUBBLE_FILL_THRESHOLD = 0.4

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

# EasyOCR languages to use
OCR_LANGUAGES = ["en"]

# Whether to use GPU for OCR (False = CPU only, safer for most setups)
OCR_USE_GPU = False

# =============================================================
# LOGGING SETTINGS
# =============================================================

# Log level: "DEBUG", "INFO", "WARNING", "ERROR"
LOG_LEVEL = "INFO"

# Whether to save debug images at each processing step
SAVE_DEBUG_IMAGES = True
