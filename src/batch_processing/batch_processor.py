"""
batch_processor.py — Task 5 (Part B): Batch Processing Orchestration
======================================================================

PURPOSE:
    Loops through a folder of images, running the entire AI pipeline
    (Tasks 1-4) on each one, and passes the results to the report generator.
"""

import os
from typing import List, Dict, Any

from src.utils.logger import get_logger
from src.utils.helpers import load_image, get_image_files
from config import DEBUG_DIR, SAVE_DEBUG_IMAGES

# Import our AI pipeline modules
from src.preprocessing.image_processor import preprocess_image
from src.qr_decoder.qr_reader import decode_answer_key
from src.ocr.text_extractor import extract_student_info
from src.bubble_detection.bubble_reader import read_bubble_sheet
from src.grading.grader import grade_quiz
from src.reporting.report_generator import generate_report

logger = get_logger(__name__)


def process_single_image(image_path: str, answer_key_cache: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Runs the entire pipeline (Tasks 1-4) on a single image.
    
    Args:
        image_path: Path to the image file.
        answer_key_cache: Optional cached AnswerKey. If the QR code can't be read on one 
                          student's sheet, we can reuse the key from a previous sheet if 
                          they are in the same batch/set.
                          
    Returns:
        Dictionary containing all extracted data (name, answers, grade).
    """
    filename = os.path.basename(image_path)
    logger.info(f"--- Processing {filename} ---")
    
    # 0. Load Image
    image = load_image(image_path)
    if image is None:
        return {"filename": filename, "error": "Could not load image"}
        
    debug_dir = DEBUG_DIR if SAVE_DEBUG_IMAGES else None
    name_no_ext = os.path.splitext(filename)[0]
    
    try:
        # 1. Preprocessing
        preprocessed = preprocess_image(image, debug_dir, name_no_ext)
        
        # 2. Task 1: QR Decoder
        answer_key = decode_answer_key(preprocessed["gray"])
        
        if answer_key is None:
            # Fallback to cache if available
            if answer_key_cache:
                logger.warning(f"QR decode failed for {filename}. Using cached Answer Key.")
                answer_key = answer_key_cache
            else:
                return {"filename": filename, "error": "QR decode failed and no cache available"}
                
        # 3. Task 2: OCR Student Info
        student_info = extract_student_info(preprocessed["gray"])
        
        # 4. Task 3: Bubble Reader
        student_answers = read_bubble_sheet(preprocessed["color"], preprocessed["thresh"])
        
        # 5. Task 4: Grading
        grade = grade_quiz(student_answers, answer_key)
        
        logger.info(f"--- Finished processing {filename} successfully ---")
        
        return {
            "filename": filename,
            "answer_key": answer_key,
            "student_info": student_info,
            "student_answers": student_answers,
            "grade": grade
        }
        
    except Exception as e:
        logger.error(f"Error processing {filename}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"filename": filename, "error": str(e)}


def process_batch(folder_path: str) -> str:
    """
    Main function for Task 5. 
    Finds all images in a folder, processes them, and generates an Excel report.
    
    Returns:
        Path to the generated report file.
    """
    logger.info(f"Starting batch processing on folder: {folder_path}")
    
    image_files = get_image_files(folder_path)
    
    if not image_files:
        logger.error("No valid image files found in folder.")
        return ""
        
    all_results = []
    
    # We cache the answer key from the first successful read.
    # Why? If student #4's QR code is smudged, we assume they are taking 
    # the same quiz as students #1-3 in this batch.
    current_answer_key = None
    
    # Process images one by one
    for index, image_path in enumerate(image_files):
        logger.info(f"Processing image {index + 1} of {len(image_files)}...")
        
        result = process_single_image(image_path, current_answer_key)
        all_results.append(result)
        
        # Update cache if we successfully read a QR code
        if "answer_key" in result and current_answer_key is None:
            current_answer_key = result["answer_key"]
            
    # Generate the final report
    report_path = generate_report(all_results, output_format="xlsx")
    
    logger.info(f"Batch processing complete! Processed {len(image_files)} files.")
    return report_path
