"""
grader.py — Task 4: Quiz Grading Logic
========================================

PURPOSE:
    Compares the student's extracted answers against the decoded answer key,
    calculates scores, and generates a per-question breakdown.

ASSIGNMENT REQUIREMENTS MET:
    - Compare bubble-by-bubble against decoded key
    - Count correct, incorrect, unattempted
    - Calculate total marks (handle optional negative marking)
    - Generate per-question breakdown: tick (✓), cross (✗), dash (—)
    - Calculate final score and percentage
"""

from typing import Dict, Any

from src.utils.logger import get_logger
from src.utils.constants import PART_LABELS, SYMBOL_CORRECT, SYMBOL_INCORRECT, SYMBOL_UNATTEMPTED, SYMBOL_INVALID, UNATTEMPTED, INVALID
from config import MARKS_PER_CORRECT, NEGATIVE_MARKING, MARKS_UNATTEMPTED, GRADE_BOUNDARIES

logger = get_logger(__name__)


def calculate_letter_grade(percentage: float) -> str:
    """
    Converts a percentage score into a standard Letter Grade.
    Uses the GRADE_BOUNDARIES defined in config.py.
    """
    if percentage >= GRADE_BOUNDARIES["A"]: return "A"
    if percentage >= GRADE_BOUNDARIES["B"]: return "B"
    if percentage >= GRADE_BOUNDARIES["C"]: return "C"
    if percentage >= GRADE_BOUNDARIES["D"]: return "D"
    return "F"


def grade_quiz(student_answers: Dict[str, Dict[str, str]], 
               answer_key: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function for Task 4. Compares student answers to the key.
    
    Args:
        student_answers: Output from Task 3 (bubble_reader.py)
        answer_key: Output from Task 1 (qr_reader.py)
        
    Returns:
        GradeReport dictionary containing scores and visual breakdown.
    """
    logger.info("Starting grading process...")
    
    # Initialize counters
    correct_count = 0
    incorrect_count = 0
    unattempted_count = 0
    total_marks = 0.0
    
    # We will build a breakdown dictionary mapping e.g., "part1" -> "Q01" -> "✓"
    breakdown = {
        "part1": {},
        "part2": {}
    }
    
    # Track maximum possible marks to calculate percentage later
    max_possible_marks = 0
    
    # Loop through both parts of the quiz
    for part in PART_LABELS:
        # Get the dictionaries for just this part
        student_part = student_answers.get(part, {})
        key_part = answer_key.get(part, {})
        
        # Loop through every question in the answer key for this part
        for q_label, correct_ans in key_part.items():
            max_possible_marks += MARKS_PER_CORRECT
            
            # What did the student answer? (Defaults to UNATTEMPTED if missing)
            student_ans = student_part.get(q_label, UNATTEMPTED)
            
            # Case 1: Unattempted (Blank)
            if student_ans == UNATTEMPTED:
                unattempted_count += 1
                total_marks += MARKS_UNATTEMPTED
                breakdown[part][q_label] = SYMBOL_UNATTEMPTED
                
            # Case 2: Invalid (Multiple bubbles filled)
            elif student_ans == INVALID:
                incorrect_count += 1
                total_marks -= NEGATIVE_MARKING # Often penalized same as incorrect
                breakdown[part][q_label] = SYMBOL_INVALID
                
            # Case 3: Correct Answer
            elif student_ans == correct_ans:
                correct_count += 1
                total_marks += MARKS_PER_CORRECT
                breakdown[part][q_label] = SYMBOL_CORRECT
                
            # Case 4: Incorrect Answer
            else:
                incorrect_count += 1
                total_marks -= NEGATIVE_MARKING
                breakdown[part][q_label] = SYMBOL_INCORRECT
                
    # Calculate percentage (prevent division by zero just in case)
    if max_possible_marks > 0:
        percentage = (total_marks / max_possible_marks) * 100
        # Prevent negative percentage if heavy negative marking is used
        percentage = max(0.0, percentage)
    else:
        percentage = 0.0
        
    letter_grade = calculate_letter_grade(percentage)
    
    # Build the final GradeReport dictionary
    grade_report = {
        "correct": correct_count,
        "incorrect": incorrect_count,
        "unattempted": unattempted_count,
        "total_marks": total_marks,
        "max_marks": max_possible_marks,
        "percentage": round(percentage, 2), # Round to 2 decimal places
        "grade": letter_grade,
        "breakdown": breakdown
    }
    
    logger.info(f"Grading complete. Score: {total_marks}/{max_possible_marks} ({letter_grade})")
    
    return grade_report
