"""
grader.py — Task 4: Quiz Grading Logic
========================================

PURPOSE:
    Compares the student's extracted answers against the decoded answer key,
    calculates scores, and generates a per-question breakdown with detailed
    analysis and error handling.

IMPROVEMENTS:
    - Robust answer key format handling
    - Case-insensitive comparison
    - Flexible answer key structures
    - Detailed error reporting
    - Support for partial credit
    - Comprehensive logging
"""

from typing import Dict, Any, Optional, List, Tuple
import re

from src.utils.logger import get_logger
from src.utils.constants import (
    PART_LABELS, OPTION_LABELS,
    SYMBOL_CORRECT, SYMBOL_INCORRECT, 
    SYMBOL_UNATTEMPTED, SYMBOL_INVALID,
    UNATTEMPTED, INVALID
)
from config import (
    MARKS_PER_CORRECT, 
    NEGATIVE_MARKING, 
    MARKS_UNATTEMPTED,
    GRADE_BOUNDARIES,
    QUESTIONS_PER_PART
)

logger = get_logger(__name__)


def normalize_answer(answer: Any) -> Optional[str]:
    """
    Normalize an answer value to a standard format.
    
    Handles:
    - None/UNATTEMPTED -> None
    - INVALID -> "INVALID"
    - String answers -> Uppercase single letter
    - Numbers -> Converted to letters (1->A, 2->B, etc.)
    """
    if answer is None or answer == UNATTEMPTED:
        return UNATTEMPTED
    
    if answer == INVALID or str(answer).upper() == "INVALID":
        return INVALID
    
    # Convert to string and clean
    answer_str = str(answer).strip().upper()
    
    # Remove any extra characters, keep only first letter if it's A-D
    answer_str = re.sub(r'[^A-D0-9]', '', answer_str)
    
    if not answer_str:
        return UNATTEMPTED
    
    # Handle numeric answers (1=A, 2=B, 3=C, 4=D)
    if answer_str.isdigit():
        num = int(answer_str)
        if 1 <= num <= 4:
            return OPTION_LABELS[num - 1]  # 1->A, 2->B, etc.
        return INVALID
    
    # Single letter answer
    if answer_str[0] in OPTION_LABELS:
        return answer_str[0]
    
    # Unknown format
    logger.warning(f"Unknown answer format: '{answer}', treating as INVALID")
    return INVALID


def parse_answer_key(answer_key: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """
    Parse and normalize the answer key into a standard format.
    
    Handles various answer key formats:
    - Standard: {"part1": {"Q01": "A", ...}, "part2": {...}}
    - Flat: {"Q01": "A", "Q02": "B", ...}
    - With metadata: {"quiz_title": "...", "set": "A", "part1": {...}}
    - QR payload format with nested structures
    """
    normalized = {"part1": {}, "part2": {}}
    
    if not answer_key:
        logger.error("Empty answer key provided")
        return normalized
    
    # Check if it's the QR payload format with nested structure
    for part in PART_LABELS:
        if part in answer_key:
            part_data = answer_key[part]
            if isinstance(part_data, dict):
                for q_label, answer in part_data.items():
                    normalized[part][q_label] = normalize_answer(answer)
    
    # If no parts found, try to infer from flat structure
    if not normalized["part1"] and not normalized["part2"]:
        # Try to organize flat structure into parts
        temp_answers = {}
        for key, value in answer_key.items():
            # Skip metadata keys
            if key in ["quiz_title", "set", "class", "subject", "title"]:
                continue
            
            # Check if it looks like a question key
            if re.match(r'^Q?\d+$', str(key), re.IGNORECASE):
                q_label = f"Q{int(re.search(r'\d+', str(key)).group()):02d}"
                temp_answers[q_label] = normalize_answer(value)
        
        # Split into parts (Q01-Q08 -> part1, Q09-Q16 -> part2, etc.)
        for q_label, answer in temp_answers.items():
            q_num = int(re.search(r'\d+', q_label).group())
            if q_num <= QUESTIONS_PER_PART:
                normalized["part1"][q_label] = answer
            else:
                # Map Q09-Q16 to Q01-Q08 in part2
                adjusted_label = f"Q{(q_num - QUESTIONS_PER_PART):02d}"
                normalized["part2"][adjusted_label] = answer
    
    logger.info(f"Parsed answer key: Part1={len(normalized['part1'])} questions, "
                f"Part2={len(normalized['part2'])} questions")
    
    return normalized


def calculate_letter_grade(percentage: float) -> str:
    """
    Converts a percentage score into a standard Letter Grade.
    Uses the GRADE_BOUNDARIES defined in config.py.
    """
    # Sort boundaries in descending order
    sorted_boundaries = sorted(GRADE_BOUNDARIES.items(), key=lambda x: x[1], reverse=True)
    
    for grade, boundary in sorted_boundaries:
        if percentage >= boundary:
            return grade
    
    return "F"


def compare_answers(student_ans: Optional[str], correct_ans: Optional[str]) -> Tuple[str, float, str]:
    """
    Compare student answer with correct answer.
    
    Returns:
        Tuple of (symbol, marks_awarded, status)
        - symbol: Visual indicator (✓, ✗, —, ⚠)
        - marks_awarded: Points earned
        - status: "correct", "incorrect", "unattempted", "invalid"
    """
    # Normalize both answers
    student_norm = normalize_answer(student_ans)
    correct_norm = normalize_answer(correct_ans)
    
    # Case 1: Unattempted
    if student_norm == UNATTEMPTED:
        return SYMBOL_UNATTEMPTED, MARKS_UNATTEMPTED, "unattempted"
    
    # Case 2: Invalid (multiple bubbles or unreadable)
    if student_norm == INVALID:
        marks = -NEGATIVE_MARKING if NEGATIVE_MARKING > 0 else 0
        return SYMBOL_INVALID, marks, "invalid"
    
    # Case 3: Correct Answer
    if student_norm == correct_norm:
        return SYMBOL_CORRECT, MARKS_PER_CORRECT, "correct"
    
    # Case 4: Incorrect Answer
    marks = -NEGATIVE_MARKING if NEGATIVE_MARKING > 0 else 0
    return SYMBOL_INCORRECT, marks, "incorrect"


def grade_quiz(student_answers: Dict[str, Dict[str, str]], 
               answer_key: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function for Task 4. Compares student answers to the key.
    
    Args:
        student_answers: Output from Task 3 (bubble_reader.py)
                       Format: {"part1": {"Q01": "A", ...}, "part2": {...}}
        answer_key: Output from Task 1 (qr_reader.py)
                    Can be various formats (see parse_answer_key)
    
    Returns:
        GradeReport dictionary containing scores and visual breakdown.
    """
    logger.info("=" * 50)
    logger.info("Starting grading process...")
    logger.info("=" * 50)
    
    # Parse and normalize answer key
    normalized_key = parse_answer_key(answer_key)
    
    # Initialize counters
    correct_count = 0
    incorrect_count = 0
    unattempted_count = 0
    invalid_count = 0
    total_marks = 0.0
    
    # Initialize breakdown structure
    breakdown = {part: {} for part in PART_LABELS}
    
    # Track questions processed and maximum possible marks
    questions_processed = 0
    max_possible_marks = 0
    
    # Process each part
    for part in PART_LABELS:
        student_part = student_answers.get(part, {})
        key_part = normalized_key.get(part, {})
        
        if not key_part:
            logger.warning(f"No answer key found for {part}")
            continue
        
        logger.info(f"Grading {part}...")
        
        # Process each question in the answer key
        for q_label in sorted(key_part.keys()):
            correct_ans = key_part.get(q_label)
            student_ans = student_part.get(q_label, UNATTEMPTED)
            
            questions_processed += 1
            max_possible_marks += MARKS_PER_CORRECT
            
            # Compare answers
            symbol, marks, status = compare_answers(student_ans, correct_ans)
            
            # Update counters
            if status == "correct":
                correct_count += 1
            elif status == "incorrect":
                incorrect_count += 1
            elif status == "unattempted":
                unattempted_count += 1
            elif status == "invalid":
                invalid_count += 1
            
            total_marks += marks
            breakdown[part][q_label] = symbol
            
            # Log detailed comparison
            logger.debug(f"  {q_label}: Student={student_ans}, Key={correct_ans}, "
                        f"Result={status}, Marks={marks:+g}")
        
        # Handle questions in student answers but not in key (extra questions)
        for q_label in student_part:
            if q_label not in key_part:
                logger.warning(f"  {q_label} in student answers but not in answer key")
    
    # Calculate percentage
    if max_possible_marks > 0:
        percentage = (total_marks / max_possible_marks) * 100
        # Clamp percentage to valid range
        percentage = max(0.0, min(100.0, percentage))
    else:
        percentage = 0.0
        logger.warning("No questions found in answer key!")
    
    letter_grade = calculate_letter_grade(percentage)
    
    # Build detailed report
    grade_report = {
        "correct": correct_count,
        "incorrect": incorrect_count,
        "unattempted": unattempted_count,
        "invalid": invalid_count,
        "total_marks": round(total_marks, 2),
        "max_marks": max_possible_marks,
        "percentage": round(percentage, 2),
        "grade": letter_grade,
        "breakdown": breakdown,
        "questions_processed": questions_processed,
        "scoring": {
            "marks_per_correct": MARKS_PER_CORRECT,
            "negative_marking": NEGATIVE_MARKING,
            "marks_unattempted": MARKS_UNATTEMPTED
        }
    }
    
    logger.info("=" * 50)
    logger.info(f"Grading complete: {correct_count}/{questions_processed} correct")
    logger.info(f"Score: {total_marks}/{max_possible_marks} ({percentage:.1f}%) - Grade: {letter_grade}")
    logger.info(f"Breakdown: {correct_count}✓ {incorrect_count}✗ {unattempted_count}— {invalid_count}⚠")
    logger.info("=" * 50)
    
    return grade_report


def generate_detailed_report(student_answers: Dict[str, Dict[str, str]],
                            answer_key: Dict[str, Any],
                            grade_report: Dict[str, Any]) -> str:
    """
    Generate a human-readable detailed grading report.
    
    Returns:
        Formatted string with detailed question-by-question analysis
    """
    normalized_key = parse_answer_key(answer_key)
    
    lines = [
        "=" * 60,
        "DETAILED GRADING REPORT",
        "=" * 60,
        "",
        f"Score: {grade_report['total_marks']}/{grade_report['max_marks']} "
        f"({grade_report['percentage']:.1f}%)",
        f"Grade: {grade_report['grade']}",
        f"",
        f"Summary: {grade_report['correct']} Correct | "
        f"{grade_report['incorrect']} Incorrect | "
        f"{grade_report['unattempted']} Unattempted | "
        f"{grade_report['invalid']} Invalid",
        "",
        "-" * 60,
    ]
    
    for part in PART_LABELS:
        lines.append(f"\n{part.upper()}:")
        lines.append("-" * 40)
        
        student_part = student_answers.get(part, {})
        key_part = normalized_key.get(part, {})
        breakdown_part = grade_report['breakdown'].get(part, {})
        
        for q_label in sorted(key_part.keys()):
            correct = key_part.get(q_label, "?")
            student = student_part.get(q_label, UNATTEMPTED) or "—"
            symbol = breakdown_part.get(q_label, "?")
            
            status_text = {
                SYMBOL_CORRECT: "CORRECT",
                SYMBOL_INCORRECT: "INCORRECT",
                SYMBOL_UNATTEMPTED: "UNATTEMPTED",
                SYMBOL_INVALID: "INVALID"
            }.get(symbol, "UNKNOWN")
            
            lines.append(f"  {q_label}: Student={student:4s} | Key={correct:4s} | {symbol} {status_text}")
    
    lines.extend(["", "=" * 60])
    
    return "\n".join(lines)
