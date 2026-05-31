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
    - String answers -> Uppercase single letter (A, B, C, D)
    - Numbers -> Converted to letters (1->A, 2->B, etc.)
    """
    if answer is None:
        return UNATTEMPTED
    
    # Handle INVALID constant
    if answer == INVALID:
        return INVALID
    
    # Convert to string and clean
    answer_str = str(answer).strip().upper()
    
    # Handle empty string
    if not answer_str or answer_str == "NONE" or answer_str == "NULL":
        return UNATTEMPTED
    
    # Check for invalid marker
    if answer_str == "INVALID" or answer_str == "⚠":
        return INVALID
    
    # Remove any extra characters, keep only valid option letters
    answer_str = re.sub(r'[^A-D0-9]', '', answer_str)
    
    if not answer_str:
        return UNATTEMPTED
    
    # Handle numeric answers (1=A, 2=B, 3=C, 4=D)
    if answer_str.isdigit():
        num = int(answer_str)
        if 1 <= num <= 4:
            return OPTION_LABELS[num - 1]  # 1->A, 2->B, etc.
        return INVALID
    
    # Single letter answer - validate it's A-D
    if answer_str[0] in OPTION_LABELS:
        return answer_str[0]
    
    # Unknown format
    logger.warning(f"Unknown answer format: '{answer}', treating as INVALID")
    return INVALID


def parse_answer_key(answer_key: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """
    Parse and normalize the answer key into a standard format.
    
    Expected input format from QR decoder:
    {
        "quiz_title": "AI Quiz SP2026",
        "set": "C",
        "part1": {"Q01": "D", "Q02": "A", ...},
        "part2": {"Q01": "C", "Q02": "D", ...}
    }
    
    Returns:
    {
        "part1": {"Q01": "D", "Q02": "A", ...},
        "part2": {"Q01": "C", "Q02": "D", ...}
    }
    """
    normalized = {"part1": {}, "part2": {}}
    
    if not answer_key:
        logger.error("Empty answer key provided")
        return normalized
    
    logger.debug(f"Parsing answer key: {answer_key}")
    
    # Check for standard format with part1 and part2 keys
    for part in PART_LABELS:
        if part in answer_key:
            part_data = answer_key[part]
            if isinstance(part_data, dict):
                for q_label, answer in part_data.items():
                    normalized_answer = normalize_answer(answer)
                    if normalized_answer:  # Only add if valid
                        normalized[part][q_label] = normalized_answer
                        logger.debug(f"  {part} {q_label} = {normalized_answer}")
    
    # If no parts found in expected format, try to infer from flat structure
    if not normalized["part1"] and not normalized["part2"]:
        logger.warning("No standard part structure found, trying flat format...")
        temp_answers = {}
        for key, value in answer_key.items():
            # Skip metadata keys
            if key in ["quiz_title", "set", "class", "subject", "title", "quiz_set"]:
                continue
            
            # Check if it looks like a question key (Q01, Q1, 1, etc.)
            q_match = re.match(r'^Q?(\d+)$', str(key), re.IGNORECASE)
            if q_match:
                q_num = int(q_match.group(1))
                q_label = f"Q{q_num:02d}"
                temp_answers[q_label] = normalize_answer(value)
        
        # Split into parts (Q01-Q08 -> part1, Q09-Q16 -> part2)
        for q_label, answer in temp_answers.items():
            q_num = int(re.search(r'\d+', q_label).group())
            if 1 <= q_num <= QUESTIONS_PER_PART:
                normalized["part1"][q_label] = answer
            elif QUESTIONS_PER_PART < q_num <= QUESTIONS_PER_PART * 2:
                # Map Q09-Q16 to Q01-Q08 in part2
                adjusted_label = f"Q{(q_num - QUESTIONS_PER_PART):02d}"
                normalized["part2"][adjusted_label] = answer
    
    part1_count = len(normalized["part1"])
    part2_count = len(normalized["part2"])
    logger.info(f"Parsed answer key: Part1={part1_count} questions, Part2={part2_count} questions")
    
    if part1_count == 0 and part2_count == 0:
        logger.error("Failed to parse any answers from answer key!")
        logger.error(f"Answer key content: {answer_key}")
    
    return normalized


def calculate_letter_grade(percentage: float) -> str:
    """
    Converts a percentage score into a standard Letter Grade.
    Uses the GRADE_BOUNDARIES defined in config.py.
    """
    # Ensure percentage is within valid range
    percentage = max(0.0, min(100.0, percentage))
    
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
    
    logger.debug(f"Comparing: student='{student_norm}' vs correct='{correct_norm}'")
    
    # Case 1: Unattempted
    if student_norm is None:
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
    
    # Validate that we have an answer key
    if not normalized_key["part1"] and not normalized_key["part2"]:
        logger.error("No valid answer key found! Cannot grade.")
        return {
            "correct": 0,
            "incorrect": 0,
            "unattempted": 0,
            "invalid": 0,
            "total_marks": 0,
            "max_marks": 0,
            "percentage": 0,
            "grade": "F",
            "breakdown": {"part1": {}, "part2": {}},
            "questions_processed": 0,
            "error": "No valid answer key found"
        }
    
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
        student_part = student_answers.get(part, {}) if student_answers else {}
        key_part = normalized_key.get(part, {})
        
        if not key_part:
            logger.warning(f"No answer key found for {part}, skipping...")
            continue
        
        logger.info(f"Grading {part}...")
        
        # Process each question in the answer key
        for q_label in sorted(key_part.keys()):
            correct_ans = key_part.get(q_label)
            student_ans = student_part.get(q_label, UNATTEMPTED) if student_part else UNATTEMPTED
            
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
        if student_part:
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
        
        student_part = student_answers.get(part, {}) if student_answers else {}
        key_part = normalized_key.get(part, {})
        breakdown_part = grade_report['breakdown'].get(part, {})
        
        for q_label in sorted(key_part.keys()):
            correct = key_part.get(q_label, "?")
            student = student_part.get(q_label, UNATTEMPTED) if student_part else UNATTEMPTED
            if student is None:
                student = "—"
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
