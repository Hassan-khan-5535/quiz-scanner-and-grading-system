"""
test_grading.py — Unit Tests for Task 4 (Grading)
===================================================

Run with: python -m pytest tests/test_grading.py
"""

import sys
import os

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.grading.grader import grade_quiz, calculate_letter_grade
from src.utils.constants import UNATTEMPTED, INVALID

def test_perfect_score():
    """Test when a student gets 100% correct."""
    answer_key = {
        "part1": {"Q01": "A", "Q02": "B"},
        "part2": {"Q01": "C"}
    }
    
    student_answers = {
        "part1": {"Q01": "A", "Q02": "B"},
        "part2": {"Q01": "C"}
    }
    
    report = grade_quiz(student_answers, answer_key)
    
    assert report["correct"] == 3
    assert report["incorrect"] == 0
    assert report["unattempted"] == 0
    assert report["invalid"] == 0
    assert report["total_marks"] == 3.0
    assert report["percentage"] == 100.0
    assert report["grade"] == "A"
    assert report["breakdown"]["part1"]["Q01"] == "✓"


def test_mixed_score():
    """Test a mix of correct, incorrect, unattempted, and invalid answers."""
    answer_key = {
        "part1": {"Q01": "A", "Q02": "B", "Q03": "C", "Q04": "D"}
    }
    
    student_answers = {
        "part1": {
            "Q01": "A",        # Correct
            "Q02": "C",        # Incorrect
            "Q03": UNATTEMPTED,# Unattempted
            "Q04": INVALID     # Multiple filled
        }
    }
    
    report = grade_quiz(student_answers, answer_key)
    
    assert report["correct"] == 1
    assert report["incorrect"] == 1  # 1 wrong answer
    assert report["unattempted"] == 1
    assert report["invalid"] == 1    # 1 invalid (multiple bubbles)
    assert report["total_marks"] == 1.0
    assert report["percentage"] == 25.0
    assert report["grade"] == "F"
    assert report["breakdown"]["part1"]["Q01"] == "✓"
    assert report["breakdown"]["part1"]["Q02"] == "✗"
    assert report["breakdown"]["part1"]["Q03"] == "—"
    assert report["breakdown"]["part1"]["Q04"] == "⚠"


def test_letter_grade_boundaries():
    """Ensure grading matches the standard academic scale."""
    assert calculate_letter_grade(95.0) == "A"
    assert calculate_letter_grade(90.0) == "A"
    assert calculate_letter_grade(85.0) == "B"
    assert calculate_letter_grade(75.0) == "C"
    assert calculate_letter_grade(65.0) == "D"
    assert calculate_letter_grade(59.9) == "F"


def test_numeric_answer_conversion():
    """Test that numeric answers (1,2,3,4) are converted to letters (A,B,C,D)."""
    answer_key = {
        "part1": {"Q01": "A", "Q02": "B", "Q03": "C", "Q04": "D"}
    }
    
    student_answers = {
        "part1": {
            "Q01": "1",  # Should convert to A
            "Q02": 2,    # Should convert to B
            "Q03": "3",  # Should convert to C
            "Q04": 4     # Should convert to D
        }
    }
    
    report = grade_quiz(student_answers, answer_key)
    
    assert report["correct"] == 4
    assert report["incorrect"] == 0
    assert report["total_marks"] == 4.0
    assert report["percentage"] == 100.0


def test_case_insensitive_comparison():
    """Test that answer comparison is case-insensitive."""
    answer_key = {
        "part1": {"Q01": "A", "Q02": "B"}
    }
    
    student_answers = {
        "part1": {
            "Q01": "a",  # lowercase
            "Q02": "B"   # uppercase
        }
    }
    
    report = grade_quiz(student_answers, answer_key)
    
    assert report["correct"] == 2
    assert report["incorrect"] == 0
