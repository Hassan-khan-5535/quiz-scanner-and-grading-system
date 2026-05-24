"""
report_generator.py — Task 5 (Part A): Report Generation
==========================================================

PURPOSE:
    Takes the aggregated data from all processed quizzes and converts it
    into a structured Excel (.xlsx) or CSV report using Pandas.

ASSIGNMENT REQUIREMENTS MET:
    - All specific columns required by assignment (Quiz, Set, Name, Reg No, etc.)
    - Summary row at the bottom (Class average, highest, lowest)
    - Auto-named with timestamp
"""

import os
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd

from src.utils.logger import get_logger
from config import RESULTS_DIR
from src.utils.helpers import ensure_directory

logger = get_logger(__name__)


def generate_report(batch_results: List[Dict[str, Any]], output_format: str = "xlsx") -> str:
    """
    Main function to generate the final grade report.
    
    Args:
        batch_results: List of dictionaries, where each dict represents ONE student's
                       fully processed quiz (contains OCR info, answers, grade).
        output_format: "xlsx" or "csv"
        
    Returns:
        The full file path where the report was saved.
    """
    logger.info(f"Generating {output_format.upper()} report for {len(batch_results)} students...")
    
    if not batch_results:
        logger.error("No results to generate report from.")
        return ""
        
    # We need to flatten the nested dictionaries into a flat row for Pandas.
    flat_rows = []
    
    for result in batch_results:
        # If processing failed for this image, it might missing keys
        if "error" in result:
            flat_rows.append({"File": result.get("filename", "Unknown"), "Error": result["error"]})
            continue
            
        student_info = result["student_info"]
        answer_key = result["answer_key"]
        student_answers = result["student_answers"]
        grade = result["grade"]
        
        # Build the flat row
        row = {
            "Quiz": answer_key.get("quiz_title", "Unknown"),
            "Set": answer_key.get("set", "Unknown"),
            "Class": "BSE-4A",         # Hardcoded default per assignment/our discussion
            "Subject": "Artificial Intelligence", # Hardcoded default
            "Name": student_info.get("name", "Unknown"),
            "Reg No": student_info.get("reg_no", "Unknown"),
        }
        
        # Add Part 1 Answers
        part1 = student_answers.get("part1", {})
        for i in range(1, 9):
            q_label = f"Q{str(i).zfill(2)}"
            # E.g., "Part1_Q01": "A"
            row[f"Part1_{q_label}"] = part1.get(q_label, "—")
            
        # Add Part 2 Answers
        part2 = student_answers.get("part2", {})
        for i in range(1, 9):
            q_label = f"Q{str(i).zfill(2)}"
            row[f"Part2_{q_label}"] = part2.get(q_label, "—")
            
        # Add Grading Metrics
        row["Correct"] = grade.get("correct", 0)
        row["Incorrect"] = grade.get("incorrect", 0)
        row["Unattempted"] = grade.get("unattempted", 0)
        row["Total Marks"] = grade.get("total_marks", 0)
        row["Percentage"] = grade.get("percentage", 0.0)
        row["Grade"] = grade.get("grade", "F")
        
        flat_rows.append(row)
        
    # Create the Pandas DataFrame
    df = pd.DataFrame(flat_rows)
    
    # Calculate Summary Statistics (only for successful rows, ignore error rows)
    # The assignment asks for: class average, highest score, lowest score
    if "Total Marks" in df.columns:
        valid_scores = pd.to_numeric(df["Total Marks"], errors='coerce').dropna()
        
        if not valid_scores.empty:
            avg_score = valid_scores.mean()
            max_score = valid_scores.max()
            min_score = valid_scores.min()
            
            # Create a blank row to separate data from summary
            blank_row = {col: "" for col in df.columns}
            
            # Create the summary row
            summary_row = {col: "" for col in df.columns}
            summary_row["Quiz"] = "SUMMARY STATS"
            summary_row["Correct"] = f"Average: {avg_score:.2f}"
            summary_row["Incorrect"] = f"Highest: {max_score}"
            summary_row["Unattempted"] = f"Lowest: {min_score}"
            
            # Append rows using concat (append is deprecated in newer Pandas)
            df = pd.concat([df, pd.DataFrame([blank_row, summary_row])], ignore_index=True)
            
    # Save the file
    ensure_directory(RESULTS_DIR)
    
    # Auto-name with timestamp (e.g., "Quiz_Report_20260524_1845.xlsx")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Use the quiz title from the first valid row if available
    first_valid = next((r for r in flat_rows if "Quiz" in r), None)
    prefix = first_valid["Quiz"].replace(" ", "_") if first_valid else "Quiz_Report"
    
    filename = f"{prefix}_{timestamp}.{output_format}"
    filepath = os.path.join(RESULTS_DIR, filename)
    
    if output_format == "xlsx":
        # index=False means we don't save the row numbers (0, 1, 2...)
        df.to_excel(filepath, index=False)
    else:
        df.to_csv(filepath, index=False)
        
    logger.info(f"Report successfully saved to: {filepath}")
    return filepath
