"""
app.py — Streamlit Web Interface for Demo
===========================================

PURPOSE:
    Provides a clean, modern web interface for the Quiz Scanner.
    Allows the user to upload a single image for testing, or run
    batch processing with a single click.

HOW TO RUN:
    streamlit run src/ui/app.py
"""

import os
import sys
import pandas as pd
import streamlit as st
from PIL import Image
import numpy as np
import cv2

# Ensure the parent directory is in the Python path so imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.batch_processing.batch_processor import process_single_image, process_batch
from config import SAMPLES_BATCH_DIR

# --- STREAMLIT PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Automated Quiz Scanner",
    page_icon="📝",
    layout="wide"
)

# --- HELPER TO CONVERT UPLOADED FILE TO OPENCV FORMAT ---
def load_uploaded_image(uploaded_file) -> str:
    """Saves the uploaded file temporarily so our OpenCV pipeline can read it."""
    temp_dir = os.path.join(os.path.dirname(SAMPLES_BATCH_DIR), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, "temp_upload.jpg")
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    return temp_path


# --- UI LAYOUT ---

st.title("📝 Automated Quiz Scanner & Grading System")
st.markdown("""
**Course:** Artificial Intelligence (BSE-4A) SP2026  
This AI-powered system automatically scans quiz sheets, decodes the QR answer key, 
reads handwritten student information via OCR, and grades the bubbles.
""")

st.divider()

# Split the screen into two columns: Left for controls, Right for results
col_left, col_right = st.columns([1, 2])

with col_left:
    st.header("Upload Quiz")
    uploaded_file = st.file_uploader("Upload a quiz sheet (JPG/PNG)", type=["jpg", "jpeg", "png"])
    
    st.divider()
    
    st.header("Batch Processing")
    st.info(f"Will process all images in: `{SAMPLES_BATCH_DIR}`")
    if st.button("Run Batch Processing", type="primary"):
        with st.spinner("Processing batch... This may take a minute."):
            report_path = process_batch(SAMPLES_BATCH_DIR)
            
            if report_path and os.path.exists(report_path):
                st.session_state["report_path"] = report_path
                st.success("Batch processing complete!")
            else:
                st.error("Batch processing failed or no images found.")
                
    if "report_path" in st.session_state:
        # Provide a download button for the generated Excel file
        with open(st.session_state["report_path"], "rb") as file:
            btn = st.download_button(
                label="📥 Download Excel Report",
                data=file,
                file_name=os.path.basename(st.session_state["report_path"]),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

with col_right:
    if uploaded_file is not None:
        st.header("Analysis Results")
        
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Quiz Sheet", use_container_width=True)
        
        with st.spinner("Running AI Pipeline (QR -> OCR -> Bubbles -> Grading)..."):
            # Save temp file for our pipeline
            temp_path = load_uploaded_image(uploaded_file)
            
            # Run the AI
            result = process_single_image(temp_path)
            
        if "error" in result:
            st.error(f"Error processing image: {result['error']}")
        else:
            # --- DISPLAY RESULTS ---
            student = result["student_info"]
            grade = result["grade"]
            key = result["answer_key"]
            
            st.success("Quiz graded successfully!")
            
            # --- ANSWER KEY SECTION (from QR Code) ---
            with st.expander("📱 QR Code Answer Key", expanded=True):
                st.subheader(f"Quiz: {key.get('quiz_title', 'Unknown')} (Set {key.get('set', '?')})")
                
                # Create columns for Part-I and Part-II answer keys
                ak_col1, ak_col2 = st.columns(2)
                
                with ak_col1:
                    st.markdown("**Part-I Answer Key**")
                    part1_answers = key.get('part1', {})
                    if part1_answers:
                        ak_data1 = []
                        for i in range(1, 9):
                            q_num = f"Q{str(i).zfill(2)}"
                            ak_data1.append({
                                "Question": q_num,
                                "Answer": part1_answers.get(q_num, "—")
                            })
                        st.table(pd.DataFrame(ak_data1))
                    else:
                        st.info("No Part-I answers found")
                
                with ak_col2:
                    st.markdown("**Part-II Answer Key**")
                    part2_answers = key.get('part2', {})
                    if part2_answers:
                        ak_data2 = []
                        for i in range(1, 9):
                            q_num = f"Q{str(i).zfill(2)}"
                            ak_data2.append({
                                "Question": q_num,
                                "Answer": part2_answers.get(q_num, "—")
                            })
                        st.table(pd.DataFrame(ak_data2))
                    else:
                        st.info("No Part-II answers found")
            
            st.divider()
            
            # --- STUDENT INFO SECTION ---
            st.subheader("👤 Student Information")
            info_col1, info_col2 = st.columns(2)
            with info_col1:
                st.markdown(f"**Name:** {student.get('name', 'Unknown')}")
            with info_col2:
                st.markdown(f"**Registration #:** {student.get('reg_no', 'Unknown')}")
            
            st.divider()
            
            # --- SCORE METRICS SECTION ---
            st.subheader("📊 Score Summary")
            metric_cols = st.columns(4)
            metric_cols[0].metric("Score", f"{grade['total_marks']}/{grade['max_marks']}")
            metric_cols[1].metric("Percentage", f"{grade['percentage']}%")
            metric_cols[2].metric("Grade", grade['grade'])
            metric_cols[3].metric("Correct", f"{grade['correct']}/{grade['questions_processed']}")
            
            # Show summary counts
            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
            summary_col1.metric("✓ Correct", grade['correct'])
            summary_col2.metric("✗ Incorrect", grade['incorrect'])
            summary_col3.metric("— Unattempted", grade['unattempted'])
            summary_col4.metric("⚠ Invalid", grade['invalid'])
            
            st.divider()
            
            # --- QUESTION BREAKDOWN SECTION ---
            st.subheader("📝 Question Breakdown")
            
            # Convert breakdown dict into a pandas DataFrame for nice display
            breakdown = grade["breakdown"]
            
            # Create a combined list of dictionaries for the table
            table_data = []
            for i in range(1, 9):
                q_num = f"Q{str(i).zfill(2)}"
                # Get student answer
                student_ans = "—"
                for part in ["part1", "part2"]:
                    if part in result.get("student_answers", {}):
                        ans = result["student_answers"][part].get(q_num)
                        if ans:
                            student_ans = ans
                            break
                
                table_data.append({
                    "Question": q_num,
                    "Part-I": breakdown.get("part1", {}).get(q_num, ""),
                    "Part-II": breakdown.get("part2", {}).get(q_num, ""),
                    "Student Ans": student_ans
                })
                
            df = pd.DataFrame(table_data)
            st.table(df)
            
            st.caption("Legend: ✓ = Correct | ✗ = Incorrect | — = Unattempted | ⚠ = Invalid (Multiple Filled)")
            
            # --- STUDENT ANSWERS SECTION ---
            with st.expander("📝 Student Answers (Detected Bubbles)", expanded=False):
                ans_col1, ans_col2 = st.columns(2)
                
                with ans_col1:
                    st.markdown("**Part-I Answers**")
                    part1_student = result.get("student_answers", {}).get("part1", {})
                    if part1_student:
                        ans_data1 = []
                        for i in range(1, 9):
                            q_num = f"Q{str(i).zfill(2)}"
                            ans = part1_student.get(q_num, "—")
                            # Get correct answer for comparison
                            correct = key.get('part1', {}).get(q_num, "—")
                            ans_data1.append({
                                "Question": q_num,
                                "Student": ans if ans else "—",
                                "Correct": correct
                            })
                        st.table(pd.DataFrame(ans_data1))
                
                with ans_col2:
                    st.markdown("**Part-II Answers**")
                    part2_student = result.get("student_answers", {}).get("part2", {})
                    if part2_student:
                        ans_data2 = []
                        for i in range(1, 9):
                            q_num = f"Q{str(i).zfill(2)}"
                            ans = part2_student.get(q_num, "—")
                            # Get correct answer for comparison
                            correct = key.get('part2', {}).get(q_num, "—")
                            ans_data2.append({
                                "Question": q_num,
                                "Student": ans if ans else "—",
                                "Correct": correct
                            })
                        st.table(pd.DataFrame(ans_data2))
    else:
        st.info("👈 Upload a quiz sheet on the left to see the AI in action.")
