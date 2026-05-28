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
                st.success("Batch processing complete!")
                
                # Provide a download button for the generated Excel file
                with open(report_path, "rb") as file:
                    btn = st.download_button(
                        label="📥 Download Excel Report",
                        data=file,
                        file_name=os.path.basename(report_path),
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            else:
                st.error("Batch processing failed or no images found.")

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
            
            # Metadata row
            st.subheader(f"Quiz Info: {key.get('quiz_title', 'Unknown')} (Set {key.get('set', '?')})")
            
            # Score metrics
            metric_cols = st.columns(4)
            metric_cols[0].metric("Score", f"{grade['total_marks']}/{grade['max_marks']}")
            metric_cols[1].metric("Percentage", f"{grade['percentage']}%")
            metric_cols[2].metric("Grade", grade['grade'])
            
            st.markdown(f"**Name:** {student.get('name', 'Unknown')}")
            st.markdown(f"**Registration #:** {student.get('reg_no', 'Unknown')}")
            
            # Breakdown Table
            st.subheader("Question Breakdown")
            
            # Convert our breakdown dict into a pandas DataFrame for nice display
            breakdown = grade["breakdown"]
            
            # Create a combined list of dictionaries for the table
            table_data = []
            for i in range(1, 9):
                q_num = f"Q{str(i).zfill(2)}"
                table_data.append({
                    "Question": q_num,
                    "Part-I": breakdown.get("part1", {}).get(q_num, ""),
                    "Part-II": breakdown.get("part2", {}).get(q_num, "")
                })
                
            df = pd.DataFrame(table_data)
            st.table(df)
            
            st.caption("Legend: ✓ = Correct | ✗ = Incorrect | — = Unattempted | ⚠ = Invalid (Multiple Filled)")
    else:
        st.info("👈 Upload a quiz sheet on the left to see the AI in action.")
