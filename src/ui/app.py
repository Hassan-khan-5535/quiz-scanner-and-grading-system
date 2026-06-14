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
import fitz  # PyMuPDF for PDF handling

# Ensure the parent directory is in the Python path so imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.batch_processing.batch_processor import process_single_image, process_batch
from src.reporting.report_generator import generate_report
from config import SAMPLES_DIR, SAMPLES_BATCH_DIR, SAMPLES_SINGLE_DIR

# --- STREAMLIT PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Automated Quiz Scanner",
    page_icon="📝",
    layout="wide"
)

# --- PROFESSIONAL CSS STYLING ---
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Card styling */
    .stCard {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
    }
    
    /* Header styling */
    h1 {
        color: #1e3a5f;
        font-weight: 600;
        border-bottom: 3px solid #2c5f8d;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    
    h2, h3 {
        color: #2c5f8d;
        font-weight: 600;
        margin-top: 1.5rem;
    }
    
    h4 {
        color: #1e3a5f;
        font-weight: 600;
    }
    
    /* Answer key table styling */
    .answer-key-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
        background: white;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .answer-key-table th {
        background-color: #2c5f8d !important;
        color: white !important;
        padding: 0.75rem;
        text-align: center;
        font-weight: 600;
        font-size: 0.95rem;
    }
    
    .answer-key-table td {
        padding: 0.6rem;
        text-align: center;
        border-bottom: 1px solid #e0e0e0;
        font-size: 1rem;
        color: #1e3a5f !important;
    }
    
    .answer-key-table tr:hover {
        background-color: #f0f4f8;
    }
    
    .answer-key-table tr:last-child td {
        border-bottom: none;
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .main {
            background-color: #0e1117 !important;
        }
        
        .stCard {
            background: #262730 !important;
            border: 1px solid #404040 !important;
        }
        
        .answer-key-table {
            background: #262730 !important;
        }
        
        .answer-key-table th {
            background-color: #1e3a5f !important;
        }
        
        .answer-key-table td {
            color: #fafafa !important;
            border-bottom: 1px solid #404040 !important;
        }
        
        .answer-key-table tr:hover {
            background-color: #2d3748 !important;
        }
        
        h1, h2, h3, h4 {
            color: #fafafa !important;
        }
        
        .section-header {
            background: linear-gradient(90deg, #1e3a5f 0%, #2c5f8d 100%) !important;
        }
        
        .metric-card {
            background: #262730 !important;
            border-left: 4px solid #4a90e2 !important;
        }
        
        .metric-value {
            color: #fafafa !important;
        }
        
        .metric-label {
            color: #b0b0b0 !important;
        }
        
        .student-card {
            background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%) !important;
            border-left: 4px solid #4a90e2 !important;
        }
        
        .success-box {
            background-color: #1a4731 !important;
            border-left: 4px solid #48bb78 !important;
            color: #c6f6d5 !important;
        }
        
        .footer {
            background-color: #262730 !important;
            border-top: 2px solid #404040 !important;
            color: #b0b0b0 !important;
        }
    }
    
    /* Info box styling */
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.25rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Success box */
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
        color: #155724;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.25rem;
        border-radius: 8px;
        border-left: 4px solid #2c5f8d;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e3a5f;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Grade colors */
    .grade-A { color: #28a745; }
    .grade-B { color: #2c5f8d; }
    .grade-C { color: #ffc107; }
    .grade-D { color: #fd7e14; }
    .grade-F { color: #dc3545; }
    
    /* Button styling */
    .stButton>button {
        background-color: #2c5f8d;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.6rem 1.75rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #1e3a5f;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #ffffff;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #666;
        padding: 1.5rem;
        margin-top: 2rem;
        border-top: 2px solid #e0e0e0;
        background-color: white;
    }
    
    .footer p {
        margin: 0.5rem 0;
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #2c5f8d 0%, #1e3a5f 100%);
        color: white;
        padding: 0.75rem 1.25rem;
        border-radius: 6px;
        margin: 1.5rem 0 1rem 0;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Student info card */
    .student-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.25rem;
        border-radius: 8px;
        border-left: 4px solid #2c5f8d;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# --- HELPER FUNCTIONS ---

def load_uploaded_image(uploaded_file) -> str:
    """Saves the uploaded file temporarily so our OpenCV pipeline can read it."""
    temp_dir = os.path.join(os.path.dirname(SAMPLES_BATCH_DIR), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, "temp_upload.jpg")
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    return temp_path


def convert_pdf_to_images(uploaded_file) -> list:
    """Converts a PDF file to a list of image paths (one per page)."""
    temp_dir = os.path.join(os.path.dirname(SAMPLES_BATCH_DIR), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save the uploaded PDF temporarily
    pdf_path = os.path.join(temp_dir, "temp_upload.pdf")
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Convert each page to an image
    doc = fitz.open(pdf_path)
    image_paths = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Render page to image (zoom factor for better quality)
        mat = fitz.Matrix(2, 2)  # 2x zoom for better quality
        pix = page.get_pixmap(matrix=mat)
        
        # Save as JPG
        img_path = os.path.join(temp_dir, f"pdf_page_{page_num + 1}.jpg")
        pix.save(img_path)
        image_paths.append(img_path)
    
    doc.close()
    return image_paths


def display_answer_key(answer_key: dict):
    """Display the decoded QR code answer key in a professional format."""
    st.markdown('<div class="section-header">📋 Answer Key (from QR Code)</div>', unsafe_allow_html=True)
    
    # Quiz info card
    st.markdown(f"""
    <div class='stCard'>
        <h4 style='color: #1e3a5f; margin: 0;'>{answer_key.get('quiz_title', 'Unknown Quiz')}</h4>
        <p style='color: #666; margin: 0.5rem 0 0 0; font-size: 1.1rem;'>
            Set: <strong style='color: #2c5f8d;'>{answer_key.get('set', '?')}</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Answer tables for Part-I and Part-II
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Part-I")
        part1 = answer_key.get('part1', {})
        
        # Create table rows
        rows = ""
        for q_num in sorted(part1.keys()):
            answer = part1[q_num]
            rows += f"<tr><td><strong>{q_num}</strong></td><td>{answer}</td></tr>"
        
        st.markdown(f"""
        <table class='answer-key-table'>
            <thead>
                <tr>
                    <th>Question</th>
                    <th>Answer</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### Part-II")
        part2 = answer_key.get('part2', {})
        
        # Create table rows
        rows = ""
        for q_num in sorted(part2.keys()):
            answer = part2[q_num]
            rows += f"<tr><td><strong>{q_num}</strong></td><td>{answer}</td></tr>"
        
        st.markdown(f"""
        <table class='answer-key-table'>
            <thead>
                <tr>
                    <th>Question</th>
                    <th>Answer</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
        """, unsafe_allow_html=True)


def display_student_info(student_info: dict):
    """Display student information in a professional card."""
    st.markdown('<div class="section-header">👤 Student Information</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class='student-card'>
        <div style='margin-bottom: 0.75rem;'>
            <span style='color: #666; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.5px;'>Name</span><br>
            <strong style='font-size: 1.2rem; color: #1e3a5f;'>{student_info.get('name', 'Unknown')}</strong>
        </div>
        <div>
            <span style='color: #666; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.5px;'>Registration Number</span><br>
            <strong style='font-size: 1.2rem; color: #1e3a5f;'>{student_info.get('reg_no', 'Unknown')}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)


def display_grade_summary(grade: dict):
    """Display grade summary with professional metric cards."""
    st.markdown('<div class="section-header">📊 Grade Summary</div>', unsafe_allow_html=True)
    
    grade_letter = grade['grade']
    grade_class = f"grade-{grade_letter}"
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Score</div>
            <div class='metric-value'>{grade['total_marks']}/{grade['max_marks']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Percentage</div>
            <div class='metric-value'>{grade['percentage']}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Grade</div>
            <div class='metric-value {grade_class}'>{grade_letter}</div>
        </div>
        """, unsafe_allow_html=True)


def display_question_breakdown(grade: dict, answer_key: dict, student_answers: dict):
    """Display detailed question breakdown with student's attempted answers and color coding."""
    st.markdown('<div class="section-header">✓ Question Breakdown</div>', unsafe_allow_html=True)
    
    breakdown = grade["breakdown"]
    # Use student's attempted answers for the displayed letters
    part1_attempted = student_answers.get('part1', {})
    part2_attempted = student_answers.get('part2', {})
    
    # Create table rows with color coding
    rows = ""
    for i in range(1, 9):
        q_num = f"Q{str(i).zfill(2)}"
        part1_status = breakdown.get("part1", {}).get(q_num, "")
        part2_status = breakdown.get("part2", {}).get(q_num, "")
        part1_answer = part1_attempted.get(q_num, "?")
        part2_answer = part2_attempted.get(q_num, "?")
        
        # Color code the status symbols
        def colorize(status, answer):
            if status == "✓":
                return f"<span style='color: #28a745; font-weight: bold; font-size: 1.2rem;'>{answer} {status}</span>"
            elif status == "✗":
                return f"<span style='color: #dc3545; font-weight: bold; font-size: 1.2rem;'>{answer} {status}</span>"
            elif status == "—":
                return f"<span style='color: #6c757d; font-weight: bold;'>{answer} {status}</span>"
            elif status == "⚠":
                return f"<span style='color: #fd7e14; font-weight: bold; font-size: 1.2rem;'>{answer} {status}</span>"
            return f"{answer} {status}"
        
        part1_colored = colorize(part1_status, part1_answer)
        part2_colored = colorize(part2_status, part2_answer)
        
        rows += f"<tr><td><strong>{q_num}</strong></td><td>{part1_colored}</td><td>{part2_colored}</td></tr>"
    
    st.markdown(f"""
    <table class='answer-key-table'>
        <thead>
            <tr>
                <th>Question</th>
                <th>Part-I</th>
                <th>Part-II</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background: #f8f9fa; padding: 0.75rem; border-radius: 4px; margin-top: 0.5rem; color: #333333;'>
        <strong style='color: #333333;'>Legend:</strong> 
        <span style='color: #28a745; font-weight: bold;'>✓</span> <span style='color: #333333;'>= Correct</span> | 
        <span style='color: #dc3545; font-weight: bold;'>✗</span> <span style='color: #333333;'>= Incorrect</span> | 
        <span style='color: #6c757d; font-weight: bold;'>—</span> <span style='color: #333333;'>= Unattempted</span> | 
        <span style='color: #fd7e14; font-weight: bold;'>⚠</span> <span style='color: #333333;'>= Invalid (Multiple Filled)</span>
    </div>
    """, unsafe_allow_html=True)


# --- UI LAYOUT ---

# Professional header
st.markdown("""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 2rem; border-radius: 12px; margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
    <h1 style='color: white; margin: 0; border: none;'>📝 Automated Quiz Scanner & Grading System</h1>
    <p style='color: rgba(255,255,255,0.95); font-size: 1.1rem; margin: 1rem 0 0 0;'>
        <strong>Course:</strong> Artificial Intelligence (BSE-4A) | <strong>Semester:</strong> SP2026
    </p>
    <p style='color: rgba(255,255,255,0.9); font-size: 0.95rem; margin: 0.5rem 0 0 0;'>
        This AI-powered system automatically scans quiz sheets, decodes the QR answer key, 
        reads handwritten student information via OCR, and grades the bubbles.
    </p>
</div>
""", unsafe_allow_html=True)

# Split the screen into two columns: Left for controls, Right for results
col_left, col_right = st.columns([1, 2])

with col_left:
    # --- Sample Sheets Download Section ---
    st.markdown('<div class="section-header">📄 Sample Quiz Sheets</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='background: #eef2f7; padding: 0.75rem; border-radius: 6px; margin-bottom: 0.75rem; color: #333333;'>
        <span style='font-size: 0.9rem; color: #333333;'>Download a sample sheet to test the scanner. These are the only supported quiz templates.</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Find all sample sheets in the single folder
    sample_files = sorted([
        f for f in os.listdir(SAMPLES_SINGLE_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]) if os.path.isdir(SAMPLES_SINGLE_DIR) else []
    
    if sample_files:
        for i, sample_name in enumerate(sample_files):
            sample_path = os.path.join(SAMPLES_SINGLE_DIR, sample_name)
            with open(sample_path, "rb") as sf:
                file_ext = os.path.splitext(sample_name)[1].lower()
                mime_type = "image/jpeg" if file_ext in (".jpg", ".jpeg") else "image/png"
                st.download_button(
                    label=f"📥 Sample {i + 1} — {sample_name}",
                    data=sf,
                    file_name=sample_name,
                    mime=mime_type,
                    key=f"sample_download_{i}"
                )
    else:
        st.warning("No sample sheets found.")
    
    st.markdown("---")
    
    # --- Upload Section ---
    st.markdown('<div class="section-header">📤 Upload Quiz</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a quiz sheet (JPG/PNG/PDF)", type=["jpg", "jpeg", "png", "pdf"])
    
    st.markdown("---")
    
    st.markdown('<div class="section-header">⚙️ Batch Processing</div>', unsafe_allow_html=True)
    st.info(f"Will process all images in: `{SAMPLES_DIR}`")
    if st.button("Run Batch Processing", type="primary"):
        with st.spinner("Processing batch... This may take a minute."):
            report_path = process_batch(SAMPLES_DIR)
            
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
        st.markdown('<div class="section-header">🔍 Analysis Results</div>', unsafe_allow_html=True)
        
        # Check if uploaded file is PDF
        is_pdf = uploaded_file.name.lower().endswith('.pdf')
        
        # Display the uploaded file
        with st.container():
            st.markdown(f"""
            <div style='background: white; padding: 1rem; border-radius: 8px; 
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 1rem;'>
                <strong style='color: #2c5f8d;'>Uploaded Quiz Sheet</strong>
            </div>
            """, unsafe_allow_html=True)
            
            if is_pdf:
                st.info(f"📄 PDF file: {uploaded_file.name}")
                # Convert PDF to images
                with st.spinner("Converting PDF pages to images..."):
                    image_paths = convert_pdf_to_images(uploaded_file)
                st.success(f"Converted {len(image_paths)} page(s) from PDF")
            else:
                st.image(uploaded_file, use_container_width=True)
                # Load single image
                image_paths = [load_uploaded_image(uploaded_file)]
        
        # Process each page/image
        for idx, image_path in enumerate(image_paths):
            if len(image_paths) > 1:
                st.markdown(f"---")
                st.markdown(f"### 📄 Page {idx + 1}")
            
            with st.spinner("Running AI Pipeline (QR -> OCR -> Bubbles -> Grading)..."):
                # Run the AI
                result = process_single_image(image_path)
                
            if "error" in result:
                st.error(f"Error processing image: {result['error']}")
            else:
                # --- DISPLAY RESULTS ---
                student = result["student_info"]
                grade = result["grade"]
                key = result["answer_key"]
                
                st.markdown(f"""
                <div class='success-box'>
                    <strong>✓ Quiz graded successfully!</strong>
                </div>
                """, unsafe_allow_html=True)
                
                # Display answer key
                display_answer_key(key)
                
                # Display student information
                display_student_info(student)
                
                # Display grade summary
                display_grade_summary(grade)
                
                # Display question breakdown
                display_question_breakdown(grade, key, result["student_answers"])
                
                # --- Generate Excel report for single image ---
                # Reuse the same generate_report function used by batch processing
                # so the Excel format is identical.
                single_result_for_report = [result]  # wrap in list as generate_report expects a list
                excel_path = generate_report(single_result_for_report, output_format="xlsx")
                
                if excel_path and os.path.exists(excel_path):
                    st.markdown('\n                    <div class="section-header">📥 Download Report</div>\n                    ', unsafe_allow_html=True)
                    
                    with open(excel_path, "rb") as excel_file:
                        st.download_button(
                            label="📥 Download Excel Report",
                            data=excel_file,
                            file_name=os.path.basename(excel_path),
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key=f"single_download_{idx}"
                        )
    else:
        st.markdown("""
        <div style='background: white; padding: 3rem; border-radius: 8px; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center;
                    margin-top: 22rem;'>
            <div style='font-size: 4rem; margin-bottom: 1rem;'>👈</div>
            <h3 style='color: #2c5f8d; margin: 0;'>Upload a Quiz Sheet</h3>
            <p style='color: #666; margin-top: 1rem;'>
                Upload a quiz sheet on the left to see the AI in action.
            </p>
        </div>
        """, unsafe_allow_html=True)

# Professional footer
st.markdown("---")
st.markdown("""
<div class='footer'>
    <p><strong style='color: #1e3a5f; font-size: 1.1rem;'>Automated Quiz Scanner & Grading System</strong></p>
    <p style='color: #666;'>Course: Artificial Intelligence (BSE-4A) | Semester: SP2026</p>
    <p style='font-size: 0.9em; color: #999; margin-top: 0.5rem;'>
        Powered by Computer Vision & OCR Technology
    </p>
</div>
""", unsafe_allow_html=True)
