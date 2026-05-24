# Automated Quiz Scanner & Grading System

This is an AI-powered desktop application built for the **Artificial Intelligence (BSE-4A) SP2026** course assignment. It automatically scans filled quiz sheets, decodes QR codes to get the answer key, reads handwritten student information via OCR, grades the bubbles, and exports a detailed Excel/CSV report.

## 🚀 Features (Tasks Completed)
- [ ] **Task 1: QR Code Decoding** - Extracts the answer key directly from the QR code.
- [ ] **Task 2: Student Info Extraction (OCR)** - Reads handwritten Name and Registration Number.
- [ ] **Task 3: Bubble Sheet Reading** - Detects filled bubbles using contour detection and fill-ratio analysis. Handles multi-filled (invalid) and unattempted questions.
- [ ] **Task 4: Quiz Grading** - Compares student answers against the QR key and generates a score with a per-question breakdown.
- [ ] **Task 5: Batch Processing** - Processes entire folders of quiz images at once and exports results to a single Excel/CSV file with summary statistics.

## 🛠️ Technology Stack
- **Core Language**: Python 3.10+
- **Computer Vision**: OpenCV (`opencv-python`), NumPy
- **QR Decoding**: `pyzbar`
- **OCR (Handwriting)**: `easyocr`
- **Data & Reports**: `pandas`, `openpyxl`
- **User Interface**: `streamlit` (Browser-based GUI)

## 📁 Folder Structure
```
quiz-scanner/
├── src/                  # Core source code modules
│   ├── preprocessing/    # Image cleanup and rotation fix
│   ├── qr_decoder/       # QR code detection
│   ├── ocr/              # Student info extraction
│   ├── bubble_detection/ # Answer grid reading
│   ├── grading/          # Score calculation
│   ├── batch_processing/ # Folder processing logic
│   ├── reporting/        # CSV/Excel export
│   ├── ui/               # Streamlit web app
│   └── utils/            # Shared helpers & config
├── samples/              # Sample quiz sheets
├── output/               # Generated reports and debug images
├── docs/                 # Assignment instructions
├── config.py             # Global configuration settings
└── requirements.txt      # Python dependencies
```

## ⚙️ Installation & Running

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application:**
   ```bash
   streamlit run src/ui/app.py
   ```
   *The application will open automatically in your web browser.*

## 📸 Demo
*(Screenshots will be added here upon completion)*

---
**Course:** Artificial Intelligence (BSE-4A)  
**Semester:** SP 2026  
**Developer:** Hassan Khan  
