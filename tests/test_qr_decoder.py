"""
test_qr_decoder.py — Unit Tests for Task 1 (QR Parsing)
=========================================================

Run with: python -m pytest tests/test_qr_decoder.py
"""

import sys
import os

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.qr_decoder.qr_reader import parse_qr_payload

def test_valid_qr_payload():
    """Test parsing the exact format provided in the assignment."""
    payload = "AI Quiz SP2026 Set-C | Part-I: Q1=D Q2=A | Part-II: Q1=C Q2=D"
    
    result = parse_qr_payload(payload)
    
    assert result is not None
    assert result["quiz_title"] == "AI Quiz SP2026"
    assert result["set"] == "C"
    
    # Check Part 1 formatting (Q1 -> Q01)
    assert result["part1"]["Q01"] == "D"
    assert result["part1"]["Q02"] == "A"
    
    # Check Part 2 formatting
    assert result["part2"]["Q01"] == "C"
    assert result["part2"]["Q02"] == "D"


def test_invalid_qr_payload():
    """Test that it fails gracefully if it scans a random QR code."""
    # Missing the | separators
    payload_wrong_format = "This is a link to my website: http://example.com"
    result = parse_qr_payload(payload_wrong_format)
    assert result is None
    
    # Only 2 sections instead of 3
    payload_missing_part = "AI Quiz SP2026 Set-C | Part-I: Q1=D"
    result = parse_qr_payload(payload_missing_part)
    assert result is None
