"""
constants.py — Fixed Values That Never Change
===============================================

PURPOSE:
    Stores values that are determined by the quiz sheet format itself.
    These are NOT tunable settings — they are facts about the quiz.

DIFFERENCE FROM config.py:
    config.py  → settings you might CHANGE (thresholds, paths, log level)
    constants.py → FIXED facts about the quiz format (4 options, A/B/C/D)
"""

# =============================================================
# QUIZ STRUCTURE
# =============================================================

# The four bubble options on every question
OPTION_LABELS = ["A", "B", "C", "D"]

# Question labels for each part (Q01 through Q08)
QUESTION_LABELS = [f"Q{str(i).zfill(2)}" for i in range(1, 9)]
# Result: ["Q01", "Q02", "Q03", "Q04", "Q05", "Q06", "Q07", "Q08"]

# The two parts of the quiz
PART_LABELS = ["part1", "part2"]

# =============================================================
# ANSWER STATUS MARKERS
# =============================================================

# When a student doesn't fill any bubble for a question
UNATTEMPTED = None

# When a student fills more than one bubble (invalid answer)
INVALID = "INVALID"

# =============================================================
# GRADE REPORT SYMBOLS
# =============================================================

# Symbols used in the per-question breakdown
SYMBOL_CORRECT = "✓"       # Student's answer matches the key
SYMBOL_INCORRECT = "✗"     # Student's answer doesn't match
SYMBOL_UNATTEMPTED = "—"   # Student didn't fill any bubble
SYMBOL_INVALID = "⚠"       # Student filled multiple bubbles

# =============================================================
# SUPPORTED IMAGE FORMATS
# =============================================================

SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png"]
