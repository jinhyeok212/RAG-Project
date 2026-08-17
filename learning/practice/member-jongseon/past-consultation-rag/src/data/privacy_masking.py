from __future__ import annotations

import re


PATTERNS = [
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[EMAIL]"),
    (re.compile(r"\b01[016789][-\s.]?\d{3,4}[-\s.]?\d{4}\b"), "[PHONE]"),
    (re.compile(r"\b0\d{1,2}[-\s.]?\d{3,4}[-\s.]?\d{4}\b"), "[TEL]"),
    (re.compile(r"\b\d{6}[-\s]?[1-4]\d{6}\b"), "[RRN]"),
    (re.compile(r"\b\d{3,6}[-_]\d{3,8}[-_]\d{2,8}\b"), "[ORDER_ID]"),
    (re.compile(r"\b(?:주문|오더|예약|송장|운송장)\s*(?:번호|No\.?)?\s*[:#]?\s*[A-Za-z0-9-]{5,}\b", re.I), "[ORDER_ID]"),
]


def mask_text(text: str) -> str:
    masked = text or ""
    for pattern, repl in PATTERNS:
        masked = pattern.sub(repl, masked)
    return masked


def detect_privacy_patterns(text: str) -> list[str]:
    labels = []
    for pattern, repl in PATTERNS:
        if pattern.search(text or ""):
            labels.append(repl.strip("[]"))
    return labels
