from __future__ import annotations

import re


def tokenize(text: str, mode: str = "mixed") -> list[str]:
    text = (text or "").lower()
    words = re.findall(r"[가-힣A-Za-z0-9]+", text)
    if mode == "whitespace":
        return words
    tokens = words[:]
    if mode in {"mixed", "char_ngram"}:
        compact = "".join(words)
        for n in (2, 3):
            tokens.extend(compact[i : i + n] for i in range(max(0, len(compact) - n + 1)))
    return [t for t in tokens if t]
