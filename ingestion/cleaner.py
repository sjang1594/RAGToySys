import re


def clean(text: str) -> str:
    """Normalize raw OCR output into clean, queryable text."""
    text = _strip_control_chars(text)
    text = _normalize_whitespace(text)
    text = _fix_hyphenation(text)
    text = _remove_noise_lines(text)
    return text.strip()


def _strip_control_chars(text: str) -> str:
    """Remove non-printable characters except newline and tab."""
    return re.sub(r"[^\x09\x0A\x20-\x7E\u00A0-\uFFFF]", "", text)


def _normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs; preserve paragraph breaks (\n\n)."""
    # Collapse multiple spaces/tabs to single space
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ newlines to double newline (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _fix_hyphenation(text: str) -> str:
    """Rejoin words broken across lines by OCR hyphenation (e.g. 'fre-\nquency')."""
    return re.sub(r"-\n(\w)", r"\1", text)


def _remove_noise_lines(text: str) -> str:
    """Drop lines that are pure noise: only symbols, single chars, or whitespace."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Keep empty lines (paragraph breaks)
        if not stripped:
            cleaned.append("")
            continue
        # Drop lines with almost no alphanumeric content
        alnum_ratio = sum(c.isalnum() for c in stripped) / len(stripped)
        if alnum_ratio > 0.2:
            cleaned.append(line)
    return "\n".join(cleaned)
