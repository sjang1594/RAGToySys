from typing import List


SEPARATORS = ["\n\n", "\n", " "]


def chunk(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """Recursive character text splitter. Tries separators in order until chunks fit;
    text with no separator left is cut into fixed-size character windows."""
    return _split(text, chunk_size, chunk_overlap, SEPARATORS)


def _hard_split(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Fixed-size character windows with overlap, for text that contains no separator."""
    step = max(chunk_size - chunk_overlap, 1)
    starts = range(0, max(len(text) - chunk_overlap, 1), step)
    return [piece for piece in (text[i:i + chunk_size].strip() for i in starts) if piece]


def _split(text: str, chunk_size: int, chunk_overlap: int, separators: List[str]) -> List[str]:
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    separator = next((sep for sep in separators if sep in text), None)
    if separator is None:
        # str.split("") raises ValueError, so the character level is handled separately
        return _hard_split(text, chunk_size, chunk_overlap)

    splits = text.split(separator)
    chunks: List[str] = []
    current = ""

    for part in splits:
        candidate = current + (separator if current else "") + part

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            # current is full — save it
            if current.strip():
                chunks.append(current.strip())

            # if part alone exceeds chunk_size, recurse with the remaining separators
            if len(part) > chunk_size:
                next_seps = separators[separators.index(separator) + 1:]
                chunks.extend(_split(part, chunk_size, chunk_overlap, next_seps))
                current = ""
            else:
                # start new current with overlap from previous chunk
                if chunks:
                    overlap_text = chunks[-1][-chunk_overlap:] if chunk_overlap else ""
                    current = overlap_text + (separator if overlap_text else "") + part
                else:
                    current = part

    if current.strip():
        chunks.append(current.strip())

    return chunks
