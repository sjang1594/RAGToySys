from typing import List


SEPARATORS = ["\n\n", "\n", " ", ""]


def chunk(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """Recursive character text splitter. Tries separators in order until chunks fit."""
    return _split(text, chunk_size, chunk_overlap, SEPARATORS)


def _split(text: str, chunk_size: int, chunk_overlap: int, separators: List[str]) -> List[str]:
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    separator = separators[-1]  # fallback: split by character
    for sep in separators:
        if sep in text:
            separator = sep
            break

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

            # if part alone exceeds chunk_size, recurse with next separator
            if len(part) > chunk_size:
                next_seps = separators[separators.index(separator) + 1:] if separator in separators[:-1] else [""]
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
