from pathlib import Path
from typing import List, Union

from paddleocr import PaddleOCR
from pdf2image import convert_from_path

_ocr: PaddleOCR = None


def _get_ocr() -> PaddleOCR:
    global _ocr
    if _ocr is None:
        # use_angle_cls: corrects rotated text (0/90/180/270)
        # lang: 'en' for English, 'korean' for Korean+English
        _ocr = PaddleOCR(use_angle_cls=True, lang="en")
    return _ocr


def _image_path_to_text(image_path: str) -> str:
    """Run PaddleOCR on an image file path and return concatenated text."""
    result = _get_ocr().ocr(image_path, cls=True)
    if not result or not result[0]:
        return ""
    lines = [line[1][0] for line in result[0] if line[1][1] > 0.5]  # confidence > 0.5
    return "\n".join(lines)


def extract_from_image(path: Union[str, Path]) -> str:
    return _image_path_to_text(str(path))


def extract_from_pdf(path: Union[str, Path]) -> List[str]:
    """PDF → list of OCR text strings, one per page."""
    import tempfile, os
    pages = convert_from_path(str(path), dpi=300)
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, page in enumerate(pages):
            img_path = os.path.join(tmp, f"page_{i}.jpg")
            page.save(img_path, "JPEG")
            results.append(_image_path_to_text(img_path))
    return results


def extract(path: Union[str, Path]) -> List[str]:
    """Dispatch by file type. Returns list of page texts."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return extract_from_pdf(path)
    elif suffix in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}:
        return [extract_from_image(path)]
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m ingestion.ocr.extractor <file>")
        sys.exit(1)

    pages = extract(sys.argv[1])
    for i, text in enumerate(pages):
        print(f"\n--- Page {i + 1} ---\n{text}")
