import io
from dataclasses import dataclass
from pathlib import Path

from providers import get_vision_provider

_FIGURE_PROMPT = (
    "This is a page from a ML/CV/CG research paper. "
    "If this page contains an architecture diagram, model overview, or key figure, "
    "describe it concisely: what components are shown, how data flows, and what the figure illustrates. "
    "If the page has no significant figure, reply with exactly: NO_FIGURE"
)

@dataclass
class FigureDescription:
    page: int
    description: str

    def __str__(self) -> str:
        return f"[Page {self.page}] {self.description}"

def extract_figures(pdf_path: Path, max_pages: int = 8) -> list[FigureDescription]:
    """
    Render PDF pages as images using pymupdf and describe architecture figures via Vision provider.
    Returns only pages that contain meaningful figures.
    """
    try:
        import fitz
    except ImportError:
        return [FigureDescription(page=0, description="pymupdf not installed — run: pip install pymupdf")]

    doc = fitz.open(str(pdf_path))
    provider = get_vision_provider()
    results: list[FigureDescription] = []

    for page_num in range(min(max_pages, len(doc))):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("jpeg")

        description = provider.analyze(img_bytes, _FIGURE_PROMPT).strip()
        if description and description != "NO_FIGURE":
            results.append(FigureDescription(page=page_num + 1, description=description))

    doc.close()
    return results