"""Quick test: moondream vision quality on a real paper PDF."""
import sys
from pathlib import Path

# default to the paper we have
DEFAULT_PDF = Path(__file__).parent / "storage" / "papers" / "2310.06825.pdf"


def main():
    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PDF
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        sys.exit(1)

    try:
        import fitz
    except ImportError:
        print("Missing: pip install pymupdf")
        sys.exit(1)

    from providers.ollama import OllamaVisionProvider
    from config import OLLAMA_BASE_URL, OLLAMA_VISION_MODEL

    provider = OllamaVisionProvider(base_url=OLLAMA_BASE_URL, model=OLLAMA_VISION_MODEL)
    print(f"Model : {OLLAMA_VISION_MODEL}")
    print(f"PDF   : {pdf_path.name}")
    print("-" * 60)

    prompt = (
        "This is a page from a ML/CV/CG research paper. "
        "If this page contains an architecture diagram, model overview, or key figure, "
        "describe it concisely: what components are shown, how data flows, and what the figure illustrates. "
        "If the page has no significant figure, reply with exactly: NO_FIGURE"
    )

    doc = fitz.open(str(pdf_path))
    tested = 0
    for page_num in range(min(8, len(doc))):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=72)
        img_bytes = pix.tobytes("jpeg")

        print(f"\n[Page {page_num + 1}]")
        result = provider.analyze(img_bytes, "What do you see in this image?").strip()
        print(result if result else "(empty)")
        tested += 1

    doc.close()
    print(f"\n--- Done ({tested} pages tested) ---")


if __name__ == "__main__":
    main()
