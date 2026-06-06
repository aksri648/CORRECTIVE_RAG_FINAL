import os
from typing import Callable, List, Optional

import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


TEXT_FALLBACK_MIN_CHARS = 100

_OCR_READER = None
_OCR_INIT_ERROR: Optional[Exception] = None


def _get_ocr_reader():
    global _OCR_READER, _OCR_INIT_ERROR
    if _OCR_READER is not None:
        return _OCR_READER
    if _OCR_INIT_ERROR is not None:
        raise _OCR_INIT_ERROR

    try:
        import easyocr
    except ImportError as exc:
        raise ImportError(
            "easyocr is not installed. Run `pip install easyocr pdf2image`."
        ) from exc

    try:
        _OCR_READER = easyocr.Reader(["en"], gpu=False, verbose=False)
    except Exception as exc:
        _OCR_INIT_ERROR = exc
        raise
    return _OCR_READER


def _looks_image_based(pdf_path: str) -> bool:
    try:
        pages = PyPDFLoader(pdf_path).load()
    except Exception:
        return True
    if not pages:
        return True
    total = " ".join(p.page_content for p in pages).strip()
    return len(total) < TEXT_FALLBACK_MIN_CHARS


def _ocr_pdf_pages(
    pdf_path: str,
    source_name: str,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> List[Document]:
    try:
        from pdf2image import convert_from_path
    except ImportError as exc:
        raise ImportError(
            "pdf2image is not installed. Run `pip install pdf2image` and install "
            "the poppler system package (e.g. `apt-get install -y poppler-utils`)."
        ) from exc

    if progress_cb:
        progress_cb("Loading EasyOCR model (first time downloads ~100 MB)...")
    reader = _get_ocr_reader()

    if progress_cb:
        progress_cb("Rasterizing PDF pages for OCR...")
    images = convert_from_path(pdf_path, dpi=200)

    docs: List[Document] = []
    total_pages = len(images)
    for page_num, image in enumerate(images, start=1):
        if progress_cb:
            progress_cb(f"Running OCR on page {page_num}/{total_pages}...")
        img_array = np.array(image.convert("RGB"))
        text_blocks = reader.readtext(img_array, detail=0, paragraph=True)
        page_text = "\n".join(text_blocks).strip()

        docs.append(
            Document(
                page_content=page_text,
                metadata={
                    "source": source_name,
                    "page": page_num,
                    "ocr": True,
                },
            )
        )

    if progress_cb:
        chars = sum(len(d.page_content) for d in docs)
        progress_cb(f"OCR finished: {total_pages} page(s), {chars} characters extracted.")
    return docs


def load_pdf(
    pdf_path: str,
    source_name: str,
    force_ocr: bool = False,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> List[Document]:
    if not force_ocr and not _looks_image_based(pdf_path):
        if progress_cb:
            progress_cb("Text-based PDF detected. Extracting text with PyPDFLoader...")
        pages = PyPDFLoader(pdf_path).load()
        for page in pages:
            page.metadata["source"] = source_name
            page.metadata["ocr"] = False
        if progress_cb:
            chars = sum(len(p.page_content) for p in pages)
            progress_cb(f"Text extraction finished: {len(pages)} page(s), {chars} characters.")
        return pages

    if progress_cb and not force_ocr:
        progress_cb("PDF appears to be scanned/image-based. Falling back to EasyOCR.")
    return _ocr_pdf_pages(pdf_path, source_name, progress_cb=progress_cb)
