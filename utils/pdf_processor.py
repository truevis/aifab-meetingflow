"""PDF text-extraction utilities using PyMuPDF."""

from __future__ import annotations

import os
from typing import Callable, Union

import pymupdf


PdfSource = Union[str, os.PathLike[str], bytes]
PageCallback = Callable[[int, int], None]


def _open_pdf(pdf_source: PdfSource) -> pymupdf.Document:
    """Open a PDF from a filesystem path or uploaded bytes."""
    if isinstance(pdf_source, bytes):
        return pymupdf.open(stream=pdf_source, filetype="pdf")
    return pymupdf.open(pdf_source)


def _first_page_png(pdf: pymupdf.Document, zoom: float = 1.5) -> bytes:
    """Render the first PDF page to PNG bytes for a visual check."""
    if len(pdf) == 0:
        raise ValueError("PDF has no pages")
    pixmap = pdf[0].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    return pixmap.tobytes("png")


def extract_pages_and_first_page_png(
    pdf_source: PdfSource,
    page_callback: PageCallback | None = None,
) -> tuple[list[str], bytes]:
    """Extract page text and a first-page PNG preview from one PDF open."""
    with _open_pdf(pdf_source) as pdf:
        preview = _first_page_png(pdf)
        page_count = len(pdf)
        pages: list[str] = []
        for index, page in enumerate(pdf, start=1):
            pages.append(page.get_text())
            if page_callback:
                page_callback(index, page_count)
        return pages, preview


def extract_pages_from_pdf(
    pdf_source: PdfSource,
    page_callback: PageCallback | None = None,
) -> list[str]:
    """Extract text from every page while preserving page boundaries."""
    with _open_pdf(pdf_source) as pdf:
        page_count = len(pdf)
        pages: list[str] = []
        for index, page in enumerate(pdf, start=1):
            pages.append(page.get_text())
            if page_callback:
                page_callback(index, page_count)
        return pages


def extract_text_from_pdf(
    pdf_path: PdfSource,
    page_callback: PageCallback | None = None,
) -> str:
    """Extract all text from a PDF."""
    return "\n".join(extract_pages_from_pdf(pdf_path, page_callback=page_callback))


def extract_text_from_pdf_page(pdf_path: PdfSource, page_num: int) -> str:
    """Extract text from a zero-indexed PDF page."""
    with _open_pdf(pdf_path) as pdf:
        return pdf[page_num].get_text()


def get_pdf_page_count(pdf_path: PdfSource) -> int:
    """Return the number of pages in a PDF."""
    with _open_pdf(pdf_path) as pdf:
        return len(pdf)
