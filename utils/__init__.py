"""Utility modules for the tax-document classifier."""

from .classifier import build_classification_request, classify_tax_document
from .pdf_processor import (
    extract_pages_from_pdf,
    extract_text_from_pdf,
    extract_text_from_pdf_page,
    get_pdf_page_count,
)

__all__ = [
    "extract_text_from_pdf",
    "extract_text_from_pdf_page",
    "extract_pages_from_pdf",
    "get_pdf_page_count",
    "classify_tax_document",
    "build_classification_request",
]
