"""
Redoc - Universal Document Converter

A powerful, modular document conversion framework supporting multiple formats including
PDF, HTML, XML, JSON, DOCX, and EPUB with OCR and AI-powered content generation.
"""

__version__ = "0.1.0"

from .core import Redoc
from .exceptions import RedocError

__all__ = ['Redoc', 'RedocError']
