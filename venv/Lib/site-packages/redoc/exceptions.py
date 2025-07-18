"""Custom exceptions for the Redoc package."""


class RedocError(Exception):
    """Base exception for all Redoc-related errors."""
    pass


class ConversionError(RedocError):
    """Raised when document conversion fails."""
    pass


class UnsupportedFormatError(RedocError):
    """Raised when an unsupported format is requested."""
    pass


class OCRProcessingError(RedocError):
    """Raised when OCR processing fails."""
    pass


class TemplateError(RedocError):
    """Raised when there's an error with document templates."""
    pass


class ValidationError(RedocError):
    """Raised when input validation fails."""
    pass
