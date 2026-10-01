"""The one error the EDGAR package raises, so callers catch a single type."""


class EdgarError(RuntimeError):
    """Raised when EDGAR cannot serve what we asked for, or served something malformed."""
