class IngestionError(Exception):
    """A document could not be indexed. `retryable=False` means retrying will not help
    (corrupt file, no extractable text, OCR disabled, ...)."""

    def __init__(self, message: str, *, retryable: bool = False):
        """`message` is shown to users as the document's error; `retryable` tells the worker to try again."""
        super().__init__(message)
        self.retryable = retryable
