"""File-type registry: extension -> LangChain loader."""

from pathlib import Path

from langchain_core.documents import Document

from app.core.config import Settings
from app.ingestion.errors import IngestionError
from app.ingestion.loaders.base import FileLoader
from app.ingestion.loaders.office import DocxLoader, PptxLoader
from app.ingestion.loaders.pdf import PdfLoader
from app.ingestion.loaders.spreadsheet import SpreadsheetLoader
from app.ingestion.loaders.text import ImageLoader, TextLoader
from app.ingestion.ocr import OcrEngine

LOADERS: dict[str, type[FileLoader]] = {
    ".pdf": PdfLoader,
    ".docx": DocxLoader,
    ".pptx": PptxLoader,
    ".xlsx": SpreadsheetLoader,
    ".csv": SpreadsheetLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
    ".markdown": TextLoader,
    ".png": ImageLoader,
    ".jpg": ImageLoader,
    ".jpeg": ImageLoader,
}

SUPPORTED_EXTENSIONS = frozenset(LOADERS)


def build_loader(path: str | Path, extension: str, ocr: OcrEngine, settings: Settings) -> FileLoader:
    """Pick the loader for a file extension (raises IngestionError when unsupported)."""
    loader_cls = LOADERS.get(extension.lower())
    if loader_cls is None:
        raise IngestionError(f"Unsupported file type: {extension}")
    if loader_cls is PdfLoader:
        return PdfLoader(path, ocr, ocr_min_page_chars=settings.ocr_min_page_chars, ocr_dpi=settings.ocr_dpi)
    return loader_cls(path, ocr)


def load_blocks(path: str | Path, extension: str, ocr: OcrEngine, settings: Settings) -> tuple[list[Document], int | None]:
    """Load a file into structural blocks; returns (blocks, page/slide/sheet count)."""
    loader = build_loader(path, extension, ocr, settings)
    blocks = loader.load()
    return blocks, loader.page_count
