"""Shared pieces for the file loaders.

Every loader is a LangChain `BaseLoader` that yields one `Document` per *structural block*:

    metadata["block"]         "text" | "table" | "slide"
    metadata["heading_path"]  e.g. ["2. Core Features", "2.2 Student Management"]
    metadata["page"]          page number (PDF) or slide number (PPTX), 1-based
    metadata["sheet"]         sheet name (XLSX/CSV)
    metadata["table_header"], metadata["table_rows"]   for table blocks

The splitter (app/ingestion/splitter.py) turns blocks into retrieval chunks.
"""

from collections.abc import Iterable
from pathlib import Path

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from app.ingestion.ocr import NoOcr, OcrEngine


def make_block(
    text: str,
    block_type: str = "text",
    *,
    heading_path: Iterable[str] = (),
    page: int | None = None,
    sheet: str | None = None,
    **meta,
) -> Document:
    """Create one structural block (`block_type` is "text", "table" or "slide"); extra keyword
    arguments are stored as metadata."""
    return Document(
        page_content=text,
        metadata={"block": block_type, "heading_path": list(heading_path), "page": page, "sheet": sheet, **meta},
    )


def make_table_block(header: list[str], rows: list[list[str]], **kwargs) -> Document:
    """Create a table block: Markdown text plus the header and rows kept in metadata."""
    return make_block(
        table_to_markdown(header, rows), "table", table_header=header, table_rows=rows, **kwargs
    )


def markdown_row(cells: list[str]) -> str:
    """One Markdown table row; pipes and newlines inside cells are escaped."""
    return "| " + " | ".join(c.replace("|", "\\|").replace("\n", " ") for c in cells) + " |"


def table_to_markdown(header: list[str], rows: list[list[str]]) -> str:
    """Render a header and rows as a Markdown table."""
    separator = "| " + " | ".join("---" for _ in header) + " |"
    return "\n".join([markdown_row(header), separator, *(markdown_row(r) for r in rows)])


def clean_cell(value) -> str:
    """Cell value as a single-line string (None becomes empty)."""
    return "" if value is None else " ".join(str(value).split())


class HeadingStack:
    """Tracks the current heading path while walking a document top to bottom."""

    def __init__(self) -> None:
        """Start with an empty heading path."""
        self._stack: list[tuple[int, str]] = []

    def push(self, level: int, title: str) -> None:
        """Enter a heading: drop headings of the same or deeper level, then add this one."""
        while self._stack and self._stack[-1][0] >= level:
            self._stack.pop()
        self._stack.append((level, title.strip()))

    @property
    def path(self) -> list[str]:
        """Current heading titles from top level down."""
        return [title for _, title in self._stack]


class FileLoader(BaseLoader):
    def __init__(self, path: str | Path, ocr: OcrEngine | None = None):
        """Remember the file path and the OCR engine used for images / scanned pages."""
        self.path = Path(path)
        self.ocr = ocr or NoOcr()
        self.page_count: int | None = None
