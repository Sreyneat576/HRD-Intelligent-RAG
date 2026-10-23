"""Plain text (.txt), Markdown (.md) and image (.png/.jpg, via OCR) loaders."""

from collections.abc import Iterator

from langchain_core.documents import Document
from PIL import Image, ImageOps, UnidentifiedImageError

from app.ingestion.errors import IngestionError
from app.ingestion.loaders.base import FileLoader
from app.ingestion.loaders.markdown import markdown_to_blocks
from app.ingestion.loaders.spreadsheet import decode_text


class TextLoader(FileLoader):
    def lazy_load(self) -> Iterator[Document]:
        """Decode the file and split it into paragraphs (Markdown structure for .md)."""
        text = decode_text(self.path.read_bytes())
        is_markdown = self.path.suffix.lower() in (".md", ".markdown")
        yield from markdown_to_blocks(text, parse_markdown=is_markdown)


class ImageLoader(FileLoader):
    def lazy_load(self) -> Iterator[Document]:
        """OCR the image (EXIF rotation applied) and split the text into blocks."""
        try:
            image = ImageOps.exif_transpose(Image.open(self.path))
        except (UnidentifiedImageError, OSError) as exc:
            raise IngestionError(f"Cannot open image: {exc}") from exc
        self.page_count = 1
        result = self.ocr.recognize(image)
        yield from markdown_to_blocks(
            result.text,
            page=1,
            parse_markdown=result.is_markdown,
            ocr=True,
            ocr_confidence=result.confidence,
        )
