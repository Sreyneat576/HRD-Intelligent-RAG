"""PDF loader (PyMuPDF).

* Headings come from the PDF outline (bookmarks / TOC) when it has one; otherwise from font size
  relative to the body text. They are kept as a heading path, e.g. ["2. Core Features", "2.7. AI Chatbot"].
* Tables found by PyMuPDF become table blocks (rows kept together by the splitter).
* Header/footer lines repeated on most pages are dropped.
* Pages with almost no text layer (scans) are rendered and sent to OCR.
"""

import io
import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass

import pymupdf
from langchain_core.documents import Document
from PIL import Image

from app.ingestion.errors import IngestionError
from app.ingestion.loaders.base import FileLoader, HeadingStack, clean_cell, make_block, make_table_block
from app.ingestion.loaders.markdown import markdown_to_blocks

_BULLET_GLYPHS = "•●○▪▫■□◦‣⁃∙➢✓✔*"
_GLYPH_BULLET = re.compile(rf"^\s*[{_BULLET_GLYPHS}\-–]\s*")
_NUMBERED = re.compile(r"^\s*(\d+[.)]|[a-z][.)])\s+")
_DOT_LEADER = re.compile(r"\s*\.{4,}\s*")
_EDGE_RATIO = 0.08  # top/bottom 8% of the page is where headers/footers live


@dataclass
class _Line:
    text: str
    size: float
    bold: bool
    y0: float
    y1: float
    block_no: int


def _normalise(text: str) -> str:
    """Header/footer comparison key: lower case, digits replaced (page numbers differ)."""
    return re.sub(r"\d+", "#", text.strip().lower())


def _clean_line(text: str) -> str:
    """Remove zero-width spaces, turn bullet glyphs into '- ' and collapse dot leaders."""
    text = text.replace("​", "").replace(" ", " ")
    text = _GLYPH_BULLET.sub("- ", text)
    text = _DOT_LEADER.sub(" ... ", text)
    return " ".join(text.split())


def _title_key(text: str) -> str:
    """Comparison key for matching a line against an outline title."""
    return " ".join(text.lower().replace("​", "").split()).rstrip(" :.")


class PdfLoader(FileLoader):
    def __init__(self, *args, ocr_min_page_chars: int = 50, ocr_dpi: int = 200, **kwargs):
        """OCR thresholds: pages with fewer text characters are rendered at `ocr_dpi` and OCR'd."""
        super().__init__(*args, **kwargs)
        self.ocr_min_page_chars = ocr_min_page_chars
        self.ocr_dpi = ocr_dpi

    def _page_lines(self, page: pymupdf.Page) -> list[_Line]:
        """Text lines of a page with average font size, boldness and position."""
        lines = []
        for block_no, block in enumerate(page.get_text("dict", sort=True)["blocks"]):
            if block.get("type") != 0:
                continue
            for line in block["lines"]:
                spans = [s for s in line["spans"] if s["text"].strip(" ​")]
                if not spans:
                    continue
                text = _clean_line("".join(s["text"] for s in spans))
                if not text or text == "-":
                    continue
                chars = sum(len(s["text"]) for s in spans)
                size = sum(s["size"] * len(s["text"]) for s in spans) / max(chars, 1)
                bold = all(s["flags"] & 16 or "bold" in s["font"].lower() for s in spans)
                lines.append(_Line(text, round(size, 1), bold, line["bbox"][1], line["bbox"][3], block_no))
        return lines

    def _outline(self, pdf: pymupdf.Document) -> dict[int, list[tuple[str, int]]]:
        """page -> [(title key, level)] from the PDF bookmarks, if the PDF has a usable outline."""
        entries = [(lvl, _title_key(title), page) for lvl, title, page in pdf.get_toc(simple=True)]
        entries = [e for e in entries if e[1]]
        outline: dict[int, list[tuple[str, int]]] = {}
        if len(entries) >= 3:
            for level, key, page in entries:
                outline.setdefault(page, []).append((key, level))
        return outline

    def lazy_load(self) -> Iterator[Document]:
        """Yield text and table blocks page by page with the current heading path."""
        try:
            pdf = pymupdf.open(self.path)
        except Exception as exc:
            raise IngestionError(f"Cannot open PDF: {exc}") from exc
        if pdf.needs_pass:
            raise IngestionError("PDF is password protected")
        self.page_count = pdf.page_count

        pages = []
        for page in pdf:
            try:
                tables = list(page.find_tables().tables)
            except Exception:
                tables = []
            pages.append((page, self._page_lines(page), tables))

        # Repeated header/footer lines (same text, ignoring digits, at the page edges).
        edge_counts: Counter[str] = Counter()
        for page, lines, _ in pages:
            height = page.rect.height
            edge = {_normalise(l.text) for l in lines if l.y1 < height * _EDGE_RATIO or l.y0 > height * (1 - _EDGE_RATIO)}
            edge_counts.update(edge)
        repeated = {t for t, n in edge_counts.items() if len(pages) >= 3 and n >= len(pages) * 0.5}

        outline = self._outline(pdf)

        # Fallback when there is no outline: body font size = most common size by characters,
        # clearly larger sizes are headings (largest = level 1).
        size_chars: Counter[float] = Counter()
        for _, lines, _ in pages:
            for l in lines:
                size_chars[l.size] += len(l.text)
        body_size = size_chars.most_common(1)[0][0] if size_chars else 10.0
        heading_sizes = sorted({s for s in size_chars if s >= body_size * 1.15}, reverse=True)[:4]

        def heading_level(line: _Line, page_no: int) -> int | None:
            """Heading level of a line (outline match, else font size / bold), or None for body text."""
            if outline:
                key = _title_key(line.text)
                for title, level in outline.get(page_no, []):
                    if key == title or (len(key) >= 8 and title.startswith(key) and len(key) >= len(title) / 2):
                        return level
                return None
            if len(line.text) > 150 or line.text.startswith("- "):
                return None
            if line.size in heading_sizes:
                return heading_sizes.index(line.size) + 1
            if line.bold and line.size >= body_size and len(line.text) <= 80 and not line.text.endswith((".", ",", ";")):
                return len(heading_sizes) + 1
            return None

        stack = HeadingStack()
        for page_no, (page, lines, tables) in enumerate(pages, start=1):
            height = page.rect.height
            lines = [
                l for l in lines
                if not ((l.y1 < height * _EDGE_RATIO or l.y0 > height * (1 - _EDGE_RATIO)) and _normalise(l.text) in repeated)
            ]
            text_chars = sum(len(l.text) for l in lines)
            if text_chars < self.ocr_min_page_chars and page.get_images():
                yield from self._ocr_page(page, page_no, stack)
                continue

            pending_tables = sorted(tables, key=lambda t: t.bbox[1])
            table_rects = [pymupdf.Rect(t.bbox) for t in tables]
            paragraph: list[str] = []
            last_block = None

            def flush() -> Iterator[Document]:
                """Emit the current paragraph as a text block."""
                text = " ".join(paragraph).strip()
                paragraph.clear()
                if text:
                    yield make_block(text, heading_path=stack.path, page=page_no)

            for line in lines:
                while pending_tables and pending_tables[0].bbox[1] <= line.y0:
                    yield from flush()
                    yield from self._table_block(pending_tables.pop(0), stack, page_no)
                mid_y = (line.y0 + line.y1) / 2
                if any(r.y0 <= mid_y <= r.y1 for r in table_rects):
                    continue  # text inside a table is emitted with the table
                level = heading_level(line, page_no)
                if level is not None:
                    yield from flush()
                    stack.push(level, line.text)
                    last_block = None
                    continue
                new_item = line.text.startswith("- ") or _NUMBERED.match(line.text)
                if (last_block is not None and line.block_no != last_block) or new_item:
                    yield from flush()
                paragraph.append(line.text)
                last_block = line.block_no
            yield from flush()
            for table in pending_tables:
                yield from self._table_block(table, stack, page_no)

    def _table_block(self, table, stack: HeadingStack, page_no: int) -> Iterator[Document]:
        """Turn a PyMuPDF table into a table block (first row = header)."""
        rows = [[_clean_line(clean_cell(c)) for c in row] for row in table.extract()]
        rows = [r for r in rows if any(r)]
        if len(rows) < 2:
            if rows:
                yield make_block(" | ".join(c for c in rows[0] if c), heading_path=stack.path, page=page_no)
            return
        yield make_table_block(rows[0], rows[1:], heading_path=stack.path, page=page_no)

    def _ocr_page(self, page: pymupdf.Page, page_no: int, stack: HeadingStack) -> Iterator[Document]:
        """Render a scanned page to an image and OCR it into blocks."""
        pix = page.get_pixmap(dpi=self.ocr_dpi)
        image = Image.open(io.BytesIO(pix.tobytes("png")))
        result = self.ocr.recognize(image)
        meta = {"ocr": True, "ocr_confidence": result.confidence}
        yield from markdown_to_blocks(
            result.text, page=page_no, stack=stack, parse_markdown=result.is_markdown, **meta
        )
