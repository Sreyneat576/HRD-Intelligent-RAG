"""Structure-aware chunking.

* Prose: blocks under the same heading (and page) are joined, then split with LangChain's
  RecursiveCharacterTextSplitter measured in bge-m3 tokens (paragraph -> line -> sentence -> word).
* Slides: one chunk per slide; consecutive small slides are merged up to `slide_merge_tokens`.
* Tables: row groups with the header repeated, never cut mid-row.
* Every chunk starts with a context header, e.g. "[File: Policy.pdf | Section: Allowance > IT | Page 3]",
  so the filename / section / location are part of what gets embedded and keyword-indexed.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion.loaders.base import markdown_row, table_to_markdown
from app.llm.tokenizer import TokenCounter

_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


def _range_label(name: str, first: int, last: int) -> str:
    """'Slide 3' or 'Slides 3-4' style label."""
    return f"{name} {first}" if first == last else f"{name}s {first}-{last}"


def context_header(filename: str, meta: dict) -> str:
    """The '[File: ... | Section: ... | Page n]' line put in front of every chunk."""
    parts = [f"File: {filename}"]
    if meta.get("sheet"):
        parts.append(f"Sheet: {meta['sheet']}")
    section = " > ".join((meta.get("heading_path") or [])[-3:])
    if section and section != meta.get("sheet") and meta.get("block") != "slide":
        parts.append(f"Section: {section}")
    if meta.get("block") == "slide":
        parts.append(_range_label("Slide", meta["page"], meta.get("last_page", meta["page"])))
        if meta.get("heading_path") and not meta["heading_path"][0].startswith("Slide "):
            parts.append(f"Title: {meta['heading_path'][0]}")
    elif meta.get("page") and not meta.get("sheet"):
        parts.append(f"Page {meta['page']}")
    if meta.get("row_start"):
        parts.append(_range_label("Row", meta["row_start"], meta["row_end"]))
    return "[" + " | ".join(parts) + "]"


class StructureAwareSplitter:
    def __init__(
        self,
        token_counter: TokenCounter,
        max_tokens: int = 450,
        overlap_tokens: int = 60,
        slide_merge_tokens: int = 300,
        table_rows_per_chunk: int = 25,
    ):
        """Chunk size limits (in tokens) and the LangChain recursive splitter used for prose."""
        self.count = token_counter
        self.max_tokens = max_tokens
        self.slide_merge_tokens = slide_merge_tokens
        self.table_rows_per_chunk = table_rows_per_chunk
        self.prose = RecursiveCharacterTextSplitter(
            chunk_size=max_tokens,
            chunk_overlap=overlap_tokens,
            length_function=token_counter,
            separators=_SEPARATORS,
            keep_separator="end",
            strip_whitespace=True,
        )

    # ------------------------------------------------------------------ public
    def split(self, blocks: list[Document], filename: str) -> list[Document]:
        """Turn a document's blocks into chunks (prose, slides and tables each have their own rule)."""
        self._filename = filename
        self._chunks: list[Document] = []
        prose: list[Document] = []
        slides: list[Document] = []

        def key(doc: Document):
            """Blocks with the same key (section, page, sheet) can share a chunk."""
            m = doc.metadata
            return tuple(m.get("heading_path") or []), m.get("page"), m.get("sheet"), m.get("kind")

        for block in blocks:
            kind = block.metadata.get("block", "text")
            if kind == "slide":
                self._flush_prose(prose)
                self._add_slide(block, slides)
            elif kind == "table":
                self._flush_prose(prose)
                self._flush_slides(slides)
                self._split_table(block)
            else:
                self._flush_slides(slides)
                if prose and key(prose[-1]) != key(block):
                    self._flush_prose(prose)
                prose.append(block)
        self._flush_prose(prose)
        self._flush_slides(slides)
        for index, chunk in enumerate(self._chunks):
            chunk.metadata["chunk_index"] = index
        return self._chunks

    # ----------------------------------------------------------------- helpers
    def _emit(self, body: str, meta: dict) -> None:
        """Add one chunk: context header + body, with its token count and section path."""
        body = body.strip()
        if not body:
            return
        meta = {k: v for k, v in meta.items() if k not in ("table_header", "table_rows")}
        content = f"{context_header(self._filename, meta)}\n{body}"
        meta["token_count"] = self.count(content)
        meta["section_path"] = " > ".join(meta.get("heading_path") or []) or None
        self._chunks.append(Document(page_content=content, metadata=meta))

    def _flush_prose(self, prose: list[Document]) -> None:
        """Join the buffered prose blocks of one section and split them into chunks."""
        if not prose:
            return
        meta = dict(prose[0].metadata)
        confidences = [b.metadata["ocr_confidence"] for b in prose if b.metadata.get("ocr_confidence") is not None]
        if confidences:
            meta["ocr_confidence"] = min(confidences)
        # List items stay on consecutive lines; paragraphs are separated by a blank line.
        text = prose[0].page_content
        for block in prose[1:]:
            text += ("\n" if block.page_content.startswith("- ") else "\n\n") + block.page_content
        for piece in self._split_text(text):
            self._emit(piece, meta)
        prose.clear()

    def _split_text(self, text: str) -> list[str]:
        """Recursive split by paragraph / line / sentence / word, hard token split as last resort."""
        pieces = []
        for piece in self.prose.split_text(text):
            # A piece can still be too long when it has no separators at all (e.g. one long OCR line).
            if self.count(piece) > self.max_tokens * 1.2:
                pieces.extend(self.count.split_by_tokens(piece, self.max_tokens))
            else:
                pieces.append(piece)
        return pieces

    def _add_slide(self, slide: Document, buffer: list[Document]) -> None:
        """Buffer a slide for merging, or split it on its own when it is too long."""
        tokens = self.count(slide.page_content)
        if tokens > self.max_tokens:
            self._flush_slides(buffer)
            for piece in self._split_text(slide.page_content):
                self._emit(piece, dict(slide.metadata))
            return
        buffered = sum(self.count(s.page_content) for s in buffer)
        if buffer and buffered + tokens > self.slide_merge_tokens:
            self._flush_slides(buffer)
        buffer.append(slide)

    def _flush_slides(self, buffer: list[Document]) -> None:
        """Emit the buffered slides as one chunk."""
        if not buffer:
            return
        meta = dict(buffer[0].metadata)
        meta["last_page"] = buffer[-1].metadata["page"]
        meta["slides"] = [s.metadata["page"] for s in buffer]
        self._emit("\n\n".join(s.page_content for s in buffer), meta)
        buffer.clear()

    def _split_table(self, block: Document) -> None:
        """Cut a table into row groups that fit the token limit, header repeated in each."""
        header: list[str] = block.metadata["table_header"]
        rows: list[list[str]] = block.metadata["table_rows"]
        header_tokens = self.count(table_to_markdown(header, []))
        group: list[list[str]] = []
        group_tokens = header_tokens
        start = 1

        def flush() -> None:
            """Emit the current row group as a chunk and start a new one."""
            nonlocal group, group_tokens, start
            if group:
                meta = dict(block.metadata, row_start=start, row_end=start + len(group) - 1)
                self._emit(table_to_markdown(header, group), meta)
                start += len(group)
            group, group_tokens = [], header_tokens

        for row in rows:
            row_tokens = self.count(markdown_row(row))
            if group and (group_tokens + row_tokens > self.max_tokens or len(group) >= self.table_rows_per_chunk):
                flush()
            group.append(row)
            group_tokens += row_tokens
        flush()
