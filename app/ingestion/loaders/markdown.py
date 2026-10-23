"""Markdown / plain text -> blocks. Used for .md and .txt files and for PP-StructureV3 OCR output."""

import re

from langchain_core.documents import Document

from app.ingestion.loaders.base import HeadingStack, clean_cell, make_block, make_table_block

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_TABLE_SEPARATOR = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
_FENCE = re.compile(r"^(```|~~~)")


def _split_row(line: str) -> list[str]:
    """Split a Markdown table row into cleaned cells."""
    line = line.strip().removeprefix("|").removesuffix("|")
    return [clean_cell(c) for c in re.split(r"(?<!\\)\|", line)]


def markdown_to_blocks(
    text: str,
    *,
    page: int | None = None,
    stack: HeadingStack | None = None,
    parse_markdown: bool = True,
    **meta,
) -> list[Document]:
    """Paragraphs separated by blank lines become text blocks. With `parse_markdown`, `#` headings
    update the heading path, pipe tables become table blocks and fenced code stays in one block."""
    stack = stack or HeadingStack()
    blocks: list[Document] = []
    paragraph: list[str] = []
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    def flush() -> None:
        """Emit the collected paragraph lines as one text block."""
        body = "\n".join(paragraph).strip()
        if body:
            blocks.append(make_block(body, heading_path=stack.path, page=page, **meta))
        paragraph.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        if parse_markdown and _FENCE.match(line.strip()):
            flush()
            fence = [line]
            i += 1
            while i < len(lines):
                fence.append(lines[i])
                if _FENCE.match(lines[i].strip()):
                    break
                i += 1
            paragraph.extend(fence)
            flush()
        elif parse_markdown and (m := _HEADING.match(line.strip())):
            flush()
            stack.push(len(m.group(1)), m.group(2))
        elif (
            parse_markdown
            and line.strip().startswith("|")
            and i + 1 < len(lines)
            and _TABLE_SEPARATOR.match(lines[i + 1].strip())
        ):
            flush()
            header = _split_row(line)
            rows = []
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(_split_row(lines[i]))
                i += 1
            blocks.append(make_table_block(header, rows, heading_path=stack.path, page=page, **meta))
            continue
        elif not line.strip():
            flush()
        else:
            paragraph.append(line.rstrip())
        i += 1
    flush()
    return blocks
