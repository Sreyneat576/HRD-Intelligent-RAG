"""Each loader turns its file type into structural blocks with the right metadata."""

import pytest

from app.ingestion.errors import IngestionError
from app.ingestion.loaders import load_blocks
from app.ingestion.loaders.markdown import markdown_to_blocks
from app.ingestion.ocr import NoOcr
from tests.fakes import FakeOcr


def test_pdf_headings_pages_and_text(sample_pdf, settings):
    blocks, pages = load_blocks(sample_pdf, ".pdf", NoOcr(), settings)
    assert pages == 2
    by_text = {b.page_content: b.metadata for b in blocks}
    meal = by_text["Students receive a meal allowance every month."]
    assert meal["heading_path"] == ["Allowance Policy"] and meal["page"] == 1
    high = next(m for t, m in by_text.items() if t.startswith("A score of 90"))
    assert high["heading_path"] == ["Allowance Policy", "IT Score Allowance"]
    korean = next(m for t, m in by_text.items() if t.startswith("Korean allowance"))
    assert korean["page"] == 2 and korean["heading_path"][-1] == "Korean Language Allowance"


def test_docx_heading_path_and_table(sample_docx, settings):
    blocks, _ = load_blocks(sample_docx, ".docx", NoOcr(), settings)
    text = next(b for b in blocks if b.page_content.startswith("Late arrival"))
    assert text.metadata["heading_path"] == ["Student Handbook", "Attendance"]
    table = next(b for b in blocks if b.metadata["block"] == "table")
    assert table.metadata["table_header"] == ["Status", "Meaning"]
    assert table.metadata["table_rows"] == [["P", "Present"], ["A", "Absent"]]


def test_pptx_one_block_per_slide_with_notes(sample_pptx, settings):
    blocks, pages = load_blocks(sample_pptx, ".pptx", NoOcr(), settings)
    assert pages == 2 and [b.metadata["block"] for b in blocks] == ["slide", "slide"]
    assert blocks[0].metadata["page"] == 1 and blocks[0].metadata["heading_path"] == ["Spring Boot Basics"]
    assert "Controllers handle HTTP requests" in blocks[0].page_content
    assert "Speaker notes: Explain Spring Boot Basics" in blocks[0].page_content


def test_xlsx_detects_header_below_title_row_and_adds_summary(sample_xlsx, settings):
    blocks, _ = load_blocks(sample_xlsx, ".xlsx", NoOcr(), settings)
    summary, table = blocks
    assert summary.metadata["kind"] == "sheet_summary"
    assert "60 rows x 3 columns" in summary.page_content
    assert "Score: numeric, min 50, max 99" in summary.page_content
    assert table.metadata["table_header"] == ["Name", "Class", "Score"]
    assert len(table.metadata["table_rows"]) == 60 and table.metadata["sheet"] == "Scores"


def test_csv(sample_csv, settings):
    blocks, _ = load_blocks(sample_csv, ".csv", NoOcr(), settings)
    table = blocks[-1]
    assert table.metadata["table_header"] == ["Name", "Date", "Status"]
    assert len(table.metadata["table_rows"]) == 30
    assert "Status: Absent (10), Present (20)" in blocks[0].page_content or "Present (20)" in blocks[0].page_content


def test_txt_with_legacy_encoding(tmp_path, settings):
    path = tmp_path / "notes.txt"
    path.write_bytes("Café meeting notes.\n\nSecond paragraph about allowances.".encode("cp1252"))
    blocks, _ = load_blocks(path, ".txt", NoOcr(), settings)
    assert [b.page_content for b in blocks] == ["Café meeting notes.", "Second paragraph about allowances."]


def test_markdown_headings_tables_and_code():
    md = "# Policy\nIntro text.\n\n## Fees\n| Item | Amount |\n|---|---|\n| Meal | 50 |\n\n```\ncode | not a table\n```\n"
    blocks = markdown_to_blocks(md)
    assert blocks[0].page_content == "Intro text." and blocks[0].metadata["heading_path"] == ["Policy"]
    table = blocks[1]
    assert table.metadata["block"] == "table" and table.metadata["heading_path"] == ["Policy", "Fees"]
    assert table.metadata["table_rows"] == [["Meal", "50"]]
    assert blocks[2].page_content.startswith("```")


def test_image_uses_ocr(sample_png, settings):
    blocks, pages = load_blocks(sample_png, ".png", FakeOcr(), settings)
    assert pages == 1 and "Student: Dara" in blocks[0].page_content
    assert blocks[0].metadata["ocr"] is True and blocks[0].metadata["ocr_confidence"] == 0.93


def test_image_without_ocr_fails_cleanly(sample_png, settings):
    with pytest.raises(IngestionError, match="OCR"):
        load_blocks(sample_png, ".png", NoOcr(), settings)


def test_unsupported_type(tmp_path, settings):
    path = tmp_path / "x.exe"
    path.write_bytes(b"MZ")
    with pytest.raises(IngestionError, match="Unsupported"):
        load_blocks(path, ".exe", NoOcr(), settings)
