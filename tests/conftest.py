"""Shared fixtures: sample files of every supported type, generated on the fly."""

import os

import pytest

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from app.core.config import Settings  # noqa: E402
from app.llm.tokenizer import get_token_counter  # noqa: E402


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings(_env_file=None)


@pytest.fixture(scope="session")
def token_counter(settings):
    return get_token_counter(settings.tokenizer_name)


@pytest.fixture
def sample_pdf(tmp_path):
    """Two-page PDF: a big heading, body text, a sub-heading and a table."""
    import pymupdf

    path = tmp_path / "policy.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 80), "Allowance Policy", fontsize=20)
    page.insert_text((72, 120), "Students receive a meal allowance every month.", fontsize=11)
    page.insert_text((72, 140), "The IT score allowance depends on the monthly IT score.", fontsize=11)
    page.insert_text((72, 180), "IT Score Allowance", fontsize=15)
    page.insert_text((72, 210), "A score of 90 or more gives the highest allowance.", fontsize=11)
    page2 = doc.new_page()
    page2.insert_text((72, 80), "Korean Language Allowance", fontsize=15)
    page2.insert_text((72, 110), "Korean allowance is paid to students who pass the Korean test.", fontsize=11)
    doc.save(path)
    return path


@pytest.fixture
def sample_docx(tmp_path):
    from docx import Document

    path = tmp_path / "handbook.docx"
    doc = Document()
    doc.add_heading("Student Handbook", level=1)
    doc.add_paragraph("Students must attend every class.")
    doc.add_heading("Attendance", level=2)
    doc.add_paragraph("Late arrival three times counts as one absence.")
    table = doc.add_table(rows=3, cols=2)
    for r, (a, b) in enumerate([("Status", "Meaning"), ("P", "Present"), ("A", "Absent")]):
        table.cell(r, 0).text, table.cell(r, 1).text = a, b
    doc.save(path)
    return path


@pytest.fixture
def sample_pptx(tmp_path):
    from pptx import Presentation

    path = tmp_path / "lesson.pptx"
    prs = Presentation()
    for title, body in [("Spring Boot Basics", "Controllers handle HTTP requests"), ("JPA", "Entities map to tables")]:
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = title
        slide.placeholders[1].text = body
        slide.notes_slide.notes_text_frame.text = f"Explain {title} with an example"
    prs.save(path)
    return path


@pytest.fixture
def sample_xlsx(tmp_path):
    from openpyxl import Workbook

    path = tmp_path / "scores.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.title = "Scores"
    ws.append(["Gen 12 score sheet"])  # title row above the header
    ws.append(["Name", "Class", "Score"])
    for i in range(60):
        ws.append([f"Student {i}", "PP" if i % 2 else "SR", 50 + i % 50])
    wb.save(path)
    return path


@pytest.fixture
def sample_csv(tmp_path):
    path = tmp_path / "attendance.csv"
    rows = ["Name,Date,Status"] + [f"Student {i},2026-10-0{i % 9 + 1},{'Present' if i % 3 else 'Absent'}" for i in range(30)]
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


@pytest.fixture
def sample_png(tmp_path):
    from PIL import Image

    path = tmp_path / "scan.png"
    Image.new("RGB", (200, 100), "white").save(path)
    return path
