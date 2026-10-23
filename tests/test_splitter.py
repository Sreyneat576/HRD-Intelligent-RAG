"""Structure-aware chunking rules."""

from app.ingestion.loaders.base import make_block, make_table_block
from app.ingestion.splitter import StructureAwareSplitter, context_header


def make_splitter(token_counter, **kwargs):
    defaults = {"max_tokens": 120, "overlap_tokens": 20, "slide_merge_tokens": 80, "table_rows_per_chunk": 10}
    return StructureAwareSplitter(token_counter, **{**defaults, **kwargs})


def test_long_section_is_split_within_limit_with_overlap(token_counter):
    sentences = [f"Rule {i} says students must submit the weekly report number {i} on time." for i in range(40)]
    blocks = [make_block(" ".join(sentences), heading_path=["Policy", "Reports"], page=3)]
    chunks = make_splitter(token_counter).split(blocks, "policy.pdf")
    assert len(chunks) > 3
    for chunk in chunks:
        assert chunk.page_content.startswith("[File: policy.pdf | Section: Policy > Reports | Page 3]")
        body = chunk.page_content.split("\n", 1)[1]
        assert token_counter.count(body) <= 120
    # consecutive chunks share some text (overlap)
    first_tail = chunks[0].page_content.split()[-3:]
    assert " ".join(first_tail) in chunks[1].page_content


def test_sections_and_pages_are_not_mixed(token_counter):
    blocks = [
        make_block("Meal allowance is 50 dollars.", heading_path=["Allowance", "Meal"], page=1),
        make_block("Korean allowance needs a pass.", heading_path=["Allowance", "Korean"], page=1),
        make_block("Korean test is in May.", heading_path=["Allowance", "Korean"], page=2),
    ]
    chunks = make_splitter(token_counter).split(blocks, "a.pdf")
    assert len(chunks) == 3
    assert [c.metadata["section_path"] for c in chunks] == ["Allowance > Meal", "Allowance > Korean", "Allowance > Korean"]
    assert [c.metadata["chunk_index"] for c in chunks] == [0, 1, 2]


def test_table_row_groups_repeat_header_and_never_split_rows(token_counter):
    rows = [[f"Student {i}", "Present" if i % 2 else "Absent", str(i)] for i in range(25)]
    block = make_table_block(["Name", "Status", "Day"], rows, heading_path=["Attendance"], sheet="Attendance")
    chunks = make_splitter(token_counter).split([block], "att.xlsx")
    assert sum(c.metadata["row_end"] - c.metadata["row_start"] + 1 for c in chunks) == 25
    assert chunks[0].metadata["row_start"] == 1 and chunks[-1].metadata["row_end"] == 25
    for chunk in chunks:
        assert "| Name | Status | Day |" in chunk.page_content
        assert chunk.page_content.startswith("[File: att.xlsx | Sheet: Attendance | Rows ")
        assert chunk.metadata["row_end"] - chunk.metadata["row_start"] + 1 <= 10
        assert "table_rows" not in chunk.metadata


def test_small_slides_are_merged_large_ones_kept_apart(token_counter):
    slides = [
        make_block("# Intro\n- Welcome", "slide", heading_path=["Intro"], page=1),
        make_block("# Agenda\n- Spring\n- JPA", "slide", heading_path=["Agenda"], page=2),
        make_block("# Details\n" + "- detail point about controllers and services\n" * 12, "slide", heading_path=["Details"], page=3),
    ]
    chunks = make_splitter(token_counter).split(slides, "deck.pptx")
    assert chunks[0].metadata["slides"] == [1, 2]
    assert "Slides 1-2" in chunks[0].page_content
    assert all(c.metadata["page"] == 3 for c in chunks[1:])


def test_context_header_variants():
    assert context_header("a.pdf", {"heading_path": ["A", "B"], "page": 2}) == "[File: a.pdf | Section: A > B | Page 2]"
    assert context_header("d.pptx", {"block": "slide", "heading_path": ["Intro"], "page": 4}) == (
        "[File: d.pptx | Slide 4 | Title: Intro]"
    )
    assert context_header("s.xlsx", {"sheet": "S", "heading_path": ["S"], "row_start": 1, "row_end": 1}) == (
        "[File: s.xlsx | Sheet: S | Row 1]"
    )
