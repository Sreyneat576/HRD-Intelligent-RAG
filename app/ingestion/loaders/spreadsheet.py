"""Excel (.xlsx) and CSV loaders (pandas + openpyxl).

Each sheet yields a *summary* text block (columns, row count, value ranges) followed by one table
block holding all rows; the splitter cuts the table into row groups with the header repeated."""

import io
import math
from collections.abc import Iterator
from datetime import date, datetime

import pandas as pd
from charset_normalizer import from_bytes
from langchain_core.documents import Document

from app.ingestion.errors import IngestionError
from app.ingestion.loaders.base import FileLoader, clean_cell, make_block, make_table_block

_MAX_CATEGORIES = 12


def _format(value) -> str:
    """Cell value as text: dates in ISO format, whole floats without '.0', empty for NaN."""
    if value is None or (isinstance(value, float) and math.isnan(value)) or value is pd.NaT:
        return ""
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.date().isoformat() if (value.hour, value.minute, value.second) == (0, 0, 0) else value.isoformat(sep=" ")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return clean_cell(value)


def decode_text(data: bytes) -> str:
    """Decode bytes with the detected encoding (UTF-8, Windows-1252, ...)."""
    best = from_bytes(data).best()
    return str(best) if best is not None else data.decode("utf-8", errors="replace")


def _unique_header(raw: list[str]) -> list[str]:
    """Header names with blanks filled ('Column 3') and duplicates numbered."""
    seen: dict[str, int] = {}
    header = []
    for i, name in enumerate(raw):
        name = name or f"Column {i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name} ({seen[name]})"
        else:
            seen[name] = 1
        header.append(name)
    return header


def sheet_summary(sheet: str, header: list[str], rows: list[list[str]]) -> str:
    """Plain-language summary of a sheet: size, columns, numeric ranges and category counts."""
    lines = [f"Sheet '{sheet}': {len(rows)} rows x {len(header)} columns.", "Columns: " + ", ".join(header) + "."]
    for j, name in enumerate(header):
        values = [r[j] for r in rows if j < len(r) and r[j] != ""]
        if not values:
            continue
        numbers = []
        for v in values:
            try:
                numbers.append(float(v.replace(",", "")))
            except ValueError:
                pass
        if len(numbers) >= 0.8 * len(values):
            lines.append(
                f"- {name}: numeric, min {min(numbers):g}, max {max(numbers):g}, average {sum(numbers) / len(numbers):.2f}"
            )
            continue
        distinct = pd.Series(values).value_counts()
        if len(distinct) <= _MAX_CATEGORIES:
            counts = ", ".join(f"{k} ({v})" for k, v in distinct.items())
            lines.append(f"- {name}: {counts}")
        else:
            lines.append(f"- {name}: {len(distinct)} distinct values, e.g. {', '.join(distinct.index[:3])}")
    return "\n".join(lines)


class SpreadsheetLoader(FileLoader):
    def _read_sheets(self) -> dict[str, pd.DataFrame]:
        """All sheets as raw DataFrames (CSV counts as one sheet)."""
        try:
            if self.path.suffix.lower() == ".csv":
                text = decode_text(self.path.read_bytes())
                frame = pd.read_csv(
                    io.StringIO(text), header=None, dtype=str, sep=None, engine="python", skip_blank_lines=True
                )
                return {self.path.stem: frame}
            return pd.read_excel(self.path, sheet_name=None, header=None, dtype=object, engine="openpyxl")
        except Exception as exc:
            raise IngestionError(f"Cannot read spreadsheet: {exc}") from exc

    def lazy_load(self) -> Iterator[Document]:
        """Per sheet: a summary text block, then one table block with all rows."""
        sheets = self._read_sheets()
        self.page_count = len(sheets)
        for sheet, frame in sheets.items():
            frame = frame.dropna(how="all").dropna(axis=1, how="all")
            if frame.empty:
                continue
            # Header = first row that fills at least half of the columns (skips title rows above it).
            min_filled = max(1, math.ceil(frame.shape[1] / 2))
            header_pos = next((i for i in range(len(frame)) if frame.iloc[i].notna().sum() >= min_filled), 0)
            header = _unique_header([_format(v) for v in frame.iloc[header_pos]])
            rows = [[_format(v) for v in row] for row in frame.iloc[header_pos + 1 :].itertuples(index=False)]
            rows = [r for r in rows if any(r)]
            sheet_name = str(sheet)
            yield make_block(
                sheet_summary(sheet_name, header, rows),
                heading_path=[sheet_name],
                sheet=sheet_name,
                kind="sheet_summary",
            )
            if rows:
                yield make_table_block(header, rows, heading_path=[sheet_name], sheet=sheet_name)
