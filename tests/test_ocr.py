"""OCR engine selection: PaddleOCR first, Surya when PaddleOCR's result looks wrong (calibrated on the
OCR_Test benchmark pages), plus the per-page recorder used for the ingestion trace."""

import pytest
from PIL import Image

from app.ingestion.errors import IngestionError
from app.ingestion.ocr import AutoOcrEngine, OcrEngine, OcrRecorder, OcrResult, SuryaOcrEngine, english_ratio

ENGLISH = "Kingdom of Cambodia\nThe holiday for all students and staff will be on Monday. Please return to class on time."


class StubEngine(OcrEngine):
    """Returns a fixed result and counts calls."""

    def __init__(self, name: str, result: OcrResult | Exception):
        """`result` is returned (or raised) on every call."""
        self.name = name
        self.result = result
        self.calls = 0

    def recognize(self, image: Image.Image) -> OcrResult:
        """Return or raise the configured result."""
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def auto(primary_result, fallback_result=None):
    """AutoOcrEngine over two stub engines."""
    primary = StubEngine("paddle", primary_result)
    fallback = StubEngine("surya", fallback_result or OcrResult("អត្ថបទខ្មែរ Khmer text", 0.9, engine="surya", seconds=3.0))
    return AutoOcrEngine(primary, fallback), primary, fallback


IMAGE = Image.new("RGB", (10, 10), "white")


def test_clean_english_page_stays_on_paddle():
    engine, _, fallback = auto(OcrResult(ENGLISH, 0.997, engine="paddle", seconds=1.0, line_confidences=[0.99, 0.998]))
    result = engine.recognize(IMAGE)
    assert result.engine == "paddle" and fallback.calls == 0 and result.fallback_reason is None


def test_khmer_page_low_confidence_falls_back_to_surya():
    # p01_Khmer: PaddleOCR mean confidence 0.47 (Khmer read as Latin noise)
    engine, _, fallback = auto(OcrResult("SamCsagnta amnngngn nagisimo", 0.4739, engine="paddle", seconds=2.0))
    result = engine.recognize(IMAGE)
    assert result.engine == "surya" and fallback.calls == 1
    assert "low PaddleOCR confidence" in result.fallback_reason
    assert [a["engine"] for a in result.attempts] == ["paddle", "surya"] and result.seconds == 5.0


def test_mixed_page_with_unreadable_lines_falls_back():
    # p02_KhmerEng: mean 0.886 looks fine, but the Khmer lines score far lower
    lines = [0.99] * 10 + [0.42, 0.51, 0.38, 0.6]
    engine, _, _ = auto(OcrResult(ENGLISH, 0.886, engine="paddle", line_confidences=lines))
    assert "lines unreadable" in engine.recognize(IMAGE).fallback_reason


def test_confident_table_with_few_english_words_stays_on_paddle():
    # p03_HRDTable: English-word ratio 0.038 but confidence 0.997
    table = "No Activities Time Duration Remark Checking attendance 7:00AM 10 mn SR vs PP 60 mn Break Game Lunch Prize Ceremony"
    engine, _, fallback = auto(OcrResult(table, 0.997, engine="paddle", line_confidences=[0.99] * 12))
    assert engine.recognize(IMAGE).engine == "paddle" and fallback.calls == 0


def test_fallback_unavailable_keeps_paddle_result_with_reason():
    engine, _, _ = auto(OcrResult("noise", 0.3, engine="paddle"), IngestionError("Surya is not installed"))
    result = engine.recognize(IMAGE)
    assert result.engine == "paddle" and "fallback unavailable" in result.fallback_reason


def test_english_ratio():
    assert english_ratio(ENGLISH) > 0.15
    assert english_ratio("SamCsagnta amnngngn nagisimo smethognag ngsn ssnnannisligansn") == 0.0


def test_recorder_summarises_pages():
    recorder = OcrRecorder(StubEngine("auto", OcrResult("ខ្មែរ", 0.9, engine="surya", fallback_reason="low confidence")))
    recorder.recognize(IMAGE)
    recorder.recognize(IMAGE)
    summary = recorder.summary()
    assert summary["pages"] == 2 and summary["engines"] == {"surya": 2} and summary["khmer_pages"] == 2
    assert summary["fallbacks"] == ["low confidence", "low confidence"]
    assert OcrRecorder(StubEngine("x", OcrResult("a"))).summary() is None


def test_surya_missing_raises_clear_error(tmp_path):
    engine = SuryaOcrEngine(python=str(tmp_path / "no-python.exe"))
    assert not engine.available
    with pytest.raises(IngestionError, match="Surya is not installed"):
        engine.recognize(IMAGE)


def test_auto_release_frees_both_engines():
    class Releasing(StubEngine):
        """Stub that counts release() calls."""

        released = 0

        def release(self) -> None:
            """Count the call."""
            self.released += 1

    primary, fallback = Releasing("paddle", OcrResult("a")), Releasing("surya", OcrResult("b"))
    AutoOcrEngine(primary, fallback).release()
    assert primary.released == 1 and fallback.released == 1
