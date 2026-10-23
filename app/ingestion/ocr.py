"""OCR engines for scanned PDF pages and images. PaddleOCR is the default (spec §5.2), Tesseract is
the light fallback, and PP-StructureV3 can be switched on for layout-aware Markdown output."""

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from statistics import mean

import numpy as np
from PIL import Image

from app.ingestion.errors import IngestionError

log = logging.getLogger(__name__)


@dataclass
class OcrResult:
    text: str
    confidence: float | None = None
    is_markdown: bool = False


class OcrEngine(ABC):
    name: str

    @abstractmethod
    def recognize(self, image: Image.Image) -> OcrResult:
        """Extract text from one image."""


class NoOcr(OcrEngine):
    name = "none"

    def recognize(self, image: Image.Image) -> OcrResult:
        """Always fails: OCR is switched off."""
        raise IngestionError("This file needs OCR, but OCR_ENGINE is set to 'none'")


class PaddleOcrEngine(OcrEngine):
    name = "paddle"

    def __init__(self, lang: str = "en", structure: bool = False, mkldnn: bool = False):
        """Configure PaddleOCR; the models are loaded lazily on first use. `mkldnn` stays off by default:
        Paddle's oneDNN CPU path fails on Windows ("ConvertPirAttribute2RuntimeAttribute not support")."""
        self.lang = lang
        self.structure = structure
        self.mkldnn = mkldnn
        self._model = None
        self._lock = threading.Lock()  # Paddle pipelines are not thread-safe

    def _load(self):
        """Create the PaddleOCR (text) or PP-StructureV3 (layout to Markdown) pipeline."""
        common = {"lang": self.lang, "use_doc_orientation_classify": False, "use_doc_unwarping": False,
                  "enable_mkldnn": self.mkldnn}
        try:
            if self.structure:
                from paddleocr import PPStructureV3

                return PPStructureV3(**common)
            from paddleocr import PaddleOCR

            return PaddleOCR(**common, use_textline_orientation=False)
        except ImportError as exc:
            raise IngestionError(
                "PaddleOCR is not installed. Run: pip install -e .[ocr-paddle] (or set OCR_ENGINE=tesseract)"
            ) from exc

    def recognize(self, image: Image.Image) -> OcrResult:
        """Run PaddleOCR on the image; returns the text lines and their mean confidence."""
        bgr = np.array(image.convert("RGB"))[:, :, ::-1].copy()
        with self._lock:
            if self._model is None:
                self._model = self._load()
            results = list(self._model.predict(bgr))
        if self.structure:
            markdown = "\n\n".join(res.markdown["markdown_texts"] for res in results)
            return OcrResult(markdown, None, is_markdown=True)
        texts: list[str] = []
        scores: list[float] = []
        for res in results:
            texts.extend(res["rec_texts"])
            scores.extend(float(s) for s in res["rec_scores"])
        return OcrResult("\n".join(t for t in texts if t.strip()), round(mean(scores), 4) if scores else None)


class TesseractEngine(OcrEngine):
    name = "tesseract"
    _LANGS = {"en": "eng", "ko": "kor", "korean": "kor", "km": "khm"}

    def __init__(self, lang: str = "en"):
        """Map short language codes (en, ko, km) to Tesseract language names."""
        self.lang = self._LANGS.get(lang, lang)

    def recognize(self, image: Image.Image) -> OcrResult:
        """Run Tesseract; rebuild lines from word boxes and average the word confidences."""
        try:
            import pytesseract
        except ImportError as exc:
            raise IngestionError("pytesseract is not installed. Run: pip install -e .[ocr-tesseract]") from exc
        data = pytesseract.image_to_data(image, lang=self.lang, output_type=pytesseract.Output.DICT)
        lines: dict[tuple[int, int, int], list[str]] = {}
        confidences: list[float] = []
        for i, word in enumerate(data["text"]):
            if not word.strip():
                continue
            lines.setdefault((data["block_num"][i], data["par_num"][i], data["line_num"][i]), []).append(word)
            if float(data["conf"][i]) >= 0:
                confidences.append(float(data["conf"][i]) / 100)
        text = "\n".join(" ".join(words) for _, words in sorted(lines.items()))
        return OcrResult(text, round(mean(confidences), 4) if confidences else None)


def build_ocr_engine(engine: str, lang: str = "en", structure: bool = False, paddle_mkldnn: bool = False) -> OcrEngine:
    """OCR engine selected by the OCR_ENGINE setting."""
    if engine == "paddle":
        return PaddleOcrEngine(lang=lang, structure=structure, mkldnn=paddle_mkldnn)
    if engine == "tesseract":
        return TesseractEngine(lang=lang)
    return NoOcr()
