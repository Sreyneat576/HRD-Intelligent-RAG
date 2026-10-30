"""OCR engines for scanned PDF pages and images.

* PaddleOCR - fast, very accurate on English, but has no Khmer model (Khmer comes out as Latin noise).
* Surya (surya-ocr 0.17, "Surya 1") - reads Khmer, English and mixed pages; slower (CPU). It runs in a
  separate worker process with its own virtualenv, see app/ingestion/surya_worker.py.
* auto (default) - PaddleOCR first; when its result looks wrong (low confidence, or text that is not
  English, e.g. Khmer read as Latin noise) the page is OCR'd again with Surya.
* Tesseract - light fallback; PP-StructureV3 can be switched on for layout-aware Markdown output.

The engine choice follows the team's OCR benchmark (github.com/TyDine22/OCR_Test): Surya had the best
Khmer and table accuracy, PaddleOCR the best English speed.
"""

import json
import logging
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image

from app.ingestion.errors import IngestionError

log = logging.getLogger(__name__)

_WORDS = re.compile(r"[A-Za-z]{2,}")
_KHMER = re.compile(r"[ក-៿]")
_STOPWORDS = frozenset(
    "the a an and or of to in on for with is are was were be by at from as this that it its your our we you "
    "will has have not no all any can may must should if please date name student students class time day".split()
)


@dataclass
class OcrResult:
    text: str
    confidence: float | None = None
    is_markdown: bool = False
    engine: str = ""
    seconds: float = 0.0
    fallback_reason: str | None = None
    attempts: list[dict] = field(default_factory=list)
    line_confidences: list[float] = field(default_factory=list)


class OcrEngine(ABC):
    name: str

    @abstractmethod
    def recognize(self, image: Image.Image) -> OcrResult:
        """Extract text from one image."""

    def close(self) -> None:
        """Release resources (worker processes); called at shutdown."""

    def release(self) -> None:
        """Free memory held between jobs (no-op unless the engine runs a worker process)."""


class NoOcr(OcrEngine):
    name = "none"

    def recognize(self, image: Image.Image) -> OcrResult:
        """Always fails: OCR is switched off."""
        raise IngestionError("This file needs OCR, but OCR_ENGINE is set to 'none'")


class PaddleOcrEngine(OcrEngine):
    name = "paddle"

    def __init__(self, lang: str = "en", structure: bool = False, mkldnn: bool = False,
                 det_model: str | None = "PP-OCRv5_mobile_det", max_side: int = 1600):
        """Configure PaddleOCR; the models are loaded lazily on first use. `mkldnn` stays off by default:
        Paddle's oneDNN CPU path fails on Windows ("ConvertPirAttribute2RuntimeAttribute not support").
        The mobile detector and a `max_side` limit keep CPU time per scanned page reasonable."""
        self.lang = lang
        self.structure = structure
        self.mkldnn = mkldnn
        self.det_model = det_model
        self.max_side = max_side
        self._model = None
        self._lock = threading.Lock()  # Paddle pipelines are not thread-safe

    def _load(self):
        """Create the PaddleOCR (text) or PP-StructureV3 (layout to Markdown) pipeline."""
        common = {"lang": self.lang, "use_doc_orientation_classify": False, "use_doc_unwarping": False,
                  "enable_mkldnn": self.mkldnn}
        if self.det_model:
            common["text_detection_model_name"] = self.det_model
        if self.max_side:
            common.update(text_det_limit_type="max", text_det_limit_side_len=self.max_side)
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
        started = time.perf_counter()
        bgr = np.array(image.convert("RGB"))[:, :, ::-1].copy()
        with self._lock:
            if self._model is None:
                self._model = self._load()
            results = list(self._model.predict(bgr))
        seconds = round(time.perf_counter() - started, 3)
        if self.structure:
            markdown = "\n\n".join(res.markdown["markdown_texts"] for res in results)
            return OcrResult(markdown, None, is_markdown=True, engine="ppstructure", seconds=seconds)
        texts: list[str] = []
        scores: list[float] = []
        for res in results:
            texts.extend(res["rec_texts"])
            scores.extend(float(s) for s in res["rec_scores"])
        text = "\n".join(t for t in texts if t.strip())
        return OcrResult(text, round(mean(scores), 4) if scores else None, engine="paddle", seconds=seconds)


class SuryaOcrEngine(OcrEngine):
    """Surya OCR through a long-lived worker process running in `.venv-surya` (see surya_worker.py).
    The worker is started on first use and restarted if it dies."""

    name = "surya"

    def __init__(self, python: str, device: str = "cpu", startup_timeout: float = 1800, page_timeout: float = 900):
        """`python` is the Surya virtualenv's interpreter; the first start may download ~1.5 GB of models."""
        self.python = python
        self.device = device
        self.startup_timeout = startup_timeout
        self.page_timeout = page_timeout
        self._process: subprocess.Popen | None = None
        self._lines: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        self._next_id = 0

    @property
    def available(self) -> bool:
        """True when the Surya virtualenv exists (it is optional)."""
        return Path(self.python).exists()

    def _read_stdout(self, process: subprocess.Popen) -> None:
        """Background thread: forward the worker's protocol lines to the queue."""
        for line in process.stdout:
            if line.startswith("@@"):
                self._lines.put(line.strip())

    def _wait_for(self, kind: str, timeout: float) -> dict:
        """Wait for the next `@@<kind>` message from the worker, or fail on timeout / exit."""
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.close()
                raise IngestionError(f"Surya worker did not answer within {timeout:.0f}s", retryable=True)
            try:
                line = self._lines.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                if self._process is not None and self._process.poll() is not None:
                    raise IngestionError("Surya worker stopped unexpectedly (see data/logs/surya-worker.log)",
                                         retryable=True)
                continue
            if line.startswith(f"@@{kind} "):
                return json.loads(line.split(" ", 1)[1])

    def _ensure_started(self) -> None:
        """Start the worker process (and load its models) if it is not running."""
        if self._process is not None and self._process.poll() is None:
            return
        if not self.available:
            raise IngestionError(
                f"Surya is not installed ({self.python} not found). Create .venv-surya - see README, 'Surya OCR'."
            )
        log_dir = Path("data/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        self._lines = queue.Queue()
        self._process = subprocess.Popen(
            # Absolute path: Windows CreateProcess does not resolve relative interpreter paths.
            [str(Path(self.python).resolve()), "-u", str(Path(__file__).with_name("surya_worker.py"))],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=(log_dir / "surya-worker.log").open("a", encoding="utf-8"),
            text=True,
            encoding="utf-8",
            # Small batches keep Surya's CPU memory peak low (one page at a time anyway).
            env={"RECOGNITION_BATCH_SIZE": "8", "DETECTOR_BATCH_SIZE": "1", **os.environ,
                 "TORCH_DEVICE": self.device, "PYTHONIOENCODING": "utf-8"},
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0,
        )
        threading.Thread(target=self._read_stdout, args=(self._process,), daemon=True).start()
        ready = self._wait_for("READY", self.startup_timeout)
        log.info("Surya worker ready on %s (models loaded in %ss)", ready.get("device"), ready.get("load_seconds"))

    def recognize(self, image: Image.Image) -> OcrResult:
        """Send the image to the worker and return its lines and mean confidence."""
        started = time.perf_counter()
        with self._lock:
            self._ensure_started()
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                image.convert("RGB").save(tmp.name)
            try:
                self._next_id += 1
                self._process.stdin.write(json.dumps({"id": self._next_id, "image": tmp.name}) + "\n")
                self._process.stdin.flush()
                reply = self._wait_for("RESULT", self.page_timeout)
            finally:
                Path(tmp.name).unlink(missing_ok=True)
        if reply.get("error"):
            raise IngestionError(f"Surya OCR failed: {reply['error']}", retryable=True)
        lines = reply.get("lines", [])
        confidences = [line["confidence"] for line in lines if line.get("confidence")]
        return OcrResult(
            "\n".join(line["text"] for line in lines),
            round(mean(confidences), 4) if confidences else None,
            engine="surya",
            seconds=round(time.perf_counter() - started, 3),
        )

    def close(self) -> None:
        """Stop the worker process."""
        if self._process is not None and self._process.poll() is None:
            self._process.kill()
        self._process = None

    def release(self) -> None:
        """Stop the worker between jobs: Surya holds ~5 GB of RAM on the CPU (it restarts on the next scan)."""
        with self._lock:
            if self._process is not None:
                log.info("Stopping the Surya worker to free memory")
            self.close()


class TesseractEngine(OcrEngine):
    name = "tesseract"
    _LANGS = {"en": "eng", "ko": "kor", "korean": "kor", "km": "khm"}

    def __init__(self, lang: str = "en"):
        """Map short language codes (en, ko, km) to Tesseract language names."""
        self.lang = self._LANGS.get(lang, lang)

    def recognize(self, image: Image.Image) -> OcrResult:
        """Run Tesseract; rebuild lines from word boxes and average the word confidences."""
        started = time.perf_counter()
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
        return OcrResult(text, round(mean(confidences), 4) if confidences else None, engine="tesseract",
                         seconds=round(time.perf_counter() - started, 3))


def english_ratio(text: str) -> float:
    """Share of words that are common English words. Real English text scores ~0.15-0.4;
    Khmer that PaddleOCR misread as Latin letters scores close to 0."""
    words = _WORDS.findall(text)
    return sum(w.lower() in _STOPWORDS for w in words) / len(words) if words else 0.0


class AutoOcrEngine(OcrEngine):
    """PaddleOCR first, Surya when PaddleOCR's result looks wrong."""

    name = "auto"

    def __init__(self, primary: OcrEngine, fallback: OcrEngine, min_confidence: float = 0.85,
                 min_english_ratio: float = 0.04, min_words: int = 12, low_line_confidence: float = 0.70,
                 max_low_line_share: float = 0.15):
        """Fallback rules (calibrated on the OCR_Test benchmark pages):
        * no text, or mean confidence below `min_confidence` (a Khmer page: 0.47);
        * at least 3 lines and `max_low_line_share` of lines below `low_line_confidence`
          (a mixed Khmer/English page averages 0.89, but its Khmer lines score far lower);
        * fewer than `min_english_ratio` common English words while confidence is not high
          (an English table has few such words but 0.997 confidence, so it stays on PaddleOCR)."""
        self.primary = primary
        self.fallback = fallback
        self.min_confidence = min_confidence
        self.min_english_ratio = min_english_ratio
        self.min_words = min_words
        self.low_line_confidence = low_line_confidence
        self.max_low_line_share = max_low_line_share

    def fallback_reason(self, result: OcrResult) -> str | None:
        """Why the primary result should not be trusted, or None when it looks fine."""
        if not result.text.strip():
            return "PaddleOCR found no text"
        if result.confidence is not None and result.confidence < self.min_confidence:
            return f"low PaddleOCR confidence ({result.confidence:.2f})"
        lines = result.line_confidences
        low = [c for c in lines if c < self.low_line_confidence]
        if len(low) >= 3 and len(low) / len(lines) >= self.max_low_line_share:
            return f"{len(low)} of {len(lines)} lines unreadable for PaddleOCR (possibly Khmer)"
        high_confidence = result.confidence is not None and result.confidence >= 0.95
        if (not high_confidence and len(_WORDS.findall(result.text)) >= self.min_words
                and english_ratio(result.text) < self.min_english_ratio):
            return "text does not look like English (possibly Khmer)"
        return None

    def recognize(self, image: Image.Image) -> OcrResult:
        """Run the primary engine; rerun with the fallback engine when needed."""
        first = self.primary.recognize(image)
        attempts = [{"engine": first.engine, "seconds": first.seconds, "confidence": first.confidence}]
        reason = self.fallback_reason(first)
        if reason is None:
            first.attempts = attempts
            return first
        try:
            second = self.fallback.recognize(image)
        except IngestionError as exc:
            log.warning("OCR fallback to %s failed (%s); keeping %s result", self.fallback.name, exc, first.engine)
            first.fallback_reason = f"{reason}; fallback unavailable: {exc}"
            first.attempts = attempts
            return first
        attempts.append({"engine": second.engine, "seconds": second.seconds, "confidence": second.confidence})
        second.fallback_reason = reason
        second.attempts = attempts
        second.seconds = round(first.seconds + second.seconds, 3)
        return second

    def close(self) -> None:
        """Close both engines."""
        self.primary.close()
        self.fallback.close()

    def release(self) -> None:
        """Release both engines' between-job memory."""
        self.primary.release()
        self.fallback.release()


class OcrRecorder(OcrEngine):
    """Wraps an engine for one ingestion job and records what happened on every page (for the trace)."""

    def __init__(self, inner: OcrEngine):
        """Forward to `inner`, keep one record per call."""
        self.inner = inner
        self.name = inner.name
        self.pages: list[dict] = []

    def recognize(self, image: Image.Image) -> OcrResult:
        """Run the wrapped engine and record engine, time, confidence and fallback reason."""
        started = time.perf_counter()
        result = self.inner.recognize(image)
        self.pages.append({
            "engine": result.engine or self.inner.name,
            "ms": round((time.perf_counter() - started) * 1000),
            "confidence": result.confidence,
            "fallback_reason": result.fallback_reason,
            "attempts": result.attempts,
            "khmer": bool(_KHMER.search(result.text)),
        })
        return result

    def summary(self) -> dict | None:
        """Totals for the document trace, or None when no page needed OCR."""
        if not self.pages:
            return None
        engines: dict[str, int] = {}
        for page in self.pages:
            engines[page["engine"]] = engines.get(page["engine"], 0) + 1
        confidences = [p["confidence"] for p in self.pages if p["confidence"] is not None]
        return {
            "mode": self.inner.name,
            "pages": len(self.pages),
            "ms": sum(p["ms"] for p in self.pages),
            "engines": engines,
            "mean_confidence": round(mean(confidences), 4) if confidences else None,
            "fallbacks": [p["fallback_reason"] for p in self.pages if p["fallback_reason"]],
            "khmer_pages": sum(p["khmer"] for p in self.pages),
            "per_page": self.pages,
        }


def build_ocr_engine(
    engine: str,
    lang: str = "en",
    structure: bool = False,
    paddle_mkldnn: bool = False,
    paddle_det_model: str | None = "PP-OCRv5_mobile_det",
    paddle_max_side: int = 1600,
    surya_python: str = "",
    surya_device: str = "cpu",
    fallback_min_confidence: float = 0.85,
    fallback_min_english_ratio: float = 0.04,
) -> OcrEngine:
    """OCR engine selected by the OCR_ENGINE setting (auto | paddle | surya | tesseract | none)."""
    paddle = lambda: PaddleOcrEngine(  # noqa: E731
        lang=lang, structure=structure, mkldnn=paddle_mkldnn, det_model=paddle_det_model, max_side=paddle_max_side
    )
    surya = lambda: SuryaOcrEngine(python=surya_python, device=surya_device)  # noqa: E731
    if engine == "auto":
        return AutoOcrEngine(paddle(), surya(), fallback_min_confidence, fallback_min_english_ratio)
    if engine == "paddle":
        return paddle()
    if engine == "surya":
        return surya()
    if engine == "tesseract":
        return TesseractEngine(lang=lang)
    return NoOcr()
