"""Surya OCR worker process.

Runs in its own virtualenv (`.venv-surya`: CPU PyTorch + surya-ocr 0.17), started and managed by
`SuryaOcrEngine` in app/ingestion/ocr.py. Keeping PyTorch out of the service's environment avoids the
PaddlePaddle / PyTorch package and OpenMP conflicts. This file must only import the standard
library, PIL and surya - never the `app` package.

Protocol (one JSON object per line, ASCII-escaped so Khmer survives any console encoding):
    parent -> worker (stdin):   {"id": 1, "image": "C:/.../page.png"}
    worker -> parent (stdout):  @@READY {"device": "cpu", "load_seconds": 12.3}
                                @@RESULT {"id": 1, "lines": [{"text": "...", "confidence": 0.98}], "seconds": 4.2}
                                @@RESULT {"id": 1, "error": "..."}
Any other output (library logs, progress bars) is ignored by the parent.
"""

import json
import os
import re
import sys
import time

os.environ.setdefault("TORCH_DEVICE", "cpu")
_TAG = re.compile(r"<[^>]+>")


def emit(kind: str, payload: dict) -> None:
    """Write one protocol message to stdout and flush it immediately."""
    sys.stdout.write(f"@@{kind} {json.dumps(payload)}\n")
    sys.stdout.flush()


def main() -> None:
    """Load the Surya models once, then answer OCR requests until stdin closes."""
    started = time.perf_counter()
    from PIL import Image
    from surya.detection import DetectionPredictor
    from surya.foundation import FoundationPredictor
    from surya.recognition import RecognitionPredictor

    detector = DetectionPredictor()
    recognizer = RecognitionPredictor(FoundationPredictor())
    emit("READY", {"device": os.environ["TORCH_DEVICE"], "load_seconds": round(time.perf_counter() - started, 2)})

    for raw in sys.stdin:
        if not raw.strip():
            continue
        request = json.loads(raw)
        started = time.perf_counter()
        try:
            image = Image.open(request["image"]).convert("RGB")
            page = recognizer([image], det_predictor=detector, sort_lines=True, math_mode=False)[0]
            lines = [
                {"text": " ".join(_TAG.sub(" ", line.text).split()), "confidence": float(line.confidence or 0.0)}
                for line in page.text_lines
            ]
            emit("RESULT", {"id": request["id"], "lines": [l for l in lines if l["text"]],
                            "seconds": round(time.perf_counter() - started, 3)})
        except Exception as exc:  # report the failure for this page and keep serving
            emit("RESULT", {"id": request.get("id"), "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
