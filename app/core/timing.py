"""Small helpers for the pipeline traces (per-stage timings stored on documents and chat answers)."""

import time


def ms_since(started: float) -> int:
    """Milliseconds elapsed since `started` (a time.perf_counter() value)."""
    return round((time.perf_counter() - started) * 1000)


def merge_trace(current: dict | None, update: dict | None) -> dict:
    """LangGraph reducer: each node adds its own keys to the shared trace dict."""
    return {**(current or {}), **(update or {})}
