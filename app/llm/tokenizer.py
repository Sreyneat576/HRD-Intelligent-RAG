"""Token counting with the embedding model's own tokenizer (bge-m3), so chunk sizes match what
the embedder sees. Uses the lightweight `tokenizers` package (no torch)."""

import logging
from functools import lru_cache

from tokenizers import Tokenizer

log = logging.getLogger(__name__)


class TokenCounter:
    def __init__(self, tokenizer_name: str):
        """Load the tokenizer from the Hugging Face hub (falls back to a 4-chars-per-token estimate offline)."""
        self._tokenizer: Tokenizer | None
        try:
            self._tokenizer = Tokenizer.from_pretrained(tokenizer_name)
            self._tokenizer.no_truncation()
            self._tokenizer.no_padding()
        except Exception as exc:  # offline or unknown model: fall back to an estimate
            log.warning("Could not load tokenizer %s (%s); estimating 4 chars per token", tokenizer_name, exc)
            self._tokenizer = None

    def count(self, text: str) -> int:
        """Number of tokens in `text` (without special tokens)."""
        if not text:
            return 0
        if self._tokenizer is None:
            return max(1, len(text) // 4)
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def __call__(self, text: str) -> int:
        """Make the counter usable directly as a `length_function`."""
        return self.count(text)

    def split_by_tokens(self, text: str, max_tokens: int) -> list[str]:
        """Hard split for text with no usable separators (e.g. a long OCR line)."""
        if self._tokenizer is None:
            step = max_tokens * 4
            return [text[i : i + step] for i in range(0, len(text), step)]
        offsets = self._tokenizer.encode(text, add_special_tokens=False).offsets
        pieces = []
        for start in range(0, len(offsets), max_tokens):
            window = offsets[start : start + max_tokens]
            pieces.append(text[window[0][0] : window[-1][1]])
        return [p for p in pieces if p.strip()]


@lru_cache
def get_token_counter(tokenizer_name: str) -> TokenCounter:
    """One shared TokenCounter per tokenizer name."""
    return TokenCounter(tokenizer_name)
