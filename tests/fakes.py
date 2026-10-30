"""Test doubles for the model services, so tests run without Ollama / TEI / OCR."""

import itertools
import math
import re
import zlib

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from PIL import Image

from app.ingestion.ocr import OcrEngine, OcrResult


class HashingEmbeddings(Embeddings):
    """Deterministic bag-of-words vectors: texts sharing words get similar vectors."""

    def __init__(self, dim: int = 1024):
        """Vector size, same as the real model (1024)."""
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        """Hash each word into one of `dim` buckets and L2-normalise."""
        vector = [0.0] * self.dim
        for word in re.findall(r"\w+", text.lower()):
            vector[zlib.crc32(word.encode()) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed many texts."""
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed one query."""
        return self._embed(text)


def fake_chat_model(answer: str = "The answer is in the first source [1].") -> GenericFakeChatModel:
    """Chat model that always replies with `answer` (streamed word by word)."""
    return GenericFakeChatModel(messages=itertools.repeat(AIMessage(answer)))


class FakeOcr(OcrEngine):
    """OCR stand-in that returns fixed text."""

    name = "fake"

    def __init__(self, text: str = "Scanned attendance form\nStudent: Dara\nStatus: Present"):
        """Text the fake OCR will 'read' from any image."""
        self.text = text

    def recognize(self, image: Image.Image) -> OcrResult:
        """Return the fixed text with a fixed confidence."""
        return OcrResult(self.text, confidence=0.93)


class FakeRetriever:
    """Duck-typed stand-in for HybridPgRetriever: returns fixed documents filtered by allowed ids."""

    def __init__(self, documents: list[Document]):
        """Documents this fake can return."""
        self.documents = documents
        self.allowed: list[str] = []
        self.queries: list[str] = []

    def for_documents(self, document_ids: list[str], mode=None) -> "FakeRetriever":
        """Copy limited to the allowed ids (shares the query log)."""
        clone = FakeRetriever(self.documents)
        clone.allowed = list(document_ids)
        clone.queries = self.queries
        return clone

    async def ainvoke(self, query: str) -> list[Document]:
        """Record the query and return the allowed documents."""
        self.queries.append(query)
        return [d for d in self.documents if d.metadata["document_id"] in self.allowed]

    async def asearch(self, query: str) -> tuple[list[Document], dict]:
        """Like HybridPgRetriever.asearch: documents plus a timing trace."""
        documents = await self.ainvoke(query)
        return documents, {"embed_ms": 1, "dense_ms": 2, "lexical_ms": 1, "fuse_ms": 0,
                           "dense_hits": len(documents), "lexical_hits": 0, "candidates": len(documents)}
