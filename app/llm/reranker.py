"""Cross-encoder reranker (BAAI/bge-reranker-v2-m3) served by TEI or vLLM, as a LangChain
document compressor."""

from collections.abc import Sequence
from typing import Literal

import httpx
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor


class RemoteReranker(BaseDocumentCompressor):
    base_url: str
    api: Literal["tei", "vllm"] = "tei"
    model: str = "BAAI/bge-reranker-v2-m3"
    top_n: int = 5
    min_score: float = 0.0
    timeout: float = 60.0

    def _request(self, query: str, texts: list[str]) -> tuple[str, dict]:
        """Build the URL and JSON body for the TEI or vLLM rerank API."""
        base = self.base_url.rstrip("/")
        if self.api == "tei":
            return f"{base}/rerank", {"query": query, "texts": texts, "truncate": True, "raw_scores": False}
        base = base.removesuffix("/v1")
        return f"{base}/v1/rerank", {"model": self.model, "query": query, "documents": texts}

    def _parse(self, payload, n: int) -> list[float]:
        """Turn the TEI / vLLM response into one score per input text, in input order."""
        scores = [0.0] * n
        items = payload if isinstance(payload, list) else payload["results"]
        for item in items:
            scores[item["index"]] = float(item.get("score", item.get("relevance_score", 0.0)))
        return scores

    async def ascore(self, query: str, texts: list[str]) -> list[float]:
        """Relevance score of each text for the query (async HTTP call)."""
        if not texts:
            return []
        url, body = self._request(query, texts)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=body)
            response.raise_for_status()
        return self._parse(response.json(), len(texts))

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Relevance score of each text for the query (blocking HTTP call)."""
        if not texts:
            return []
        url, body = self._request(query, texts)
        response = httpx.post(url, json=body, timeout=self.timeout)
        response.raise_for_status()
        return self._parse(response.json(), len(texts))

    def _select(self, documents: Sequence[Document], scores: list[float]) -> list[Document]:
        """Sort by score, drop those under `min_score`, keep `top_n`, store the score in metadata."""
        ranked = sorted(zip(documents, scores), key=lambda pair: pair[1], reverse=True)
        selected = []
        for doc, score in ranked:
            if score < self.min_score:
                break
            selected.append(Document(page_content=doc.page_content, metadata={**doc.metadata, "rerank_score": score}))
            if len(selected) == self.top_n:
                break
        return selected

    def compress_documents(
        self, documents: Sequence[Document], query: str, callbacks: Callbacks | None = None
    ) -> Sequence[Document]:
        """LangChain compressor API (sync): rerank and keep the best documents."""
        return self._select(documents, self.score(query, [d.page_content for d in documents]))

    async def acompress_documents(
        self, documents: Sequence[Document], query: str, callbacks: Callbacks | None = None
    ) -> Sequence[Document]:
        """LangChain compressor API (async): rerank and keep the best documents."""
        return self._select(documents, await self.ascore(query, [d.page_content for d in documents]))
