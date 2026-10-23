"""Retrieval quality check on your own documents.

Compares three retrieval set-ups on a question file and reports hit@k and MRR:
  dense            pgvector only
  hybrid           pgvector + full-text, fused with RRF
  hybrid+rerank    hybrid, then bge-reranker-v2-m3

Question file (JSON lines), one question per line:
  {"question": "Which OCR engines are used?", "filename": "HRD Intelligence - Project Detail.pdf", "answer_contains": "PaddleOCR"}
`filename` or `document_id` says which document should be found; `answer_contains` (optional) must
appear in the retrieved chunk for it to count as a hit.

Usage (service settings come from .env, documents must already be indexed):
  python scripts/evaluate.py eval/questions.example.jsonl --k 5
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select  # noqa: E402

from app.container import build_container  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.db.models import DocStatus, Document  # noqa: E402


def is_hit(doc, case: dict) -> bool:
    """Does this retrieved chunk answer the test case?"""
    meta = doc.metadata
    if case.get("document_id") and meta["document_id"] != case["document_id"]:
        return False
    if case.get("filename") and meta["filename"] != case["filename"]:
        return False
    needle = case.get("answer_contains")
    return not needle or needle.lower() in doc.page_content.lower()


async def evaluate(path: Path, k: int) -> None:
    """Run every question through the three set-ups and print hit@k / MRR."""
    container = build_container(get_settings())
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    async with container.sessionmaker() as session:
        ready = list((await session.scalars(select(Document.id).where(Document.status == DocStatus.READY))).all())
    if not ready:
        sys.exit("No indexed documents. Upload documents first.")

    modes = ["dense", "hybrid"] + (["hybrid+rerank"] if container.reranker else [])
    totals = {m: {"hits": 0, "rr": 0.0} for m in modes}
    for case in cases:
        line = [case["question"][:60].ljust(60)]
        for mode in modes:
            retriever = container.retriever.for_documents(ready, "hybrid" if mode == "hybrid+rerank" else mode)
            docs = await retriever.ainvoke(case["question"])
            if mode == "hybrid+rerank":
                reranker = container.reranker.model_copy(update={"top_n": k, "min_score": 0.0})
                docs = list(await reranker.acompress_documents(docs, case["question"]))
            rank = next((i for i, d in enumerate(docs[:k], start=1) if is_hit(d, case)), None)
            totals[mode]["hits"] += rank is not None
            totals[mode]["rr"] += 1 / rank if rank else 0.0
            line.append(f"{mode}: {rank or '-'}")
        print("  ".join(line))

    print(f"\n{len(cases)} questions, k={k}")
    for mode in modes:
        print(f"{mode:15s} hit@{k} = {totals[mode]['hits'] / len(cases):.2f}   MRR = {totals[mode]['rr'] / len(cases):.3f}")
    await container.engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("questions", type=Path)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(evaluate(args.questions, args.k))
