"""Builds and holds every long-lived object of the service (DB engine, LangChain models, retriever,
LangGraph graphs, ingestion worker). Tests pass fakes for the model objects."""

from dataclasses import dataclass
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.db.session import create_engine, create_sessionmaker
from app.ingestion.graph import build_ingestion_graph
from app.ingestion.ocr import OcrEngine, build_ocr_engine
from app.ingestion.splitter import StructureAwareSplitter
from app.ingestion.worker import IngestionWorker
from app.llm.providers import build_chat_model, build_embeddings, build_reranker
from app.llm.reranker import RemoteReranker
from app.llm.tokenizer import TokenCounter, get_token_counter
from app.rag.graph import build_rag_graph
from app.rag.retriever import HybridPgRetriever

_DEFAULT: Any = object()


@dataclass
class Container:
    settings: Settings
    engine: AsyncEngine
    sessionmaker: async_sessionmaker[AsyncSession]
    chat_model: BaseChatModel
    embeddings: Embeddings
    reranker: RemoteReranker | None
    ocr: OcrEngine
    token_counter: TokenCounter
    splitter: StructureAwareSplitter
    retriever: HybridPgRetriever
    ingestion_graph: Any
    rag_graph: Any
    worker: IngestionWorker


def build_container(
    settings: Settings,
    *,
    chat_model: BaseChatModel | None = None,
    embeddings: Embeddings | None = None,
    reranker: RemoteReranker | None = _DEFAULT,
    ocr: OcrEngine | None = None,
) -> Container:
    """Create all service objects from settings; any model object can be overridden (used by tests)."""
    engine = create_engine(settings.database_url)
    sessionmaker = create_sessionmaker(engine)
    chat_model = chat_model or build_chat_model(settings)
    embeddings = embeddings or build_embeddings(settings)
    reranker = build_reranker(settings) if reranker is _DEFAULT else reranker
    ocr = ocr or build_ocr_engine(
        settings.ocr_engine,
        lang=settings.ocr_lang,
        structure=settings.ocr_structure,
        paddle_mkldnn=settings.ocr_paddle_mkldnn,
        surya_python=settings.surya_python,
        surya_device=settings.surya_device,
        fallback_min_confidence=settings.ocr_fallback_min_confidence,
        fallback_min_english_ratio=settings.ocr_fallback_min_english_ratio,
    )
    token_counter = get_token_counter(settings.tokenizer_name)

    splitter = StructureAwareSplitter(
        token_counter,
        max_tokens=settings.chunk_max_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
        slide_merge_tokens=settings.slide_merge_tokens,
        table_rows_per_chunk=settings.table_rows_per_chunk,
    )
    retriever = HybridPgRetriever(
        sessionmaker=sessionmaker,
        embeddings=embeddings,
        dense_k=settings.retrieval_dense_k,
        lexical_k=settings.retrieval_lexical_k,
        rrf_k=settings.rrf_k,
        top_k=settings.rerank_candidates,
        hnsw_ef_search=settings.hnsw_ef_search,
    )
    ingestion_graph = build_ingestion_graph(
        settings=settings, ocr=ocr, splitter=splitter, embeddings=embeddings, sessionmaker=sessionmaker
    )
    rag_graph = build_rag_graph(
        settings=settings,
        chat_model=chat_model,
        retriever=retriever,
        reranker=reranker,
        token_counter=token_counter,
    )
    worker = IngestionWorker(
        settings, sessionmaker, ingestion_graph, on_job_done=None if settings.surya_keep_alive else ocr.release
    )
    return Container(
        settings=settings,
        engine=engine,
        sessionmaker=sessionmaker,
        chat_model=chat_model,
        embeddings=embeddings,
        reranker=reranker,
        ocr=ocr,
        token_counter=token_counter,
        splitter=splitter,
        retriever=retriever,
        ingestion_graph=ingestion_graph,
        rag_graph=rag_graph,
        worker=worker,
    )
