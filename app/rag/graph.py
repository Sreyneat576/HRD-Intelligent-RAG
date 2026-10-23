"""LangGraph RAG pipeline for one chat turn:

    START --(any allowed documents?)--> rewrite_query -> retrieve -> rerank --(any source left?)--> generate -> cite -> END
                    \\-- no ----------> no_documents ---------------------------------> cite           \\-- no --> not_found -> cite

rewrite_query : turns a follow-up ("what about instructors?") into a standalone search query
retrieve      : hybrid pgvector + full-text search, restricted to the allowed document ids
rerank        : bge-reranker-v2-m3 keeps the best `context_top_k` chunks above the score threshold
generate      : LLM answers from the numbered sources only, citing [n]
not_found / no_documents : fixed answers without calling the LLM
cite          : maps the [n] markers in the answer to source documents
"""

import logging
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, trim_messages
from langgraph.graph import END, START, StateGraph

from app.core.config import Settings
from app.llm.reranker import RemoteReranker
from app.llm.tokenizer import TokenCounter
from app.rag.prompts import (
    NO_DOCUMENTS_MESSAGE,
    NOT_FOUND_MESSAGE,
    answer_prompt,
    extract_citations,
    format_sources,
    rewrite_prompt,
    strip_think,
)
from app.rag.retriever import HybridPgRetriever

log = logging.getLogger(__name__)


class RagState(TypedDict, total=False):
    # input
    question: str
    history: list[BaseMessage]
    allowed_document_ids: list[str]
    # produced by the nodes
    search_query: str
    candidates: list[Document]
    sources: list[Document]
    rerank_error: str | None
    answer: str
    citations: list[dict]


def build_rag_graph(
    *,
    settings: Settings,
    chat_model: BaseChatModel,
    retriever: HybridPgRetriever,
    reranker: RemoteReranker | None,
    token_counter: TokenCounter,
):
    """Wire the RAG nodes into a compiled LangGraph StateGraph."""
    rewrite_chain = rewrite_prompt | chat_model

    def count_message_tokens(messages: list[BaseMessage]) -> int:
        """Token counter used to trim the chat history to `history_max_tokens`."""
        return sum(token_counter.count(str(m.content)) + 4 for m in messages)

    def has_documents(state: RagState) -> str:
        """Route: skip everything when the user may not read any document."""
        return "rewrite_query" if state.get("allowed_document_ids") else "no_documents"

    async def rewrite_query(state: RagState) -> RagState:
        """Make the question self-contained using the recent conversation (only for follow-ups)."""
        question = state["question"]
        history = state.get("history") or []
        if not history or not settings.query_rewrite:
            return {"search_query": question}
        try:
            message = await rewrite_chain.ainvoke(
                {"history": history[-settings.history_turns * 2 :], "question": question}
            )
            query = strip_think(str(message.content)).strip().strip('"').splitlines()[0].strip()
            return {"search_query": query or question}
        except Exception as exc:
            log.warning("Query rewrite failed, using the original question: %s", exc)
            return {"search_query": question}

    async def retrieve(state: RagState) -> RagState:
        """Hybrid search over the allowed documents only."""
        documents = await retriever.for_documents(state["allowed_document_ids"]).ainvoke(state["search_query"])
        return {"candidates": documents}

    async def rerank(state: RagState) -> RagState:
        """Cross-encoder rerank; falls back to the fused ranking if the reranker is down."""
        candidates = state.get("candidates") or []
        if not candidates:
            return {"sources": []}
        if reranker is None:
            return {"sources": candidates[: settings.context_top_k]}
        try:
            return {"sources": list(await reranker.acompress_documents(candidates, state["search_query"]))}
        except Exception as exc:
            log.warning("Reranker unavailable, using hybrid ranking: %s", exc)
            return {"sources": candidates[: settings.context_top_k], "rerank_error": str(exc)}

    def has_sources(state: RagState) -> str:
        """Route: answer with the LLM only when something relevant was found."""
        return "generate" if state.get("sources") else "not_found"

    async def generate(state: RagState) -> RagState:
        """Ask the LLM to answer from the numbered sources (tokens are streamed by the caller)."""
        sources = format_sources(state["sources"])
        prompt_without_history = answer_prompt.format_messages(history=[], sources=sources, question=state["question"])
        # History gets what is left of the context window after the sources and the answer budget.
        free = settings.llm_context_window - settings.llm_max_tokens - count_message_tokens(prompt_without_history)
        history = trim_messages(
            state.get("history") or [],
            max_tokens=max(0, min(settings.history_max_tokens, free - 100)),
            token_counter=count_message_tokens,
            strategy="last",
            start_on="human",
            allow_partial=False,
        )
        messages = answer_prompt.format_messages(history=history, sources=sources, question=state["question"])
        response = await chat_model.ainvoke(messages)
        return {"answer": strip_think(str(response.content))}

    async def not_found(state: RagState) -> RagState:
        """Nothing relevant in the allowed documents."""
        return {"answer": NOT_FOUND_MESSAGE, "sources": []}

    async def no_documents(state: RagState) -> RagState:
        """The user has no documents to search."""
        return {"answer": NO_DOCUMENTS_MESSAGE, "search_query": state["question"], "sources": []}

    async def cite(state: RagState) -> RagState:
        """Turn [n] markers in the answer into citation records."""
        return {"citations": extract_citations(state.get("answer", ""), state.get("sources") or [])}

    graph = StateGraph(RagState)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("rerank", rerank)
    graph.add_node("generate", generate)
    graph.add_node("not_found", not_found)
    graph.add_node("no_documents", no_documents)
    graph.add_node("cite", cite)

    graph.add_conditional_edges(START, has_documents, ["rewrite_query", "no_documents"])
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_conditional_edges("rerank", has_sources, ["generate", "not_found"])
    graph.add_edge("generate", "cite")
    graph.add_edge("not_found", "cite")
    graph.add_edge("no_documents", "cite")
    graph.add_edge("cite", END)
    return graph.compile(name="rag")
