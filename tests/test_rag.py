"""RAG helpers and the LangGraph chat flow (with fake retriever / LLM, no database)."""

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from app.rag.graph import build_rag_graph
from app.rag.prompts import (
    NO_DOCUMENTS_MESSAGE,
    NOT_FOUND_MESSAGE,
    ThinkTagFilter,
    extract_citations,
    strip_think,
)
from app.rag.retriever import reciprocal_rank_fusion
from tests.fakes import FakeRetriever, fake_chat_model


def doc(document_id: str, text: str, n: int = 0) -> Document:
    return Document(
        page_content=f"[File: {document_id}.pdf | Page 1]\n{text}",
        metadata={"document_id": document_id, "filename": f"{document_id}.pdf", "page": 1, "chunk_id": n, "rrf_score": 0.03},
    )


def test_rrf_rewards_items_found_by_both_searches():
    scores = reciprocal_rank_fusion([[1, 2, 3], [3, 4]], k=60)
    assert max(scores, key=scores.get) == 3
    assert scores[1] > scores[4]


def test_extract_citations_maps_markers_in_order_and_ignores_unknown():
    sources = [doc("a", "alpha"), doc("b", "beta")]
    cites = extract_citations("Beta [2] then alpha [1][2] and [7].", sources)
    assert [c["n"] for c in cites] == [2, 1]
    assert cites[0]["document_id"] == "b" and cites[0]["snippet"] == "beta"


def test_think_filter_handles_tags_split_across_tokens():
    f = ThinkTagFilter()
    out = "".join(f.feed(t) for t in ["Hel", "lo <thi", "nk>secret", " stuff</th", "ink> world"]) + f.flush()
    assert out == "Hello  world"
    assert strip_think("<think>x</think>Answer") == "Answer"


def build(settings, token_counter, documents, answer="Meal allowance is 50 [1]."):
    retriever = FakeRetriever(documents)
    graph = build_rag_graph(
        settings=settings,
        chat_model=fake_chat_model(answer),
        retriever=retriever,
        reranker=None,
        token_counter=token_counter,
    )
    return graph, retriever


async def test_graph_answers_with_citations(settings, token_counter):
    graph, _ = build(settings, token_counter, [doc("policy", "Meal allowance is 50 dollars.")])
    state = await graph.ainvoke({"question": "meal allowance?", "history": [], "allowed_document_ids": ["policy"]})
    assert state["answer"] == "Meal allowance is 50 [1]."
    assert state["citations"][0]["document_id"] == "policy"


async def test_graph_without_documents_skips_llm(settings, token_counter):
    graph, retriever = build(settings, token_counter, [doc("policy", "x")])
    state = await graph.ainvoke({"question": "q", "history": [], "allowed_document_ids": []})
    assert state["answer"] == NO_DOCUMENTS_MESSAGE and state["citations"] == []
    assert retriever.queries == []


async def test_graph_not_found_when_nothing_retrieved(settings, token_counter):
    graph, _ = build(settings, token_counter, [doc("other", "unrelated")])
    state = await graph.ainvoke({"question": "q", "history": [], "allowed_document_ids": ["policy"]})
    assert state["answer"] == NOT_FOUND_MESSAGE


async def test_graph_rewrites_follow_up_questions(settings, token_counter):
    graph, retriever = build(settings, token_counter, [doc("policy", "Korean allowance")], answer="Korean allowance policy")
    history = [HumanMessage("What is the meal allowance?"), AIMessage("It is 50 dollars.")]
    await graph.ainvoke({"question": "and the Korean one?", "history": history, "allowed_document_ids": ["policy"]})
    assert retriever.queries == ["Korean allowance policy"]


async def test_graph_streams_generate_tokens(settings, token_counter):
    graph, _ = build(settings, token_counter, [doc("policy", "Meal allowance is 50 dollars.")])
    tokens = []
    async for mode, payload in graph.astream(
        {"question": "meal?", "history": [], "allowed_document_ids": ["policy"]}, stream_mode=["messages", "updates"]
    ):
        if mode == "messages" and payload[1].get("langgraph_node") == "generate":
            tokens.append(payload[0].content)
    assert "".join(tokens) == "Meal allowance is 50 [1]."
    assert len(tokens) > 1


async def test_graph_records_stage_timings(settings, token_counter):
    graph, _ = build(settings, token_counter, [doc("policy", "Meal allowance is 50 dollars.")])
    state = await graph.ainvoke({"question": "meal?", "history": [], "allowed_document_ids": ["policy"]})
    trace = state["trace"]
    for key in ("rewrite_ms", "embed_ms", "dense_ms", "lexical_ms", "retrieve_ms", "rerank_ms", "llm_ttft_ms", "generate_ms"):
        assert isinstance(trace[key], int), key
    assert trace["kept"] == 1 and trace["candidates"] == 1 and trace["input_tokens"] > 0


async def test_graph_trace_marks_skipped_llm(settings, token_counter):
    graph, _ = build(settings, token_counter, [doc("other", "x")])
    state = await graph.ainvoke({"question": "q", "history": [], "allowed_document_ids": ["policy"]})
    assert state["trace"]["llm_skipped"] == "no relevant sources" and "generate_ms" not in state["trace"]
