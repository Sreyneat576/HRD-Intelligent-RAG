# HRD Intelligence — AI Service (RAG)

Document ingestion and RAG chat for **HRD Intelligence**, built with **FastAPI + LangChain + LangGraph**.

Users upload HRD documents (PDF, DOCX, PPTX, XLSX, CSV, TXT, MD, PNG, JPEG). The service parses them, with
OCR for scans, splits them by structure, embeds them with **bge-m3** and stores them in
**PostgreSQL + pgvector**. The chatbot then answers questions using only the documents each user is allowed to
read. It combines hybrid search (vector + full-text), **bge-reranker-v2-m3** reranking and a local
**Qwen3** LLM, and it streams answers with citations.

📘 **Full explanation of the project:** [`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md). It covers the
architecture, model choices, the ingestion and chat graphs, the API, Spring integration, the data model and
production setup.

## Stack

| Part | Choice |
|---|---|
| API | FastAPI, Server-Sent Events for streaming |
| Orchestration | LangChain (loaders, retriever, reranker, prompts) + LangGraph (ingestion graph, RAG graph) |
| LLM | Qwen3: `qwen3:4b-instruct` on Ollama (dev), `Qwen/Qwen3-8B` on vLLM (prod) |
| Embeddings | BAAI/bge-m3 (1024-dim) |
| Reranker | BAAI/bge-reranker-v2-m3 (TEI or vLLM) |
| Vector DB / index | PostgreSQL 16 + pgvector 0.8: HNSW on `halfvec(1024)` + GIN full-text, fused with RRF |
| Parsing / OCR | PyMuPDF, python-docx, python-pptx, pandas + openpyxl, PaddleOCR (PP-StructureV3 optional) / Tesseract |
| Test UI | Streamlit |

## Quick start (local development)

**Prerequisites:**
- Python 3.11+
- Docker Desktop (with WSL2 + NVIDIA GPU support for Ollama)
- About 10 GB free disk for images and models

```bash
cd ai-service

# 1. Python environment
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -e ".[dev,ui,ocr-paddle]"

# 2. Settings
copy .env.example .env            # Linux/macOS: cp .env.example .env

# 3. Infrastructure: Postgres+pgvector (:5433), Ollama (:11434, GPU), reranker (:8081)
docker compose up -d
docker compose exec ollama ollama pull qwen3:4b-instruct
docker compose exec ollama ollama pull bge-m3
#    small GPU (4 GB): run the embedding model on the CPU so the LLM keeps the GPU (.env uses EMBED_MODEL=bge-m3-cpu)
docker compose exec ollama sh -c "printf 'FROM bge-m3\nPARAMETER num_gpu 0\n' > /tmp/M && ollama create bge-m3-cpu -f /tmp/M"

# 4. Database schema
alembic upgrade head

# 5. Run the API          -> http://localhost:8000/docs
uvicorn app.main:app --reload --port 8000

# 6. Run the test UI       -> http://localhost:8501   (second terminal)
streamlit run ui/app.py
```

The first reranker start downloads `bge-reranker-v2-m3` (~2 GB) inside the TEI container. Check progress with
`docker compose logs -f reranker`. Until it is ready, chat still works without reranking: the service falls
back to the hybrid ranking.

### Run everything in Docker instead

```bash
docker compose --profile app up -d --build   # API on :8000, UI on :8501 (uses Tesseract OCR in the container)
```

## Using the test UI

Need test files? Run `python scripts/make_samples.py`. It writes a Word allowance policy, an Excel attendance sheet,
a PowerPoint lesson and a scanned-notice PNG (for OCR) into `samples/`. The project spec PDF is also there.

1. **Documents tab:**
   1. Upload one or more files into a folder.
   2. Watch the status change from 🕓 pending to ⚙️ processing to ✅ ready.
   3. Open **Inspect a document → Show chunks** to see exactly how a file was split.
2. **Chat tab:**
   1. Create a *general* or *folder* chat.
   2. Choose which documents the user may read. This simulates the permissions Spring would send.
   3. Ask questions. Answers stream in with **📎 Sources**.
   4. **🔎 Retrieval debug** shows the search query and every source's rerank / RRF score and its dense and keyword ranks.
3. The **sidebar** shows whether the database, LLM, embeddings and reranker are reachable, and lets you switch the acting user and role.

## API in one minute

All endpoints need `Authorization: Bearer <SERVICE_TOKEN>`. Chat endpoints also need `X-User-Id`.

**1. Upload a document:**

```bash
curl -X POST http://localhost:8000/api/v1/documents -H "Authorization: Bearer change-me" -F "file=@policy.pdf" -F "document_id=doc-1" -F "owner_id=admin-1" -F "folder_id=policies"
```

**2. Check its status:**

```bash
curl http://localhost:8000/api/v1/documents/doc-1 -H "Authorization: Bearer change-me"
```

**3. Start a chat session:**

```bash
curl -X POST http://localhost:8000/api/v1/chat/sessions -H "Authorization: Bearer change-me" -H "X-User-Id: admin-1" -H "Content-Type: application/json" -d "{\"scope\": \"general\"}"
```

**4. Ask a question.** Use `"stream": false` for one JSON reply, or leave it out to get a Server-Sent Events stream:

```bash
curl -X POST http://localhost:8000/api/v1/chat/sessions/<session-id>/messages -H "Authorization: Bearer change-me" -H "X-User-Id: admin-1" -H "Content-Type: application/json" -d "{\"content\": \"What is the meal allowance?\", \"allowed_document_ids\": [\"doc-1\"], \"stream\": false}"
```

The complete endpoint list and the Spring integration checklist are in the
[project guide](docs/PROJECT_GUIDE.md#9-api-reference).

## Tests

**Unit tests** need nothing running:

```bash
pytest -m "not integration"
```

**Integration tests** need Postgres+pgvector:

```bash
docker compose up -d db
```

```bash
pytest tests/integration
```

**Retrieval quality check** (hit@5 and MRR for dense, hybrid and hybrid+rerank) runs against the documents you have already indexed:

```bash
python scripts/evaluate.py eval/questions.example.jsonl --k 5
```

## Project layout

```
app/
  main.py, container.py     app factory and dependency wiring
  core/                     settings, service-token auth
  db/                       SQLAlchemy models, async sessions
  llm/                      LangChain chat model / embeddings / reranker, bge-m3 tokenizer
  ingestion/                loaders (per file type), OCR, splitter, LangGraph ingestion graph, worker
  rag/                      hybrid retriever, prompts, LangGraph RAG graph
  services/, api/           business logic and REST / SSE endpoints
ui/app.py                   Streamlit test console
migrations/                 Alembic schema (HNSW + GIN indexes)
scripts/evaluate.py         retrieval evaluation
tests/                      unit + integration tests
docs/PROJECT_GUIDE.md       full project documentation
```

## Switching to the production GPU server

Point the settings at vLLM. Nothing else changes:

```env
LLM_BASE_URL=http://gpu-server:8001/v1
LLM_MODEL=Qwen/Qwen3-8B
LLM_EXTRA_BODY={"chat_template_kwargs": {"enable_thinking": false}}
EMBED_BASE_URL=http://gpu-server:8002/v1
EMBED_MODEL=BAAI/bge-m3
RERANK_API=vllm
RERANK_BASE_URL=http://gpu-server:8003
```

See [project guide §13](docs/PROJECT_GUIDE.md#13-moving-to-production-vllm) for the vLLM commands.
