# HRD Intelligence — AI Service (RAG)

Document ingestion and RAG chat for **HRD Intelligence**, built with **FastAPI + LangChain + LangGraph**,
with a web console that shows the time of every pipeline stage.

| Part | Choice |
|---|---|
| LLM | **Qwen3.5-4B** (`qwen3.5:4b` on Ollama, thinking off) — vLLM on a GPU server in production |
| Embeddings / reranker | BAAI/bge-m3 · BAAI/bge-reranker-v2-m3 (TEI) |
| OCR | **auto**: PaddleOCR first, **Surya** (surya-ocr 0.17) when the page is Khmer or unclear |
| Vector DB | PostgreSQL 16 + pgvector (HNSW + full-text, fused with RRF) |
| UI | Console at **http://localhost:8000/console** (upload, chat, pipeline timings) |

Full explanation: [`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md).

---

## Quick start

After the one-time setup below, start Docker Desktop, then from `E:\HRD Intelligance` run:

```powershell
cd ai-service
docker compose up -d
.\.venv\Scripts\python -m uvicorn app.main:app --port 8000
```

Open **http://localhost:8000/console**. Stop: `Ctrl+C`, then `docker compose stop`.

---

## 1. One-time setup

Prerequisites: Python 3.12, Docker Desktop (WSL2 + NVIDIA GPU), ~15 GB free disk **on the drive Docker and
your user profile use (C:)**.

```bash
cd "E:/HRD Intelligance/ai-service"
```

**a. Python environment of the service**

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python -m pip install -e ".[dev,ocr-paddle]"
```

**b. Surya OCR environment** (separate on purpose: PyTorch and PaddlePaddle conflict in one environment)

```bash
python -m venv .venv-surya
```

```bash
.venv-surya/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

```bash
.venv-surya/Scripts/python -m pip install -r requirements-surya.txt
```

**c. Settings**

```bash
cp .env.example .env
```

**d. Database, Ollama and reranker containers + models**

```bash
docker compose up -d
```

```bash
docker compose exec ollama ollama pull qwen3.5:4b
```

```bash
docker compose exec ollama ollama pull bge-m3
```

```bash
docker compose exec ollama sh -c "printf 'FROM bge-m3\nPARAMETER num_gpu 0\n' > /tmp/M && ollama create bge-m3-cpu -f /tmp/M"
```

**e. Database schema**

```bash
.venv/Scripts/alembic upgrade head
```

---

## 2. Run (every time)

1. Start **Docker Desktop** and wait for *Engine running*.
2. Start the containers:

```bash
docker compose up -d
```

3. Start the service (keep this terminal open):

```bash
.venv/Scripts/python -m uvicorn app.main:app --port 8000
```

4. Open **http://localhost:8000/console** — the sidebar pill must say **All services ready**
   (the reranker needs ~1 minute after `docker compose up`).

Stop: `Ctrl+C` in the terminal, then `docker compose stop`.

---

## 3. Test

### Automated tests

```bash
.venv/Scripts/python -m pytest -m "not integration"
```

```bash
.venv/Scripts/python -m pytest tests/integration
```

The integration tests need the database container (`docker compose up -d db`); they use fake models, so they
run in seconds.

### End-to-end test in the console

Create sample files first (Word policy, Excel attendance, PowerPoint lesson, scanned notice):

```bash
.venv/Scripts/python scripts/make_samples.py
```

| Step | Where | Expected |
|---|---|---|
| 1. Upload everything in `samples/` (drag & drop), folder `policies` | Data management | Status Queued → Indexing → **Ready**; the *Index time* column fills in |
| 2. Click a document | Data management | Drawer with the **ingestion waterfall** (queue, parse, OCR, chunk, embed, store) and every chunk |
| 3. Upload a Khmer scan / photo (PNG/JPG) | Data management | OCR column shows **Surya**; drawer shows the fallback reason (e.g. *low PaddleOCR confidence*) |
| 4. New chat → ask *"How much is the meal allowance?"* | AI assistant | Streamed answer **60 USD per month [1]**, source chip, timing bar |
| 5. Click **Pipeline details** under the answer | AI assistant | Waterfall: rewrite, embed, vector search, keyword search, rerank, LLM first token, generation, total |
| 6. Ask *"And the Korean language one?"* | AI assistant | Follow-up is rewritten (see *Search query*); answer **25 USD** |
| 7. Untick all documents except the notice, ask the meal question again | AI assistant | **"I couldn't find this in your documents."** (permission filter; LLM skipped) |
| 8. Open **Pipeline log** | Pipeline log | Average per stage + one row per answer / indexing run; click a row for its waterfall |
| 9. Open **Model stack** | Model stack | Qwen3.5-4B, bge-m3, reranker, OCR auto, pgvector, all **Ready** |

### Retrieval quality

```bash
.venv/Scripts/python scripts/evaluate.py eval/questions.example.jsonl --k 5
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Surya fails with *"The paging file is too small" (os error 1455)* | Windows ran out of memory: free space on **C:** (Windows grows its page file there), close other apps, or `docker compose stop reranker` while indexing Khmer scans. Moving Docker's disk to E: (Docker Desktop → Settings → Resources → Advanced) frees ~40 GB on C:. |
| First answer takes a minute | The model is loading into the GPU; later answers are faster. |
| Answers are empty | `LLM_EXTRA_BODY={"reasoning_effort": "none"}` must be in `.env` (Qwen3.5 otherwise spends the answer budget on thinking). |
| Docker Desktop crashes at start | Rename `%LOCALAPPDATA%\Docker\run` (stale socket files) and start Docker again. |
| Console says session expired | The service restarted — reload the page. |
