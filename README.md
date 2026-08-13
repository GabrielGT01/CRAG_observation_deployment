# CRAG Assistant — Phase 4: Observability, Deployment & Monitoring

A Corrective RAG (CRAG) system shipped with LangSmith tracing, Dockerized and
deployed to AWS (ECR → EC2), covering the Phase 4 "ship it like production"
assignment.

## Scope of this repo vs. the full assignment

This repo/notebook focuses specifically on **LangSmith observability + AWS
deployment**. Two deliverables from the assignment live elsewhere or take a
different shape here — documented honestly rather than silently:

| Deliverable | Status | Note |
|---|---|---|
| Golden Dataset (20 Q&A pairs) | ✅ Done | Built in a companion notebook — `https://github.com/GabrielGT01/RAG_Evaluation_Framework-` |
| LangSmith tracing | ✅ Done | Every request logged with input, retrieved chunks, generation, and quality scores. Screenshots below. |
| Dashboard (volume, latency, top queries, RAGAS over time) | 🔁 Substituted | No custom dashboard was built. LangSmith's own project view serves this role — it already surfaces per-run latency, token cost, and the full input/output/metadata per trace. See screenshots. |
| Docker + FastAPI, concurrent-safe | 🔁 Substituted | Built with **Streamlit instead of FastAPI** (see rationale below). Concurrency handled via Streamlit's per-session thread isolation rather than an ASGI event loop. |
| Rate limiting | ✅ Done | In-memory sliding-window limiter, per session. |
| Health check endpoint | ✅ Done | Streamlit's built-in `/_stcore/health` — no custom route needed. |
| Runbook | ✅ Done | See [Runbook](#runbook) below. |
| Stretch: AWS EC2 | ✅ Done | Deployed via ECR → EC2. |

### Why Streamlit instead of FastAPI

The assignment asks for FastAPI; this build uses Streamlit instead. The
concurrency and reliability requirements are still met, just via different
mechanisms:

- **Concurrent requests without crashing** — Streamlit's server runs each
  connected browser session in its own thread with isolated `session_state`.
  Per-user state (retriever, chat history, graph) never touches module-level
  globals, so one session's failure or load doesn't affect another's, and an
  unhandled exception in one session doesn't take the process down.
- **Health check** — Streamlit ships `/_stcore/health` out of the box, which
  is what a FastAPI build would have had to hand-roll as a custom route.
- **Rate limiting** — implemented at the app layer with a `threading.Lock`
  guarding a shared sliding-window counter, independent of the web framework
  underneath.

Trade-off, for the record: Streamlit isn't a REST API — there's no JSON
endpoint another service could call programmatically, only the UI. If
programmatic access becomes a requirement, that's the concrete reason to
migrate to FastAPI later.

## Architecture

```
        START
          │
      retrieve   ────────►  FAISS retriever (built from PDF / TXT / URL sources)
          │
   grade_documents  ─────►  LLM-as-judge: is each chunk actually relevant?
          │
   (conditional edge)
     ┌────┴─────┐
     ▼          ▼
  generate   transform_query ──► web_search ──► generate
     │
    END
```

Every `generate` call is wrapped in LangSmith's `@traceable`, and the graph
itself runs under `LANGCHAIN_TRACING_V2=true`, so the full retrieve → grade →
generate path is captured per request with no manual instrumentation beyond
env vars and the decorator.

## LangSmith tracing — what gets logged

Each traced run captures, per turn:

- `question` — the user's input
- `documents` — the retrieved chunks, each with `id`, `page_content`, and
  `metadata`
- `generation` — the model's answer
- `web_search` — whether the corrective fallback fired
- `faithfulness`, `answer_relevancy`, `context_precision` — the inline
  LLM-as-judge scores for that specific answer

**Example trace — Turn 2**

![LangSmith trace showing question, generation, and per-turn metrics](https://github.com/GabrielGT01/CRAG_observation_deployment/blob/main/langsmith_tracing1.png)

**Example trace — Turn 3**

![LangSmith trace showing retrieved documents and faithfulness score](https://github.com/GabrielGT01/CRAG_observation_deployment/blob/main/langsmith_tracing3.png)

> The assignment asks for 5 traced-request screenshots — two are included
> here as a representative sample from one thread. Grab three more from
> different threads/questions in your LangSmith project before submitting,
> if this is going in for grading.

## Project structure

```
src/
├── document_ingestion/   # PDF / TXT / URL loading + chunking
├── vectorstore/          # FAISS + OpenAI embeddings
├── state/                # GraphState definition
├── node/                 # Graph nodes: retrieve, grade, generate, rewrite, web search
├── graph/                # LangGraph StateGraph wiring
├── pipeline/             # IngestionPipeline: sources -> retriever, one call
└── rag_metric/           # Faithfulness / relevancy / precision graders (LLM-as-judge)
app.py                    # Streamlit UI — chat, ingestion, rate limiting
Dockerfile
.dockerignore
requirements.txt
```

## Setup

### 1. Install

```bash
git clone <your-repo-url>
cd <repo-name>
uv pip install -r requirements.txt        # or: pip install -r requirements.txt
```

### 2. Environment variables

```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
LANGSMITH_API_KEY=lsv2_...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=TESTING-phase
```

`.env` is excluded via `.gitignore` and `.dockerignore`.

### 3. Run locally

```bash
streamlit run app.py
```

Open `http://localhost:8501`, upload a document or paste a URL, ask
questions. Every question is traced to your LangSmith project in real time —
watch it appear under **Tracing** in the LangSmith UI as you use the app.

## Runbook

Deploying and monitoring this system from scratch, end to end.

### Deploy

1. **Prerequisites**: Docker Desktop, AWS CLI v2, an AWS account, your
   `OPENAI_API_KEY` / `TAVILY_API_KEY` / `LANGSMITH_API_KEY`.
2. **IAM setup (once)**:
   - IAM User for your machine, with `AmazonEC2ContainerRegistryFullAccess` +
     `AmazonEC2FullAccess`, used to push images.
   - IAM Role for EC2, with `AmazonEC2ContainerRegistryReadOnly`, attached to
     the instance so it can pull images without stored credentials.
3. **Build & push** (Apple Silicon needs the platform flag — EC2 runs
   `linux/amd64`):
   ```bash
   docker buildx build --platform linux/amd64 -t crag-bot:latest .
   docker tag crag-bot:latest <account-id>.dkr.ecr.<region>.amazonaws.com/crag-bot:latest
   aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com
   docker push <account-id>.dkr.ecr.<region>.amazonaws.com/crag-bot:latest
   ```
4. **Launch EC2**: `t3.micro`, security group open on port 22 (restrict to
   your IP) and port **8501** (0.0.0.0/0), IAM role from step 2 attached.
5. **On the instance**: install Docker, pull the image, create a `.env` file
   with `chmod 600`, then:
   ```bash
   docker run -d --name crag-bot -p 8501:8501 --env-file .env \
     --restart unless-stopped \
     <account-id>.dkr.ecr.<region>.amazonaws.com/crag-bot:latest
   ```
6. **Verify**: `docker ps` shows `(healthy)` after ~15–20s (Dockerfile
   `HEALTHCHECK` hits `/_stcore/health`); `curl -f http://localhost:8501/_stcore/health`
   returns `ok`; browser to `http://<EC2-PUBLIC-IP>:8501`.

### Monitor

- **Traces**: LangSmith project (`LANGCHAIN_PROJECT` value) → **Tracing** —
  every request shows input, retrieved chunks, generation, latency, token
  cost, and the three quality scores, filterable by thread/date.
- **Container health**: `docker ps` (health status), `docker logs -f crag-bot`
  (real-time logs — `PYTHONUNBUFFERED=1` ensures nothing sits buffered),
  `docker stats crag-bot` (CPU/memory).
- **Rate limiting**: exposed in the app sidebar (requests / window), so
  hitting the limit is visible to the user, not just failing silently.

### Redeploy after a code change

```bash
# local
docker buildx build --platform linux/amd64 -t crag-bot:latest .
docker tag crag-bot:latest <account-id>.dkr.ecr.<region>.amazonaws.com/crag-bot:latest
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/crag-bot:latest

# on EC2
docker pull <account-id>.dkr.ecr.<region>.amazonaws.com/crag-bot:latest
docker stop crag-bot && docker rm crag-bot
docker run -d --name crag-bot -p 8501:8501 --env-file .env \
  --restart unless-stopped \
  <account-id>.dkr.ecr.<region>.amazonaws.com/crag-bot:latest
```

### Rollback

Re-tag and push a previous known-good image tag (not just `latest`) to ECR,
then repeat the pull/stop/rm/run sequence on EC2 with that tag. Tagging
images with a version or commit SHA, not just `latest`, is worth doing going
forward — right now there's no way to roll back to a specific prior build.

## Security notes

- Never bake API keys into the image — inject at runtime via `--env-file`.
- EC2 pulls from ECR via an IAM Role, no stored AWS credentials on the server.
- SSH restricted to your own IP, not `0.0.0.0/0`.
- Rotate any key immediately if it's ever been exposed in a file, commit, or
  log — don't wait to confirm it was actually used maliciously first.

## Stack

LangChain · LangGraph · LangSmith · OpenAI (`gpt-4.1-mini` / `gpt-4o-mini`) ·
Tavily · FAISS · Streamlit · Docker · AWS (ECR, EC2)

## License

MIT
