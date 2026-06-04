# 🤖 RAG LangGraph Chatbot

A production-grade, end-to-end **Retrieval-Augmented Generation (RAG) chatbot** built with
**LangGraph multi-agent architecture**. Uses the
["Attention Is All You Need"](https://arxiv.org/pdf/1706.03762.pdf) paper (Vaswani et al., 2017)
as its knowledge base.

Every component is free-tier. No paid APIs required to run this locally.

---

## 🖼️ Screenshot

```
 ─────────────────────────────────────────────────────────────────────
│  📖 RAG Chatbot — Attention Is All You Need                         │
 ─────────────────────────────────────────────────────────────────────
```

---

## 🏗️ System Architecture

```
                         USER QUERY
                              │
                    ┌─────────▼─────────┐
                    │   Memory Loader   │  ← Upstash Redis
                    │  (history + LTM)  │    short-term + long-term
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │      ROUTER       │  ← llama-3.1-8b-instant
                    │ (rag / web_search)│
                    └────┬─────────┬────┘
                         │         │
              [RAG path] │         │ [out-of-domain]
                         │         │
          ┌──────────────▼──┐  ┌───▼──────────────┐
          │  Query Rewriter  │  │   Web Search     │
          │  (8b-instant)    │  │   (Tavily)       │
          └──────────┬───────┘  └────────┬─────────┘
                     │                   │
          ┌──────────▼───────┐           │
          │    RETRIEVER      │          │
          │  Hybrid Search    │          │
          │  Dense + BM25     │          │
          │  (Qdrant Cloud)   │          │
          └──────────┬───────┘           │
                     │                   │
          ┌──────────▼───────┐           │
          │  DOCUMENT GRADER  │          │
          │  Score 0.0–1.0    │          │
          └──┬───────────┬───┘           │
        [≥1] │     [0]   │               │
             │  ┌────────▼─────────┐     │
             │  │ CORRECTIVE RAG   │     │
             │  │ (Tavily fallback)│     │
             │  └────────┬─────────┘     │
             │           │               │
          ┌──▼───────────▼───────────────▼───┐
          │           GENERATOR              │
          │     llama-3.3-70b-versatile      │
          │  Parent docs from MongoDB +      │
          │  chat history + memory context   │
          └──────────────┬───────────────────┘
                         │
          ┌──────────────▼───────────────┐
          │      RESPONSE VALIDATOR       │
          │  Hallucination / Relevance    │
          │  (8b-instant) — max 2 retries │
          └────┬─────────────────────────┘
          [valid]              [invalid → retry with reason]
               │
          ┌────▼────────────────────────┐
          │        MEMORY SAVER          │
          │  Short-term: Redis session   │
          │  Long-term:  summarise+save  │
          └─────────────────────────────┘
                         │
                    FINAL ANSWER + CITATIONS
                    (rendered in Streamlit UI)


INGESTION PIPELINE (run once before chatting):

  PDF
   ├── unstructured (hi_res) ──→ Text / Title / Table / Image / Equation
   ├── PyMuPDF ─────────────→ Raw image bytes (base64)
   └── Cross-ref extraction ──→ "see Figure 1", "Table 3" links

  Image ──→ Groq Vision (llama-4-scout) ──→ Text description
  Table ──→ HTML → Markdown
  Equation → detected + anchored to surrounding text

  Context Anchoring (images / tables / equations):
  [preceding 2 paragraphs] + [caption] + [content] + [following 2 paragraphs]

  Parent-Child Chunking:
  Parent (~1200 tokens) ──→ MongoDB Atlas (full context for LLM)
  Child  (~300  tokens) ──→ Qdrant Cloud  (vector search)
                              Dense:  all-MiniLM-L6-v2 (384-dim)
                              Sparse: BM25 (fastembed)
```

---

## 🧰 Tech Stack

| Layer | Tool | Purpose | Cost |
|---|---|---|---|
| **LLM — Generation** | [Groq](https://groq.com) `llama-3.3-70b-versatile` | Final answer generation | Free |
| **LLM — Fast tasks** | [Groq](https://groq.com) `llama-3.1-8b-instant` | Routing, grading, rewriting, validation | Free |
| **LLM — Vision** | [Groq](https://groq.com) `llama-4-scout-17b` | Image captioning | Free |
| **Agent Framework** | [LangGraph](https://langchain-ai.github.io/langgraph/) | Multi-agent stateful pipeline | Free |
| **LLM Framework** | [LangChain](https://langchain.com) | Prompts, messages, splitters | Free |
| **PDF Parsing** | [unstructured](https://unstructured.io) | Extract text/tables/images | Free |
| **Image Extraction** | [PyMuPDF](https://pymupdf.readthedocs.io) | Raw image bytes from PDF | Free |
| **Vector Database** | [Qdrant Cloud](https://cloud.qdrant.io) | Dense + sparse vector search | Free (1 GB) |
| **Dense Embeddings** | `all-MiniLM-L6-v2` via [sentence-transformers](https://sbert.net) | Semantic vectors (local) | Free |
| **Sparse Embeddings** | BM25 via [fastembed](https://qdrant.github.io/fastembed/) | Keyword vectors (local) | Free |
| **Document Store** | [MongoDB Atlas](https://cloud.mongodb.com) | Parent chunk storage | Free (512 MB) |
| **Memory** | [Upstash Redis](https://upstash.com) | Session + long-term memory | Free (10K cmd/day) |
| **Web Search** | [Tavily](https://tavily.com) | Fallback search | Free (1K/month) |
| **UI** | [Streamlit](https://streamlit.io) | Chat interface | Free |
| **API** | [FastAPI](https://fastapi.tiangolo.com) | Backend REST API | Free |
| **Package Manager** | [uv](https://docs.astral.sh/uv/) | Fast Python dependency management | Free |

---

## 📁 Project Structure

```
rag-langgraph-chatbot/
│
├── 📄 pyproject.toml          # uv project config + all dependencies
├── 📄 .python-version         # Python 3.11
├── 📄 .env.example            # Template — copy to .env and fill in keys
├── 📄 .gitignore              # Excludes .env, __pycache__, extracted_images, etc.
│
├── 📄 ingest_pipeline.py      # ONE-TIME: parse PDF → MongoDB + Qdrant
├── 📄 api.py                  # FastAPI backend (connects graph to UI)
├── 📄 main.py                 # CLI chatbot (alternative to UI)
├── 📄 ui.py                   # Streamlit chat interface
│
├── 📁 config.py               # All API keys + tuning parameters
│
├── 📁 ingestion/
│   ├── pdf_parser.py          # Parse PDF → DocumentElement list
│   ├── image_captioner.py     # Caption images using Groq vision
│   └── chunker.py             # Parent-child chunking strategy
│
├── 📁 vectorstore/
│   └── qdrant_store.py        # Qdrant hybrid search (dense + BM25)
│
├── 📁 document_store/
│   └── mongodb_store.py       # MongoDB parent chunk storage
│
├── 📁 memory/
│   ├── short_term.py          # In-session chat history (Redis)
│   └── long_term.py           # Cross-session memory summaries (Redis)
│
├── 📁 agents/
│   ├── router.py              # Classify query → rag or web_search
│   ├── query_rewriter.py      # Expand/clarify query for better retrieval
│   ├── retriever.py           # Hybrid search → fetch parent docs
│   ├── grader.py              # Score doc relevance (0.0–1.0)
│   ├── generator.py           # Generate answer + extract citations
│   ├── validator.py           # Check for hallucination/irrelevance
│   └── web_search.py          # Tavily web search fallback
│
└── 📁 graph/
    ├── state.py               # GraphState TypedDict (shared whiteboard)
    └── pipeline.py            # LangGraph wiring — nodes + edges
```

---

## ✅ Prerequisites

### System requirements
- **macOS or Linux** (Windows: use WSL2)
- **Python 3.11+**
- **uv** package manager

### Install uv
```bash
curl -Ls https://astral.sh/uv/install.sh | sh
# Restart your terminal after installing
```

### Install system dependencies

**macOS:**
```bash
brew install tesseract poppler
```

**Ubuntu / Debian:**
```bash
sudo apt-get install -y tesseract-ocr poppler-utils libmagic1
```

> `tesseract` = OCR engine (needed by unstructured for scanned PDFs)
> `poppler` = PDF rendering tools (needed for page-to-image conversion)

---

## 🔑 Step 1 — Create Free Service Accounts

You need accounts on 5 free services. Each takes about 2–3 minutes.

### 1. Groq (LLM API)
1. Go to [console.groq.com](https://console.groq.com)
2. Sign up with Google or email
3. Click **API Keys** → **Create API Key**
4. Copy the key (starts with `gsk_`)

### 2. Tavily (Web Search)
1. Go to [app.tavily.com](https://app.tavily.com)
2. Sign up → Dashboard → **API Keys**
3. Copy the key (starts with `tvly-`)

### 3. Qdrant Cloud (Vector Database)
1. Go to [cloud.qdrant.io](https://cloud.qdrant.io)
2. Sign up → **Create Cluster** → choose **Free tier**
3. Wait ~1 minute for the cluster to start
4. Go to **API Keys** → Create a key
5. Copy both the **Cluster URL** and the **API Key**

### 4. MongoDB Atlas (Document Store)
1. Go to [cloud.mongodb.com](https://cloud.mongodb.com)
2. Sign up → **Build a Database** → choose **M0 Free**
3. Choose a cloud provider and region → Create
4. **Security** → Add your IP address (or allow all: `0.0.0.0/0` for local dev)
5. **Connect** → **Drivers** → Python → Copy the connection URI
6. Replace `<password>` in the URI with your actual password

### 5. Upstash Redis (Memory)
1. Go to [console.upstash.com](https://console.upstash.com)
2. Sign up → **Create Database** → Free tier
3. Click your database → **REST API** tab
4. Copy the **UPSTASH_REDIS_REST_URL** and **UPSTASH_REDIS_REST_TOKEN**

---

## 🛠️ Step 2 — Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/BipinMandal/rag-langgraph-chatbot.git
cd rag-langgraph-chatbot

# 2. Create the virtual environment and install all dependencies
uv sync

# 3. Copy the environment template
cp .env.example .env

# 4. Open .env and fill in all 7 keys from Step 1
nano .env        # or: code .env  /  vim .env  /  open in any text editor
```

Your `.env` should look like:
```bash
GROQ_API_KEY=gsk_abc123...
TAVILY_API_KEY=tvly-abc123...
QDRANT_URL=https://abc123.us-east4-0.gcp.cloud.qdrant.io
QDRANT_API_KEY=abc123...
MONGODB_URI=mongodb+srv://user:pass@cluster0.abc.mongodb.net/...
UPSTASH_REDIS_URL=https://abc123.upstash.io
UPSTASH_REDIS_TOKEN=AXabc123...
```

---

## 📥 Step 3 — Download the Test Document

```bash
curl -L "https://arxiv.org/pdf/1706.03762.pdf" -o attention_is_all_you_need.pdf
```

This is the "Attention Is All You Need" paper — 15 pages, contains text,
tables, figures, equations, and cross-references. Perfect for testing all
RAG components.

---

## 🔄 Step 4 — Run the Ingestion Pipeline

> ⚠️ Run this **once only** before starting the chatbot.
> It is safe to re-run — all operations are upserts (no duplicates created).

```bash
uv run python ingest_pipeline.py --pdf attention_is_all_you_need.pdf
```

**What happens during ingestion:**

```
[Step 1/5] Parsing PDF …
  → unstructured extracts ~300 elements (text, titles, tables, images, equations)
  → PyMuPDF extracts raw image bytes from each page
  → Cross-references detected ("see Figure 1", "refer to Section 3.2")

[Step 2/5] Captioning images with Groq vision model …
  → Each figure gets a detailed text description via llama-4-scout
  → Figure 1 (Transformer architecture) → rich technical description

[Step 3/5] Building parent-child chunks …
  → ~87 parent chunks (1200 tokens each) → stored in MongoDB
  → ~312 child chunks (300 tokens each) → stored in Qdrant
  → Images/tables/equations → one unsplittable composite chunk each

[Step 4/5] Storing parent chunks in MongoDB Atlas …
  → Bulk upsert with parent_id index

[Step 5/5] Indexing child chunks in Qdrant Cloud …
  → Dense embeddings: all-MiniLM-L6-v2 (384-dim)
  → Sparse embeddings: BM25 via fastembed
  → First run downloads ~90 MB of embedding models (cached after)

✅  Ingestion complete!
   Parents in MongoDB : 87
   Children in Qdrant : 312
```

**Expected time:** 5–15 minutes (most time is image captioning via API)

---

## 💬 Step 5 — Run the Chatbot

### Option A — Streamlit UI (recommended)

Open **two terminals**:

**Terminal 1 — Start the API backend:**
```bash
uv run uvicorn api:app --reload --port 8000
```

**Terminal 2 — Start the UI:**
```bash
uv run streamlit run ui.py
```

Open **http://localhost:8501** in your browser.

---

### Option B — CLI (no UI)

```bash
uv run python main.py --user your_name
```

---

## 🗺️ Understanding the Pipeline — Step by Step

This section explains exactly what happens when you type a question.

### Step 1 — Memory Loading
Before anything else, the chatbot loads:
- **Short-term memory**: your last few messages this session (from Redis)
- **Long-term memory**: a bullet-point summary of your previous sessions (from Redis)

This means if you asked about attention yesterday, the chatbot remembers that context today — without reloading thousands of tokens of history.

---

### Step 2 — Routing
Your question is classified by a fast LLM (`llama-3.1-8b-instant`):

```
"How does multi-head attention work?"  →  rag         (answer from paper)
"Who won the FIFA World Cup 2022?"     →  web_search  (not in the paper)
```

Only one API call, very fast. Defaults to `rag` on any ambiguity.

---

### Step 3 — Query Rewriting (RAG path only)
The raw query is improved for vector search:

```
Before: "what's the dk thing in attention?"
After:  "What is the scaling factor √dk used for in scaled dot-product attention?"
```

Abbreviations are expanded, technical terms are added.

---

### Step 4 — Hybrid Retrieval
Two searches run simultaneously in Qdrant:
- **Dense search**: finds semantically similar content (even with different words)
- **Sparse / BM25 search**: finds exact keyword matches

Results are merged using **RRF (Reciprocal Rank Fusion)** — documents ranking
high in BOTH searches score highest.

The top-10 child chunks (300 tokens) are returned.
Their `parent_id` values are deduplicated → top-5 unique parents fetched
from MongoDB (1200 tokens each) for the LLM.

---

### Step 5 — Grading
Each parent document is scored 0.0–1.0 for relevance by the fast LLM.
Documents below `0.6` threshold are filtered out.

Cross-reference boost: if your query mentions "Figure 1" and a document
contains "as shown in Figure 1", it gets a +0.05 score bonus.

If fewer than 1 document passes: **Corrective RAG** triggers — Tavily web
search supplements the results.

---

### Step 6 — Generation
The large model (`llama-3.3-70b-versatile`) receives:
- System prompt with citation instructions
- Long-term memory context
- Last 3 chat turns
- Numbered context documents
- Your question

It generates an answer with explicit citations like:
> "As described in [Doc 2] (Section 3.2, Page 4), the attention function
> maps queries and keys via a scaled dot-product..."

---

### Step 7 — Validation
The fast LLM checks the generated answer for:
- **Hallucination**: claims not found in the context documents
- **Irrelevance**: answer doesn't address the question

If the answer fails: the pipeline retries (max 2 times) injecting the
specific failure reason into the prompt so the LLM knows what to fix.

---

### Step 8 — Memory Saving
The Q&A turn is appended to Redis session history.

When you end the session (type `exit` or click "Save & New" in UI):
the LLM summarises the entire session into bullet points and merges
them with your existing long-term memory. Next session, only the
compact summary is injected — not thousands of tokens of raw history.

---

## 🧪 Example Queries to Test

| Query | What it tests |
|---|---|
| `How does multi-head attention work?` | Core RAG retrieval — Section 3.2 |
| `What does Figure 1 show?` | Image citation rendering |
| `Compare model sizes in Table 2` | Table citation rendering |
| `What is the scaled dot-product attention formula?` | Equation retrieval |
| `What BLEU score did the big transformer achieve on WMT 2014?` | Table data extraction |
| `Why did the authors use positional encoding instead of RNN?` | Multi-section reasoning |
| `See Section 5.4 — what did they find about attention head behaviour?` | Section-filtered retrieval |
| `Who invented the transformer?` | Factual question → web search fallback |
| `What is the capital of France?` | Out-of-domain → web search routing |

---

## ⚙️ Configuration Reference

All tuning parameters are in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `PARENT_CHUNK_SIZE` | 1200 | Tokens per parent chunk (MongoDB) |
| `CHILD_CHUNK_SIZE` | 300 | Tokens per child chunk (Qdrant) |
| `TOP_K_CHILD` | 10 | Child chunks retrieved per query |
| `TOP_K_FINAL` | 5 | Unique parent docs passed to LLM |
| `RELEVANCE_THRESHOLD` | 0.6 | Min grade score to keep a doc |
| `MIN_GRADED_DOCS` | 1 | Min docs needed before corrective RAG |
| `MAX_VALIDATOR_RETRIES` | 2 | Max generation retries on validation fail |
| `MEMORY_SESSION_TTL` | 3600 | Session history TTL in seconds (1 hour) |
| `MEMORY_LONGTERM_TTL` | 7,776,000 | Long-term memory TTL in seconds (90 days) |

---

## 🐛 Troubleshooting

### `ImportError: cannot import name 'X' from 'agents.Y'`
The file wasn't saved correctly. Re-copy the file content and save.

### `EnvironmentError: Required environment variable 'X' is not set`
Your `.env` file is missing a key. Open `.env` and fill in the missing value.

### `uv run ingest — error: Failed to spawn`
Run directly instead: `uv run python ingest_pipeline.py --pdf paper.pdf`

### `uv run uvicorn — ModuleNotFoundError`
Run `uv sync` first to install all dependencies.

### Qdrant connection refused
Check your `QDRANT_URL` — it must include `https://` and end without a trailing slash.

### MongoDB `Authentication failed`
The password in your `MONGODB_URI` may contain special characters.
URL-encode them: `@` → `%40`, `#` → `%23`, etc.

### Images not showing in UI
Images are saved to `extracted_images/` during ingestion.
Make sure this folder exists and the API is running from the project root directory.

### Ingestion takes too long
Image captioning is the slow step (one Groq API call per image).
The "Attention" paper has ~10 figures. Expected time: 5–10 minutes.

---

## 📦 All Dependencies

Managed by `uv` via `pyproject.toml`. Install with `uv sync`.

```
langchain                  LLM framework
langchain-groq             Groq LLM integration
langchain-community        Community integrations
langchain-core             Base abstractions
langchain-text-splitters   RecursiveCharacterTextSplitter + MarkdownHeaderTextSplitter
langgraph                  Multi-agent stateful graph
unstructured[pdf,image]    PDF parsing (text, tables, images)
pdfplumber                 Table extraction fallback
pymupdf                    Raw image byte extraction
pytesseract                OCR for scanned pages
pillow                     Image manipulation
beautifulsoup4             HTML table → Markdown
qdrant-client              Qdrant Cloud client
fastembed                  BM25 sparse embeddings (local)
sentence-transformers      Dense embeddings all-MiniLM-L6-v2 (local)
pymongo                    MongoDB Atlas client
upstash-redis              Serverless Redis client
tavily-python              Tavily web search
python-dotenv              .env file loader
tiktoken                   Token counting
numpy                      Numerical operations
streamlit                  Chat UI
fastapi                    REST API backend
uvicorn[standard]          ASGI server
httpx                      Async HTTP client (UI → API calls)
```

---

## 🔒 Security Notes

- **Never commit `.env`** — it's in `.gitignore`
- **Never share your API keys** publicly
- For MongoDB Atlas, whitelist only your IP in production (not `0.0.0.0/0`)
- The `extracted_images/` folder contains your document's images — add to `.gitignore`

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙏 Acknowledgements

- [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762) — "Attention Is All You Need"
- [LangChain](https://langchain.com) / [LangGraph](https://langchain-ai.github.io/langgraph/)
- [Groq](https://groq.com) for blazing-fast free LLM inference
- [Qdrant](https://qdrant.tech) for the free vector database
- [Unstructured](https://unstructured.io) for PDF parsing
