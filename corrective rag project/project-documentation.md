# Corrective RAG Project

A self-correcting Retrieval-Augmented Generation (RAG) agent built with **LangGraph**, **Chroma Cloud** (online vector DB), an open-source **HuggingFace** embedding model, and a custom Kimchi OpenAI-compatible LLM endpoint. The agent first retrieves from a vector store, grades the chunks for relevance, and falls back to a **Tavily** web search (with a query rewrite) when nothing is relevant. Exposed through a **Streamlit** chat UI.

---

## Table of contents
1. [Architecture](#architecture)
2. [Folder layout](#folder-layout)
3. [Setup](#setup)
4. [Running the app](#running-the-app)
5. [Pipeline walkthrough](#pipeline-walkthrough)
6. [Configuration](#configuration)
7. [Streamlit UI](#streamlit-ui)
8. [Troubleshooting](#troubleshooting)

---

## Architecture

```
            +-------------------+
   user -->|  Streamlit (app)  |
            +---------+---------+
                      |
                      v
            +-------------------+
            |  LangGraph agent  |
            |  (agent_graph.py) |
            +---------+---------+
                      |
        +-------------+-------------+
        |             |             |
        v             v             v
   get_model    retrieve from   generate
   (Kimchi      Chroma Cloud    answer
    LLM)        (BGE embeds)
                      |
                      v
                grade documents
                (LLM-as-judge)
                      |
        +-------------+-------------+
        |                           |
        v                           v
   relevant >= 1              relevant == 0
        |                           |
        v                           v
   generate answer        rewrite query + Tavily
                                   |
                                   v
                            generate answer
```

Two storage/AI services are used:
- **Chroma Cloud** — hosted vector DB that stores embedded chunks of the knowledge base.
- **HuggingFace sentence-transformers** — open-source embedding model (`BAAI/bge-small-en-v1.5`) run locally; vectors are uploaded to Chroma Cloud.
- **Kimchi API** — OpenAI-compatible LLM endpoint (`minimax-m2.7`) used for grading, query rewriting, and final answer generation.
- **Tavily** — web search fallback when retrieved chunks are not relevant.

---

## Folder layout

```
corrective rag project/
├── app.py              # Streamlit frontend (chat UI, sidebar, history)
├── agent_graph.py      # LangGraph CRAG pipeline (nodes, edges, state)
├── vector_store.py     # Chroma Cloud bootstrap + HuggingFace embeddings
├── prompts.py          # Grader and query-rewriter system prompts
├── requirements.txt    # Python dependencies
├── .env.example        # Template for required environment variables
└── project-documentation.md
```

### File responsibilities

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit entrypoint. Renders the sidebar (env status, KB URLs, models), the question form, per-turn answers, the workflow trace, and the graded/retrieved documents. |
| `agent_graph.py` | Defines `SharedState`, the `GradeDocuments` pydantic model, and seven LangGraph nodes: `get_model`, `get_relevant_documents`, `grade_and_filter_documents`, `decide_to_generate` (conditional edge), `transform_query`, `perform_web_search`, `generate_answer_from_documents`. |
| `vector_store.py` | Builds a `chromadb.CloudClient`, creates the `rag-chroma` collection on first run by embedding the two knowledge-base URLs, and exposes a cached retriever (`@st.cache_resource`). Also handles collection reset. |
| `guardrails.py` | Two-stage input guardrail: regex-based prompt-injection detection (40+ keywords + 30+ patterns) and PII redaction (14 categories) applied before any LLM call. |
| `run.py` | Launches Streamlit headless on `STREAMLIT_PORT` and opens an ngrok tunnel to it, printing the public URL. |
| `prompts.py` | Two system prompts — `GRADE_DOCUMENTS_PROMPT` and `QUESTION_REWRITER_PROMPT`. |
| `requirements.txt` | All Python packages required to run the app. |
| `.env.example` | Template for the API keys and tunables. |

---

## Setup

### 1. Clone and enter the project

```bash
cd "corrective rag project"
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The first run will download `BAAI/bge-small-en-v1.5` (~33 MB) into the HuggingFace cache.

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in the values described in [Configuration](#configuration).

---

## Running the app

### Local only

```bash
streamlit run app.py
```

Streamlit will open the UI at `http://localhost:8501`. The first time you submit a question, the app will:

1. Connect to Chroma Cloud.
2. If the `rag-chroma` collection does not exist, load the two knowledge-base URLs, split them into ~250-token chunks, embed them with BGE, and upload the vectors to Chroma Cloud.
3. Cache the retriever for the rest of the session.

### Public URL via ngrok (pyngrok)

Set `NGROK_AUTH_TOKEN` in `.env` (get one free at https://dashboard.ngrok.com/get-started/your-authtoken) and run:

```bash
python run.py
```

The script prints something like:

```
================================================================
  Public URL:  https://<random>.ngrok-free.app
  Local URL:   http://localhost:8501
================================================================
  Press Ctrl+C to stop both Streamlit and the ngrok tunnel.
```

Optional env vars:
- `NGROK_DOMAIN` — pin to a reserved static domain (paid plan).
- `STREAMLIT_PORT` — change the local port (default `8501`).

On `Ctrl+C` the script terminates Streamlit and disconnects the ngrok tunnel cleanly.

---

## Pipeline walkthrough

The LangGraph workflow in `agent_graph.py` is:

| Step | Node | Description |
| --- | --- | --- |
| 1 | `get_model` | Initializes a `ChatOpenAI` pointed at the Kimchi base URL with the configured LLM. |
| 2 | `get_relevant_documents` | Calls the Chroma Cloud retriever with the user's question. |
| 3 | `grade_and_filter_documents` | Uses the LLM (with `with_structured_output(GradeDocuments)`) to grade every chunk as `yes`/`no`. Only relevant chunks are kept. |
| 4 | `decide_to_generate` | Conditional edge. If at least one chunk is relevant → `generate`. Otherwise → `transform_query`. |
| 5 | `transform_query` | Rewrites the question to be more web-search-friendly. |
| 6 | `perform_web_search` | Calls `TavilySearch` and uses the returned snippets as new context. |
| 7 | `generate_answer_from_documents` | Pulls the `rlm/rag-prompt` from LangChain Hub and generates the final answer with the kept context. |

Every node appends a human-readable line to `state["trace"]`, which the Streamlit UI displays in the "Workflow trace" expander.

---

## Configuration

All settings are read from environment variables (loaded by `python-dotenv`).

| Variable | Default | Description |
| --- | --- | --- |
| `KIMCHI_API_KEY` | — (required) | API key for the Kimchi OpenAI-compatible endpoint. |
| `KIMCHI_BASE_URL` | `https://llm.kimchi.dev/openai/v1` | Base URL for the LLM API. |
| `LLM_MODEL` | `minimax-m2.7` | Model name sent to the API. |
| `LLM_TEMPERATURE` | `0` | LLM sampling temperature. |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Any sentence-transformers / HuggingFace embedding model. Try `BAAI/bge-large-en-v1.5` for higher quality or `intfloat/e5-small-v2` for speed. |
| `EMBEDDING_DEVICE` | `cpu` | Set to `cuda` for GPU acceleration. |
| `TAVILY_API_KEY` | — (required for fallback) | Tavily search API key. |
| `CHROMA_API_KEY` | — (required) | Chroma Cloud API key. |
| `CHROMA_TENANT` | — (required) | Chroma Cloud tenant ID. |
| `CHROMA_DATABASE` | — (required) | Chroma Cloud database name. |
| `NGROK_AUTH_TOKEN` | — (optional, for public URL) | ngrok authtoken used by `run.py`. |
| `NGROK_DOMAIN` | _empty_ | Optional reserved ngrok domain (paid plan). |
| `STREAMLIT_PORT` | `8501` | Local port Streamlit binds to; the ngrok tunnel points here. |

> **Important:** the embedding model used at *index* time must match the one used at *query* time. If you change `EMBEDDING_MODEL`, reset the Chroma Cloud collection from the sidebar so vectors are re-embedded with the new model.

---

## Streamlit UI

The UI has three regions:

1. **Sidebar** — environment variable status (which keys are present), the knowledge base URLs, the active LLM / embedding model, and a "Reset Chroma Cloud collection" button.
2. **Question form** — a single text input with an "Ask the agent" submit button.
3. **History** — every submitted question is appended to session history. Each turn shows:
   - The final answer.
   - A banner indicating whether the answer came from Chroma Cloud or the Tavily fallback.
   - The full workflow trace.
   - A list of graded documents (relevant / not relevant).
   - The retrieved context chunks (or web-search snippets).

---

## Troubleshooting

- **`Failed to initialize vector store`** — make sure `CHROMA_API_KEY`, `CHROMA_TENANT`, and `CHROMA_DATABASE` are all set. The check in the sidebar will mark missing ones with an `x`.
- **`with_structured_output` errors** — the Kimchi endpoint must support tool calling for the configured model. The `minimax-m2.7` model in the Kimchi catalog advertises `tool_call: true`, which is what the relevance grader relies on.
- **First run is slow** — the embedding model is downloaded the first time it is used. Subsequent runs are fast.
- **Stale vectors after changing the embedding model** — open the sidebar and click "Reset Chroma Cloud collection", then re-submit a question so the collection is re-created.
- **`TAVILY_API_KEY` missing** — the app will still answer from the vector store, but the web-search fallback will fail. Set the key in `.env` for full corrective behavior.
