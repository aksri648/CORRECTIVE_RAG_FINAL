# Corrective RAG Project

A self-correcting Retrieval-Augmented Generation (RAG) agent built with **LangGraph**, **Chroma Cloud** (online vector DB), an open-source **HuggingFace** embedding model, and a custom Kimchi OpenAI-compatible LLM endpoint. The agent first retrieves from a vector store, grades the chunks for relevance, and falls back to a **Tavily** web search (with a query rewrite) when nothing is relevant. The result is optionally cross-validated against fresh web results before being shown to the user, and every user message is screened by a two-stage guardrail (regex + optional LLM classifier). Exposed through a **Streamlit** chat UI.

---

## Table of contents
1. [Architecture](#architecture)
2. [Folder layout](#folder-layout)
3. [Setup](#setup)
4. [Running the app](#running-the-app)
5. [Pipeline walkthrough](#pipeline-walkthrough)
6. [Guardrails](#guardrails)
7. [Validator cross-check](#validator-cross-check)
8. [Configuration](#configuration)
9. [Streamlit UI](#streamlit-ui)
10. [Troubleshooting](#troubleshooting)

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
   get_model    retrieve from   guardrails
   (Kimchi      Chroma Cloud    (regex + opt.
    LLM)        (BGE embeds)     ML classifier)
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
   + optional Tavily              |
     cross-validator              v
                              generate answer
                              (no cross-check;
                               web is the source)
```

Four external services are used:
- **Chroma Cloud** — hosted vector DB that stores embedded chunks of the knowledge base.
- **HuggingFace sentence-transformers** — open-source embedding model (`BAAI/bge-small-en-v1.5`) run locally; vectors are uploaded to Chroma Cloud.
- **Kimchi API** — OpenAI-compatible LLM endpoint (`minimax-m2.7`) used for grading, query rewriting, answer generation, the validator, and (optionally) the second-pass safety classifier.
- **Tavily** — web search fallback when retrieved chunks are not relevant, and cross-validation source for the optional validator.

---

## Folder layout

```
corrective rag project/
├── app.py                       # Streamlit frontend (chat UI, sidebar, history)
├── agent_graph.py               # LangGraph CRAG pipeline (nodes, edges, state)
├── vector_store.py              # Chroma Cloud bootstrap + HuggingFace embeddings
├── prompts.py                   # All system prompts (grader, rewriter, validator, classifier, etc.)
├── guardrails.py                # Two-stage input guardrail (regex + optional LLM classifier)
├── ocr.py                       # PDF loading with auto-detected text vs. image-based fallback
├── run.py                       # pyngrok launcher (local public URL)
├── requirements.txt             # Python dependencies
├── .env.example                 # Template for required environment variables
└── project-documentation.md     # This file
```

### File responsibilities

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit entrypoint. Renders the sidebar (env status, models), PDF upload/index controls (with optional EasyOCR), the question form, per-turn answers, the workflow trace, and the graded/retrieved documents. |
| `agent_graph.py` | Defines `SharedState` and the LangGraph nodes: `get_model`, `get_relevant_documents`, `grade_and_filter_documents`, `decide_to_generate` (conditional edge), `transform_query`, `perform_web_search`, `validate_response`, `generate_answer_from_documents`. Uses a tolerant yes/no parser that strips `<think>...</think>` blocks. |
| `vector_store.py` | Builds a `chromadb.CloudClient`, indexes uploaded PDFs into the `rag-pdf-chroma` collection with BGE embeddings, exposes existing retrievers, and handles collection reset. |
| `ocr.py` | Loads PDFs with auto-detected text vs. image-based fallback, and OCRs scanned pages with EasyOCR. Caches the EasyOCR reader for performance. |
| `guardrails.py` | Two-stage input guardrail. Stage 1: 40+ keywords + 30+ regex patterns block obvious prompt injections, plus 14-category PII redaction. Stage 2: an optional LLM-based second-pass classifier (variant B, only fires on borderline inputs). |
| `prompts.py` | All system prompts: `GRADE_DOCUMENTS_PROMPT`, `WEB_SEARCH_QUESTION_REWRITER_PROMPT`, `WEB_SEARCH_ANSWER_PROMPT`, `WEB_SEARCH_RAG_PROMPT`, `RAG_PROMPT`, `VALIDATOR_PROMPT`, `CLASSIFIER_PROMPT`. |
| `run.py` | Launches Streamlit headless on `STREAMLIT_PORT` and opens an ngrok tunnel to it, printing the public URL. |
| `requirements.txt` | All Python packages required to run the app. |
| `.env.example` | Template for the API keys and tunables. |

---

## Setup

### 1. Clone and enter the project

```bash
git clone https://github.com/aksri648/CORRECTIVE_RAG_FINAL.git
cd CORRECTIVE_RAG_FINAL
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

The first run will download `BAAI/bge-small-en-v1.5` (~33 MB) into the HuggingFace cache. If you plan to use OCR, also install the system dependency `poppler-utils` (e.g. `apt-get install -y poppler-utils`) — `pdf2image` requires it. EasyOCR will download its English model (~100 MB) the first time OCR is triggered.

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in the values described in [Configuration](#configuration). At minimum you need `KIMCHI_API_KEY`, `TAVILY_API_KEY`, `CHROMA_API_KEY`, `CHROMA_TENANT`, and `CHROMA_DATABASE`.

---

## Running the app

### Local only

```bash
streamlit run app.py
```

Streamlit will open the UI at `http://localhost:8501`. The normal flow is:

1. Upload one or more PDFs.
2. Click **Index uploaded PDFs**.
3. The app splits the PDF text into chunks, embeds them with BGE, and uploads vectors to Chroma Cloud collection `rag-pdf-chroma`.
4. Ask questions against the indexed PDF knowledge base.

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

### Running on Google Colab (free CPU backend)

Open the **`CorrectiveRag_with_GuardRails.ipynb`** notebook at the repo root directly in Google Colab:

> https://colab.research.google.com/github/aksri648/CORRECTIVE_RAG_FINAL/blob/main/CorrectiveRag_with_GuardRails.ipynb

**One-time setup in Colab:**
1. In the left sidebar, click the **🔑 Secrets** icon and add six secrets with exactly these names:
   - `KIMCHI_API_KEY`
   - `TAVILY_API_KEY`
   - `CHROMA_API_KEY`
   - `CHROMA_TENANT`
   - `CHROMA_DATABASE`
   - `NGROK_AUTH_TOKEN`
2. Run the three code cells in order.
3. The final cell prints a public `https://*.ngrok-free.app` URL — open it in any browser. The URL is reachable from anywhere while the Colab runtime stays connected.
4. Interrupt the cell (▶️ ■) to stop both Streamlit and the ngrok tunnel cleanly.

The notebook:
- Installs `poppler-utils` (system dep for OCR) and the project's Python requirements.
- Clones this repo into the Colab VM (skipped on re-runs).
- Loads the six secrets via `google.colab.userdata`, writes them to `.env`, and runs `run.py`.

**Why this works on free Colab:** Chroma Cloud stores vectors remotely, BGE-small is small enough to embed on Colab CPU in seconds, and the Kimchi LLM endpoint is called over HTTPS so the only thing running on the VM is the Streamlit process and the ngrok tunnel.

---

## Pipeline walkthrough

The LangGraph workflow in `agent_graph.py` is:

| Step | Node | Description |
| --- | --- | --- |
| 1 | `get_model` | Initializes a `ChatOpenAI` pointed at the Kimchi base URL with the configured LLM. |
| 2 | `get_relevant_documents` | Calls the Chroma Cloud retriever with the user's question. |
| 3 | `grade_and_filter_documents` | Uses the LLM to return `yes`/`no`, then a robust parser strips `<think>...</think>` and extracts the grade without strict JSON parsing. Only relevant chunks are kept. |
| 4 | `decide_to_generate` | Conditional edge. If at least one chunk is relevant → `generate`. Otherwise → `transform_query`. |
| 5 | `transform_query` | Rewrites the question to be more web-search-friendly (uses the strict `WEB_SEARCH_QUESTION_REWRITER_PROMPT` and cleans the result with `_clean_rewritten_query`). |
| 6 | `perform_web_search` | Calls `TavilySearch` (`max_results=5`, `search_depth="advanced"`, `include_answer="advanced"`) and uses the synthesized `answer` prepended to the snippets as new context. |
| 7 | `generate_answer_from_documents` | Picks the prompt by path: `WEB_SEARCH_RAG_PROMPT` when `used_web_search=True`, otherwise `RAG_PROMPT`. Always uses the **original** question, not the rewritten query, so the LLM responds to the user's actual question. |
| 8 | `validate_response` | (Optional, controlled by the UI checkbox) cross-checks the RAG answer against fresh Tavily results via LLM-as-judge. Rejects and triggers a second `perform_web_search` if inconsistent, or if the RAG answer signals uncertainty. |
| 9 | second `perform_web_search` (validator reject) | Skips `transform_query` and uses the original question. Sets `used_web_search=True` so the validator auto-accepts the final answer. |

Every node appends a human-readable line to `state["trace"]`, which the Streamlit UI displays in the "Workflow trace" expander.

---

## Guardrails

Two layers run **before** the LangGraph pipeline, in `guardrails.py`:

### Stage 1 — regex (always on)

- **Prompt-injection detection** in `detect_prompt_injection`:
  - 40+ literal keywords (e.g. `ignore previous instructions`, `system prompt`, `forget everything`, `DAN`, `god mode`).
  - 30+ regex patterns for paraphrased / roleplay / mode-switch attacks (e.g. `act as`, `pretend to be`, `simulate being`, `end of prompt`, `disregard all/any safety`).
  - When any keyword or pattern matches, the request is blocked with the canned response `"You are not allowed to change my property"` and never reaches the LLM.
- **PII redaction** in `strip_pii`:
  - 14 categories: email, US/international phone, SSN, credit card, IP, Aadhaar, PAN, date of birth, IBAN, passport, API keys, US zip, IPv6.
  - Replaces matches with `[TYPE_REDACTED]` placeholders before the LLM call. The UI surfaces which PII types were stripped in an info banner with a side-by-side original vs. sanitized view.

### Stage 2 — LLM-based second-pass classifier (opt-in)

Catches paraphrased or obfuscated prompt injections that slip past the regex.

- Uses the same Kimchi LLM (`minimax-m2.7`) with a dedicated `CLASSIFIER_PROMPT` that asks the model to output only `safe` or `unsafe`.
- **Trigger (variant B — suspicions only):** the input is only sent to the classifier if it is borderline. The trigger heuristic `looks_suspicious` returns `True` when the input is long (>250 chars), contains soft-probing vocabulary (e.g. `your instructions`, `system prompt`, `please ignore`, `can you share ... rules`), code blocks, long ellipsis, or hidden Unicode / control characters.
- **Fails OPEN:** if `KIMCHI_API_KEY` is missing or the classifier call errors out, the input is treated as safe so the app does not silently break.
- **Verdict is hidden from end users** — only the generic "Prompt-injection guardrail triggered" trace is shown, so the user cannot tell regex from classifier triggered the block.
- **Default OFF** in the UI; the user enables it per-question with a checkbox labeled "Use ML-based safety classifier for borderline inputs". When OFF, no extra LLM call is made.

---

## Validator cross-check

The `validate_response` node (in `agent_graph.py`) is a self-correction loop:

- Controlled per question by the **"Cross-validate RAG answer with Tavily"** checkbox (default **ON**).
- Skipped entirely when `used_web_search=True` (web is already the source — no point re-validating).
- Disabled when the checkbox is OFF (a no-op pass-through).
- Pre-checks the RAG answer with `_looks_like_uncertainty` (15 regexes for hedging language like "I don't know", "I'm not sure", "the documents do not mention") to skip the Tavily call when the RAG answer is already uncertain.
- Otherwise runs a fast Tavily cross-check (`max_results=3`, `search_depth="basic"`) and asks the LLM (via `VALIDATOR_PROMPT`) whether the RAG answer is consistent with the fresh web snippets.
- If the validator says `no` (inconsistent), the answer is rejected and routed back to `perform_web_search` (using the original question, skipping the rewrite step) for a second generation. This second time, `used_web_search=True` is set, so the validator auto-accepts the final answer.
- The UI surfaces the validation status in a colored banner with one of 7 states: `consistent`, `inconsistent`, `uncertainty`, `search_failed`, `no_web_data`, `skipped_web`, `disabled`.

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
| `CHROMA_COLLECTION_NAME` | `rag-pdf-chroma` | Chroma Cloud collection used for uploaded PDF chunks. |
| `NGROK_AUTH_TOKEN` | — (optional, for public URL) | ngrok authtoken used by `run.py`. |
| `NGROK_DOMAIN` | _empty_ | Optional reserved ngrok domain (paid plan). |
| `STREAMLIT_PORT` | `8501` | Local port Streamlit binds to; the ngrok tunnel points here. |

> **Important:** the embedding model used at *index* time must match the one used at *query* time. If you change `EMBEDDING_MODEL`, reset the Chroma Cloud collection from the sidebar so vectors are re-embedded with the new model.

---

## Streamlit UI

The UI has four regions:

1. **Sidebar** — environment variable status (which keys are present), the active LLM / embedding model, the active Chroma collection, and a "Reset Chroma Cloud collection" button.
2. **PDF indexer** — upload one or more PDFs, optionally check **Force OCR with EasyOCR** (for scanned / image-based PDFs), click "Index uploaded PDFs", or load an existing Chroma collection.
3. **Question form** — a single text input with two checkboxes and an "Ask the agent" submit button:
   - **Cross-validate RAG answer with Tavily** (default ON) — enables the validator node.
   - **Use ML-based safety classifier for borderline inputs** (default OFF) — enables the second-pass LLM classifier.
4. **History** — every submitted question is appended to session history. Each turn shows the final answer (with `<think>...</think>` blocks stripped), source path (Chroma or Tavily), validation status banner, PII-redaction notice (if any), workflow trace, graded documents, and retrieved context chunks. Blocked-by-guardrail turns show the canned response and an expandable "What triggered the block" trace.

---

## Troubleshooting

- **`Failed to initialize vector store`** — make sure `CHROMA_API_KEY`, `CHROMA_TENANT`, and `CHROMA_DATABASE` are all set. The check in the sidebar will mark missing ones with an `x`.
- **Pydantic / invalid JSON grader errors** — fixed by using a text-only yes/no grader parser instead of strict structured output. Pull the latest code if you still see this.
- **First run is slow** — the embedding model (~33 MB) and the EasyOCR English model (~100 MB) are downloaded the first time they are used. Subsequent runs reuse the cache.
- **Stale vectors after changing the embedding model** — open the sidebar and click "Reset Chroma Cloud collection", then upload and index PDFs again.
- **`TAVILY_API_KEY` missing** — the app will still answer from the vector store, but the web-search fallback and the validator will fail. Set the key in `.env` for full corrective behavior.
- **OCR errors / empty chunks** — install `poppler-utils` on the host (`apt-get install -y poppler-utils` on Debian/Ubuntu, included in the Colab notebook). EasyOCR downloads its model the first time it is used; subsequent runs reuse the cache.
- **Classifier always returns `skipped_no_api_key`** — you need to set `KIMCHI_API_KEY` in `.env` for the second-pass ML classifier to run. The classifier fails OPEN, so the app continues to work without it.
- **`minimax-m2.7` emits `<think>...</think>` blocks** — this is expected. The grader, query rewriter, and validator all use tolerant parsers that strip these blocks before extracting the answer.
- **Colab URL stops working** — free Colab runtimes disconnect after ~90 minutes of inactivity or ~12 hours total. Re-run the final cell of the notebook to get a fresh public URL.
