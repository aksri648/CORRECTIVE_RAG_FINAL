# Corrective RAG Agent

A self-correcting Retrieval-Augmented Generation agent with **PDF knowledge base, vector retrieval, web-search fallback, prompt-injection guardrails, and Tavily cross-validation** — built with LangGraph, Chroma Cloud, an open-source BGE embedding model, and a custom Kimchi LLM. Exposed through a Streamlit UI that runs in a browser anywhere via ngrok (or locally).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aksri648/CORRECTIVE_RAG_FINAL/blob/main/CorrectiveRag_with_GuardRails.ipynb)

---

## What it does

You upload one or more PDFs. The agent:

1. **Indexes** them — chunks the text, embeds it with the open-source `BAAI/bge-small-en-v1.5` model, and uploads the vectors to your **Chroma Cloud** collection.
2. **Retrieves** the most relevant chunks when you ask a question.
3. **Grades** the chunks with the LLM as a judge; rejects irrelevant ones.
4. **Falls back to a Tavily web search** (with a query rewrite) when nothing in the knowledge base is relevant.
5. **Cross-validates the final answer** against fresh Tavily results (optional, on by default) and replaces it with a web-sourced answer if the RAG answer is inconsistent or signals uncertainty.
6. **Screens every user message** through a two-stage guardrail before any of the above runs: a regex layer (40+ keywords + 30+ patterns) plus an opt-in LLM second-pass classifier that catches paraphrased prompt injections.
7. **Redacts PII** (14 categories) with `[TYPE_REDACTED]` placeholders before the LLM call.

All of this is wired together in a LangGraph state machine in `corrective rag project/agent_graph.py`.

---

## Quick start (Google Colab — recommended, free)

The fastest way to try it. The notebook installs `poppler-utils` and dependencies, pulls the repo, loads your secrets from Colab Secrets, and exposes the app over a public ngrok URL.

1. Click the **Open in Colab** button at the top of this README.
2. In the Colab sidebar, click **🔑 Secrets** and add six secrets with these exact names:
   - `KIMCHI_API_KEY`
   - `TAVILY_API_KEY`
   - `CHROMA_API_KEY`
   - `CHROMA_TENANT`
   - `CHROMA_DATABASE`
   - `NGROK_AUTH_TOKEN`
3. Run the three code cells in order. The final cell prints a `https://*.ngrok-free.app` URL — open it in any browser.
4. Upload PDFs in the sidebar, click **Index uploaded PDFs**, then ask questions.

## Quick start (local)

```bash
git clone https://github.com/aksri648/CORRECTIVE_RAG_FINAL.git
cd CORRECTIVE_RAG_FINAL/"corrective rag project"
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # then fill in your API keys
# Optional, only if you want OCR on scanned PDFs:
#   Debian/Ubuntu: sudo apt-get install -y poppler-utils
#   macOS:         brew install poppler
streamlit run app.py
```

For a public URL on localhost:

```bash
python run.py             # requires NGROK_AUTH_TOKEN in .env
```

---

## Features

| | |
| --- | --- |
| **PDF knowledge base** | Upload one or more PDFs; auto-detects text vs. scanned and falls back to EasyOCR. |
| **Hosted vector DB** | Vectors live in Chroma Cloud, so no local persistence is required. |
| **Open-source embeddings** | `BAAI/bge-small-en-v1.5` from HuggingFace (CPU by default, swap to any sentence-transformers model). |
| **Custom LLM** | Talks to a Kimchi OpenAI-compatible endpoint (`minimax-m2.7`); tolerant parser handles the model's `<think>...</think>` preamble. |
| **Web-search fallback** | Rewrites the query and runs a Tavily advanced search with synthesized answer prepended. |
| **Self-correcting** | Cross-validates the RAG answer against fresh web results; routes back through the web-search path on disagreement. |
| **Two-stage input guardrail** | Regex layer is always on; LLM second-pass classifier is opt-in (variant B: only fires on borderline inputs). |
| **PII redaction** | 14 categories — email, phone, SSN, credit card, IP, Aadhaar, PAN, DOB, IBAN, passport, API keys, ZIP, IPv6 — replaced with `[TYPE_REDACTED]` placeholders. |
| **Streamlit UI** | Per-question toggles for the validator and the ML classifier; workflow trace, graded documents, retrieved chunks, and validation status all visible. |
| **Public URL** | Free public URL via `pyngrok` in Colab or local. |

---

## Project structure

```
CORRECTIVE_RAG_FINAL/
├── README.md
└── corrective rag project/
    ├── app.py                       # Streamlit frontend
    ├── agent_graph.py               # LangGraph CRAG pipeline
    ├── vector_store.py              # Chroma Cloud + BGE embeddings
    ├── prompts.py                   # All system prompts
    ├── guardrails.py                # Regex + opt-in ML second-pass classifier + PII
    ├── ocr.py                       # PDF loading with EasyOCR fallback
    ├── run.py                       # pyngrok launcher
    ├── requirements.txt
    ├── .env.example
    └── project-documentation.md     # Full architecture + API docs
```

---

## Configuration

All settings are environment variables loaded by `python-dotenv`. See `.env.example` for the full list. The required ones are:

| Variable | Where to get it |
| --- | --- |
| `KIMCHI_API_KEY` | Kimchi OpenAI-compatible endpoint |
| `TAVILY_API_KEY` | [tavily.com](https://tavily.com/) — free tier available |
| `CHROMA_API_KEY`, `CHROMA_TENANT`, `CHROMA_DATABASE` | [trychroma.com](https://www.trychroma.com/) — free tier available |
| `NGROK_AUTH_TOKEN` | [dashboard.ngrok.com](https://dashboard.ngrok.com/get-started/your-authtoken) — free tier available |

Defaults that just work:

- `KIMCHI_BASE_URL=https://llm.kimchi.dev/openai/v1`
- `LLM_MODEL=minimax-m2.7`
- `EMBEDDING_MODEL=BAAI/bge-small-en-v1.5`
- `CHROMA_COLLECTION_NAME=rag-pdf-chroma`
- `STREAMLIT_PORT=8501`

---

## How it works (one-paragraph version)

Every user message first goes through `guardrails.apply_guardrails`: a regex layer (40+ keywords, 30+ patterns) blocks obvious prompt injections, PII is redacted, and — if the user has the "Use ML-based safety classifier" toggle on — borderline inputs (long padding, soft-probing vocabulary, code blocks, hidden Unicode) get a second-pass LLM check that catches paraphrased attacks. The clean message then enters the LangGraph state machine: `get_model → get_relevant_documents → grade_and_filter_documents`. If at least one chunk is relevant, the answer is generated from the local knowledge base. If not, the query is rewritten and a Tavily advanced search is run, and the synthesized answer is fed back to the LLM. The final RAG answer is optionally cross-validated against a fresh Tavily result; if the validator says it's inconsistent (or the answer signals uncertainty), the graph loops back through a web search using the original question. The second-generation answer carries `used_web_search=True` so the validator auto-accepts it. The Streamlit UI surfaces the workflow trace, validation status, PII redactions, graded chunks, and retrieved context for every turn.

---

## Documentation

- **`README.md`** (this file) — quick start, features, configuration, high-level overview.
- **`corrective rag project/project-documentation.md`** — full architecture diagram, file-by-file responsibilities, pipeline walkthrough, guardrails and validator deep-dives, full configuration table, UI tour, and troubleshooting.

---

## License

MIT.
