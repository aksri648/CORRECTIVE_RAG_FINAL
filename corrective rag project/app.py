import os
import re
from dotenv import load_dotenv
import streamlit as st

from agent_graph import build_graph
from guardrails import apply_guardrails
from vector_store import (
    CHROMA_COLLECTION_NAME,
    cached_build_vector_store,
    index_uploaded_pdfs,
    reset_chroma_cloud_collection,
)


load_dotenv()

st.set_page_config(
    page_title="Corrective RAG Agent",
    page_icon="CR",
    layout="wide",
)


def _check_env() -> dict:
    return {
        "KIMCHI_API_KEY": bool(os.getenv("KIMCHI_API_KEY")),
        "TAVILY_API_KEY": bool(os.getenv("TAVILY_API_KEY")),
        "CHROMA_API_KEY": bool(os.getenv("CHROMA_API_KEY")),
        "CHROMA_TENANT": bool(os.getenv("CHROMA_TENANT")),
        "CHROMA_DATABASE": bool(os.getenv("CHROMA_DATABASE")),
    }


def _render_model_info():
    from agent_graph import LLM_MODEL, KIMCHI_BASE_URL
    from vector_store import EMBEDDING_MODEL

    with st.sidebar:
        st.divider()
        st.header("Models")
        st.markdown(f"**LLM:** `{LLM_MODEL}`")
        st.caption(KIMCHI_BASE_URL)
        st.markdown(f"**Embeddings:** `{EMBEDDING_MODEL}`")


def _render_sidebar():
    with st.sidebar:
        st.header("Environment")

        env = _check_env()
        for key, ok in env.items():
            st.markdown(
                f"{':white_check_mark:' if ok else ':x:'} `{key}`"
            )

        st.divider()
        st.header("Chroma Cloud")
        st.caption(f"Collection: `{CHROMA_COLLECTION_NAME}`")

        if st.button("Reset Chroma Cloud collection", type="secondary"):
            try:
                reset_chroma_cloud_collection()
                cached_build_vector_store.clear()
                st.session_state.retriever = None
                st.session_state.indexed_pdf_names = []
                st.session_state.chunk_count = 0
                st.success("Collection deleted. Upload PDFs and index again.")
            except Exception as exc:
                st.error(f"Could not reset collection: {exc}")

        st.divider()
        st.caption(
            "Pipeline: upload PDFs, index them in Chroma Cloud, retrieve, grade documents, fall back to "
            "Tavily web search if nothing is relevant, then generate an answer."
        )


def _render_pdf_indexer():
    st.markdown("### 1. Upload and index PDFs")
    uploaded_pdfs = st.file_uploader(
        "Upload PDF files to use as your knowledge base",
        type=["pdf"],
        accept_multiple_files=True,
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        index_clicked = st.button("Index uploaded PDFs", type="primary")
        load_existing_clicked = st.button("Load existing Chroma collection", type="secondary")
    with col2:
        if st.session_state.get("indexed_pdf_names"):
            names = ", ".join(st.session_state.indexed_pdf_names)
            chunks = st.session_state.get("chunk_count", 0)
            ocr_used = st.session_state.get("ocr_used", False)
            mode = "OCR" if ocr_used else "text extraction"
            st.success(
                f"Indexed {len(st.session_state.indexed_pdf_names)} PDF(s) "
                f"via {mode}, {chunks} chunks: {names}"
            )
        elif st.session_state.get("retriever") is not None:
            st.success("Loaded existing Chroma Cloud collection.")

    force_ocr = st.checkbox(
        "Force OCR with EasyOCR (for scanned / image-based PDFs — slower)",
        value=False,
    )

    if load_existing_clicked:
        retriever = cached_build_vector_store()
        if retriever is None:
            st.warning("No existing PDF collection found. Upload PDFs and index them first.")
        else:
            st.session_state.retriever = retriever
            st.success("Existing Chroma Cloud collection loaded.")
            st.rerun()

    if not index_clicked:
        return

    if not uploaded_pdfs:
        st.warning("Please upload at least one PDF before indexing.")
        return

    status_box = st.status("Indexing uploaded PDFs...", expanded=True)

    def progress_cb(message: str):
        status_box.update(label=message)

    try:
        retriever, chunk_count = index_uploaded_pdfs(
            uploaded_pdfs,
            force_ocr=force_ocr,
            progress_cb=progress_cb,
        )
        cached_build_vector_store.clear()
        st.session_state.retriever = retriever
        st.session_state.indexed_pdf_names = [file.name for file in uploaded_pdfs]
        st.session_state.chunk_count = chunk_count
        st.session_state.ocr_used = force_ocr
        status_box.update(label="PDF knowledge base indexed", state="complete")
        st.rerun()
    except Exception as exc:
        status_box.update(label="PDF indexing failed", state="error")
        st.error(f"Failed to index PDFs: {exc}")


def _strip_think_blocks(text: str) -> str:
    if not text:
        return text
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    return cleaned.strip()


def _render_trace(trace: list[str]):
    if not trace:
        return
    with st.expander("Workflow trace", expanded=True):
        for i, step in enumerate(trace, start=1):
            st.markdown(f"**{i}.** {step}")


_VALIDATION_LABELS = {
    "consistent": ("Cross-validated against Tavily: consistent.", "success"),
    "inconsistent": ("Cross-validation found a discrepancy; answer was regenerated from web search.", "warning"),
    "uncertainty": ("RAG answer signalled uncertainty; fell back to web search.", "warning"),
    "search_failed": ("Cross-validation web search failed; RAG answer kept.", "info"),
    "no_web_data": ("No web snippets to cross-check; RAG answer kept.", "info"),
    "skipped_web": ("Cross-validation skipped (answer was already from web search).", "info"),
    "disabled": ("Cross-validation disabled for this question.", "info"),
}


def _render_validation_status(status: str, answer: str, used_web: bool):
    if used_web and status not in {"inconsistent", "uncertainty"}:
        return
    label, level = _VALIDATION_LABELS.get(status, (None, None))
    if not label:
        return
    fn = {
        "success": st.success,
        "warning": st.warning,
        "info": st.info,
    }[level]
    fn(label)


def _render_graded_documents(graded: list[dict]):
    if not graded:
        return
    with st.expander(f"Graded documents ({len(graded)})", expanded=False):
        for item in graded:
            status = "RELEVANT" if item["relevant"] else "NOT RELEVANT"
            icon = "+" if item["relevant"] else "-"
            st.markdown(f"`[{icon}]` **{status}** — _{item['snippet']}..._")


def _render_used_documents(docs: list[str], title: str):
    if not docs:
        return
    with st.expander(f"{title} ({len(docs)})", expanded=False):
        for i, doc in enumerate(docs, start=1):
            st.markdown(f"**Chunk {i}**")
            st.write(doc)
            st.divider()


def main():
    st.title("Corrective RAG Agent")
    st.markdown(
        "Upload PDFs, then ask a question. The agent first searches your **Chroma Cloud** vector store. "
        "If no chunk is relevant, it rewrites the query and runs a **Tavily** web search "
        "before generating the final answer."
    )

    _render_sidebar()
    _render_model_info()

    if "history" not in st.session_state:
        st.session_state.history = []
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "indexed_pdf_names" not in st.session_state:
        st.session_state.indexed_pdf_names = []
    if "chunk_count" not in st.session_state:
        st.session_state.chunk_count = 0
    if "ocr_used" not in st.session_state:
        st.session_state.ocr_used = False

    _render_pdf_indexer()
    st.markdown("### 2. Ask a question")

    with st.form("question_form", clear_on_submit=False):
        question = st.text_input(
            "Your question",
            placeholder="e.g. What are word embeddings and how do they work?",
        )
        validate_rag_answer = st.checkbox(
            "Cross-validate RAG answer with Tavily (slower but more accurate)",
            value=True,
            help=(
                "When ON, the agent cross-checks the RAG answer against fresh Tavily "
                "results. If they disagree, the answer is replaced with a web-sourced one."
            ),
        )
        submitted = st.form_submit_button("Ask the agent", type="primary")

    if submitted:
        if not question.strip():
            st.warning("Please enter a question.")
            st.stop()

        guardrail_result = apply_guardrails(question)

        if guardrail_result["blocked"]:
            st.session_state.history.append(
                {
                    "question": question,
                    "result": {
                        "agent_response": guardrail_result["response"],
                        "used_web_search": False,
                        "trace": [
                            "Prompt-injection guardrail triggered. Request blocked before reaching the LLM.",
                            f"Matched triggers: {', '.join(guardrail_result['injection_hits'])}",
                        ],
                        "graded_documents": [],
                        "relevant_documents": [],
                    },
                    "blocked": True,
                    "pii_stripped": [],
                }
            )
            st.rerun()

        sanitized_question = guardrail_result["sanitized_question"]
        pii_stripped = guardrail_result["pii_stripped"]

        retriever = st.session_state.get("retriever")
        if retriever is None:
            st.warning("Upload and index at least one PDF before asking questions.")
            st.stop()

        graph = build_graph()

        with st.spinner("Running the corrective RAG pipeline..."):
            result = graph.invoke(
                {
                    "question": sanitized_question,
                    "vector_store": retriever,
                    "trace": [],
                    "validate_rag_answer": validate_rag_answer,
                }
            )

        st.session_state.history.append(
            {
                "question": question,
                "sanitized_question": sanitized_question,
                "pii_stripped": pii_stripped,
                "validate_rag_answer": validate_rag_answer,
                "result": result,
                "blocked": False,
            }
        )

    for turn in reversed(st.session_state.history):
        st.markdown("---")
        st.subheader(f"Q: {turn['question']}")

        if turn.get("blocked"):
            st.error("Blocked by guardrail (prompt injection detected).")
            st.markdown("### Response")
            st.write(turn["result"].get("agent_response", ""))
            with st.expander("What triggered the block", expanded=True):
                for step in turn["result"].get("trace", []):
                    st.markdown(f"- {step}")
            continue

        pii_stripped = turn.get("pii_stripped", [])
        if pii_stripped:
            st.info(f"PII guardrail stripped: {', '.join(pii_stripped)}")
            with st.expander("Original vs. sanitized prompt", expanded=False):
                st.markdown("**Original:**")
                st.code(turn["question"])
                st.markdown("**Sent to LLM:**")
                st.code(turn.get("sanitized_question", turn["question"]))

        result = turn["result"]
        answer = _strip_think_blocks(result.get("agent_response", "No answer was produced."))
        used_web = result.get("used_web_search", False)
        validation_status = result.get("validation_status")

        if used_web:
            st.warning("Answered using the web-search fallback path.")
        else:
            st.success("Answered using the Chroma Cloud knowledge base.")

        if turn.get("validate_rag_answer") and validation_status and not used_web:
            _render_validation_status(validation_status, answer, used_web)

        st.markdown("### Answer")
        st.write(answer)

        st.divider()
        cols = st.columns(2)
        with cols[0]:
            _render_trace(result.get("trace", []))
        with cols[1]:
            _render_graded_documents(result.get("graded_documents", []))

        if used_web:
            _render_used_documents(result.get("relevant_documents", []), "Web search snippets")
        else:
            _render_used_documents(result.get("relevant_documents", []), "Retrieved context chunks")


if __name__ == "__main__":
    main()
