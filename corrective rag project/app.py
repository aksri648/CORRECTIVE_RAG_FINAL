import os
from dotenv import load_dotenv
import streamlit as st

from agent_graph import build_graph
from guardrails import apply_guardrails
from vector_store import (
    KNOWLEDGE_BASE_URLS,
    cached_build_vector_store,
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
        st.header("Knowledge Base")
        for url in KNOWLEDGE_BASE_URLS:
            st.markdown(f"- {url}")

        st.divider()
        st.header("Chroma Cloud")
        st.caption("Collection: `rag-chroma`")

        if st.button("Reset Chroma Cloud collection", type="secondary"):
            try:
                reset_chroma_cloud_collection()
                cached_build_vector_store.clear()
                st.success("Collection deleted. Reload the page to rebuild.")
            except Exception as exc:
                st.error(f"Could not reset collection: {exc}")

        st.divider()
        st.caption(
            "Pipeline: retrieve from Chroma Cloud, grade documents, fall back to "
            "Tavily web search if nothing is relevant, then generate an answer."
        )


def _render_trace(trace: list[str]):
    if not trace:
        return
    with st.expander("Workflow trace", expanded=True):
        for i, step in enumerate(trace, start=1):
            st.markdown(f"**{i}.** {step}")


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
        "Ask a question. The agent first searches a **Chroma Cloud** vector store. "
        "If no chunk is relevant, it rewrites the query and runs a **Tavily** web search "
        "before generating the final answer."
    )

    _render_sidebar()
    _render_model_info()

    if "history" not in st.session_state:
        st.session_state.history = []

    with st.form("question_form", clear_on_submit=False):
        question = st.text_input(
            "Your question",
            placeholder="e.g. What are word embeddings and how do they work?",
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

        retriever = cached_build_vector_store()
        graph = build_graph()

        with st.spinner("Running the corrective RAG pipeline..."):
            result = graph.invoke(
                {
                    "question": sanitized_question,
                    "vector_store": retriever,
                    "trace": [],
                }
            )

        st.session_state.history.append(
            {
                "question": question,
                "sanitized_question": sanitized_question,
                "pii_stripped": pii_stripped,
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
        answer = result.get("agent_response", "No answer was produced.")
        used_web = result.get("used_web_search", False)

        if used_web:
            st.warning("Answered using the web-search fallback path.")
        else:
            st.success("Answered using the Chroma Cloud knowledge base.")

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
