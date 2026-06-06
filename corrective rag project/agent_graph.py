import os
import re
from typing_extensions import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import END, StateGraph, START

from prompts import (
    GRADE_DOCUMENTS_PROMPT,
    WEB_SEARCH_ANSWER_PROMPT,
    WEB_SEARCH_QUESTION_REWRITER_PROMPT,
)


KIMCHI_BASE_URL = os.getenv("KIMCHI_BASE_URL", "https://llm.kimchi.dev/openai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "minimax-m2.7")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))


RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Use three sentences maximum and keep the answer concise.",
        ),
        (
            "human",
            "Question: {question}\n\nContext:\n{context}\n\nAnswer:",
        ),
    ]
)


WEB_SEARCH_RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", WEB_SEARCH_ANSWER_PROMPT),
        (
            "human",
            "Original user question: {question}\n\n"
            "Web search snippets:\n{context}\n\n"
            "Answer concisely:",
        ),
    ]
)


class SharedState(TypedDict, total=False):
    question: str
    original_question: str
    agent_response: str
    vector_store: object
    relevant_documents: list[str]
    graded_documents: list[dict]
    model: ChatOpenAI
    trace: list[str]
    used_web_search: bool


def get_model(shared_state: SharedState) -> SharedState:
    api_key = os.getenv("KIMCHI_API_KEY")
    if not api_key:
        raise ValueError(
            "KIMCHI_API_KEY is not set. Add it to your .env file."
        )

    shared_state["model"] = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        base_url=KIMCHI_BASE_URL,
        api_key=api_key,
    )
    shared_state.setdefault("trace", []).append(
        f"Initialized chat model '{LLM_MODEL}' via {KIMCHI_BASE_URL}."
    )
    return shared_state


def _parse_relevance_grade(raw_response: str) -> bool:
    cleaned = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.IGNORECASE | re.DOTALL)
    cleaned = cleaned.strip().lower()

    json_match = re.search(r'"binary_score"\s*:\s*"(yes|no)"', cleaned)
    if json_match:
        return json_match.group(1) == "yes"

    score_match = re.search(r"\b(yes|no)\b", cleaned)
    if score_match:
        return score_match.group(1) == "yes"

    return False


def get_relevant_documents(shared_state: SharedState) -> SharedState:
    question = shared_state["question"]
    vector_store = shared_state["vector_store"]

    documents = vector_store.invoke(question)
    shared_state["relevant_documents"] = [doc.page_content for doc in documents]

    shared_state.setdefault("trace", []).append(
        f"Retrieved {len(documents)} candidate documents from Chroma Cloud."
    )
    return shared_state


def grade_and_filter_documents(shared_state: SharedState) -> SharedState:
    question = shared_state["question"]
    model = shared_state["model"]
    documents = shared_state["relevant_documents"]

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                GRADE_DOCUMENTS_PROMPT
                + "\nReturn only one word: yes or no. Do not include reasoning, XML tags, markdown, or JSON.",
            ),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )

    retrieval_grader = grade_prompt | model | StrOutputParser()

    filtered_documents = []
    grade_log = []
    for idx, document in enumerate(documents, start=1):
        raw_grade = retrieval_grader.invoke(
            {"question": question, "document": document}
        )
        is_relevant = _parse_relevance_grade(raw_grade)
        grade_log.append(
            {
                "index": idx,
                "relevant": is_relevant,
                "snippet": document[:200],
                "raw_grade": raw_grade[:200],
            }
        )
        if is_relevant:
            filtered_documents.append(document)

    shared_state["relevant_documents"] = filtered_documents
    shared_state["graded_documents"] = grade_log
    shared_state.setdefault("trace", []).append(
        f"Grader kept {len(filtered_documents)} of {len(documents)} documents as relevant."
    )
    return shared_state


def generate_answer_from_documents(shared_state: SharedState) -> SharedState:
    model = shared_state["model"]
    documents = shared_state["relevant_documents"]
    question = shared_state.get("original_question") or shared_state["question"]

    prompt = WEB_SEARCH_RAG_PROMPT if shared_state.get("used_web_search") else RAG_PROMPT
    rag_chain = prompt | model | StrOutputParser()
    model_response = rag_chain.invoke({"context": documents, "question": question})

    shared_state["agent_response"] = model_response
    source = "web search" if shared_state.get("used_web_search") else "Chroma Cloud knowledge base"
    shared_state.setdefault("trace", []).append(
        f"Generated final answer using {len(documents)} document(s) from {source}."
    )
    return shared_state


def decide_to_generate(shared_state: SharedState) -> str:
    if len(shared_state["relevant_documents"]) > 0:
        shared_state.setdefault("trace", []).append(
            "Decision: relevant documents found, skipping web search."
        )
        return "generate"
    shared_state.setdefault("trace", []).append(
        "Decision: no relevant documents, will rewrite query and run web search."
    )
    return "transform_query"


def transform_query(shared_state: SharedState) -> SharedState:
    question = shared_state["question"]
    model = shared_state["model"]

    if not shared_state.get("original_question"):
        shared_state["original_question"] = question

    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", WEB_SEARCH_QUESTION_REWRITER_PROMPT),
            ("human", "User question: {question}\n\nOptimized search query:"),
        ]
    )
    question_rewriter = re_write_prompt | model | StrOutputParser()
    raw_output = question_rewriter.invoke({"question": question})

    better_question = _clean_rewritten_query(raw_output, fallback=question)

    shared_state["question"] = better_question
    shared_state.setdefault("trace", []).append(
        f"Rewrote question to: '{better_question}'"
    )
    return shared_state


def _clean_rewritten_query(raw_output: str, fallback: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", raw_output or "", flags=re.IGNORECASE | re.DOTALL)
    cleaned = cleaned.strip().strip('"').strip("'").strip()

    for prefix in (
        "optimized search query:",
        "rewritten question:",
        "rewritten query:",
        "search query:",
        "query:",
        "improved question:",
    ):
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix):].strip().strip('"').strip("'").strip()

    first_line = cleaned.splitlines()[0].strip() if cleaned else ""
    candidate = first_line or cleaned or fallback
    return candidate[:300]


def perform_web_search(shared_state: SharedState) -> SharedState:
    question = shared_state["question"]
    web_search_tool = TavilySearch(
        max_results=5,
        search_depth="advanced",
        include_answer="advanced",
    )

    try:
        web_results = web_search_tool.invoke({"query": question})
    except TypeError:
        web_results = web_search_tool.invoke(question)

    results_list = (
        web_results.get("results", []) if isinstance(web_results, dict) else web_results or []
    )
    documents = [item.get("content", "") for item in results_list if item.get("content")]

    synthesized = (
        web_results.get("answer") if isinstance(web_results, dict) else None
    )
    if synthesized:
        documents = [synthesized, *documents]

    if not documents:
        documents = [f"No web search results were returned for the query: {question}"]

    shared_state["relevant_documents"] = documents
    shared_state["used_web_search"] = True
    shared_state.setdefault("trace", []).append(
        f"Tavily web search returned {len(documents)} document snippet(s)."
    )
    return shared_state


def build_graph():
    workflow = StateGraph(SharedState)

    workflow.add_node("get_model", get_model)
    workflow.add_node("get_relevant_documents", get_relevant_documents)
    workflow.add_node("grade_and_filter_documents", grade_and_filter_documents)
    workflow.add_node("generate_answer_from_documents", generate_answer_from_documents)
    workflow.add_node("perform_web_search", perform_web_search)
    workflow.add_node("transform_query", transform_query)

    workflow.add_edge(START, "get_model")
    workflow.add_edge("get_model", "get_relevant_documents")
    workflow.add_edge("get_relevant_documents", "grade_and_filter_documents")
    workflow.add_conditional_edges(
        "grade_and_filter_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate_answer_from_documents",
        },
    )
    workflow.add_edge("transform_query", "perform_web_search")
    workflow.add_edge("perform_web_search", "generate_answer_from_documents")
    workflow.add_edge("generate_answer_from_documents", END)

    return workflow.compile()
