import os
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import END, StateGraph, START

from prompts import GRADE_DOCUMENTS_PROMPT, QUESTION_REWRITER_PROMPT


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


class SharedState(TypedDict, total=False):
    question: str
    agent_response: str
    vector_store: object
    relevant_documents: list[str]
    graded_documents: list[dict]
    model: ChatOpenAI
    trace: list[str]
    used_web_search: bool


class GradeDocuments(BaseModel):
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


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
    structured_llm_grader = model.with_structured_output(GradeDocuments)

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", GRADE_DOCUMENTS_PROMPT),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )

    retrieval_grader = grade_prompt | structured_llm_grader

    filtered_documents = []
    grade_log = []
    for idx, document in enumerate(documents, start=1):
        grader_response = retrieval_grader.invoke(
            {"question": question, "document": document}
        )
        is_relevant = grader_response.binary_score.lower() == "yes"
        grade_log.append({"index": idx, "relevant": is_relevant, "snippet": document[:200]})
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
    question = shared_state["question"]
    documents = shared_state["relevant_documents"]

    rag_chain = RAG_PROMPT | model | StrOutputParser()
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

    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", QUESTION_REWRITER_PROMPT),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question.",
            ),
        ]
    )
    question_rewriter = re_write_prompt | model | StrOutputParser()
    better_question = question_rewriter.invoke({"question": question})

    shared_state["question"] = better_question
    shared_state.setdefault("trace", []).append(f"Rewrote question to: '{better_question}'")
    return shared_state


def perform_web_search(shared_state: SharedState) -> SharedState:
    question = shared_state["question"]
    web_search_tool = TavilySearch(max_results=3)

    web_results = web_search_tool.invoke({"query": question})
    results_list = web_results.get("results", []) if isinstance(web_results, dict) else web_results
    documents = [item.get("content", "") for item in results_list if item.get("content")]

    shared_state["relevant_documents"] = documents
    shared_state["used_web_search"] = True
    shared_state.setdefault("trace", []).append(
        f"Tavily web search returned {len(documents)} document(s)."
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
