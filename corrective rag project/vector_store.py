import os
from typing import List
from dotenv import load_dotenv
import streamlit as st

import chromadb
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )


KNOWLEDGE_BASE_URLS = [
    "https://www.linkedin.com/pulse/word-embeddings-how-neural-net-understands-words-space-prateek-sbl5c/",
    "https://www.linkedin.com/pulse/dissecting-backpropagation-neural-networks-saurav-prateek-krcvc/",
]

CHROMA_COLLECTION_NAME = "rag-chroma"


def get_chroma_cloud_client() -> chromadb.CloudClient:
    api_key = os.getenv("CHROMA_API_KEY")
    tenant = os.getenv("CHROMA_TENANT")
    database = os.getenv("CHROMA_DATABASE")

    if not api_key or not tenant or not database:
        raise ValueError(
            "Chroma Cloud credentials missing. Please set CHROMA_API_KEY, "
            "CHROMA_TENANT and CHROMA_DATABASE in your .env file."
        )

    return chromadb.CloudClient(
        api_key=api_key,
        tenant=tenant,
        database=database,
    )


def _load_and_split_documents(urls: List[str]) -> List[Document]:
    raw_docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in raw_docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    return text_splitter.split_documents(docs_list)


def build_vector_store(progress_cb=None) -> Chroma:
    if progress_cb:
        progress_cb("Connecting to Chroma Cloud...")

    client = get_chroma_cloud_client()

    existing_collections = [c.name for c in client.list_collections()]

    if CHROMA_COLLECTION_NAME not in existing_collections:
        if progress_cb:
            progress_cb(f"Collection '{CHROMA_COLLECTION_NAME}' not found. Loading knowledge base...")
        doc_splits = _load_and_split_documents(KNOWLEDGE_BASE_URLS)
        if progress_cb:
            progress_cb(
                f"Embedding {len(doc_splits)} chunks with '{EMBEDDING_MODEL}' "
                f"and uploading to Chroma Cloud..."
            )

        vector_store = Chroma.from_documents(
            documents=doc_splits,
            embedding=get_embeddings(),
            client=client,
            collection_name=CHROMA_COLLECTION_NAME,
        )
    else:
        if progress_cb:
            progress_cb(
                f"Using existing Chroma Cloud collection '{CHROMA_COLLECTION_NAME}' "
                f"with embeddings '{EMBEDDING_MODEL}'."
            )
        vector_store = Chroma(
            client=client,
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=get_embeddings(),
        )

    return vector_store.as_retriever()


def reset_chroma_cloud_collection():
    client = get_chroma_cloud_client()
    try:
        client.delete_collection(CHROMA_COLLECTION_NAME)
    except Exception as exc:
        st.warning(f"Could not delete collection: {exc}")


@st.cache_resource(show_spinner=False)
def cached_build_vector_store():
    load_dotenv()
    status_box = st.status("Bootstrapping Chroma Cloud vector store...", expanded=True)
    progress_lines = []

    def progress_cb(msg: str):
        progress_lines.append(msg)
        status_box.update(label=msg)

    try:
        retriever = build_vector_store(progress_cb=progress_cb)
        status_box.update(label="Vector store ready", state="complete")
        return retriever
    except Exception as exc:
        status_box.update(label="Vector store failed to initialize", state="error")
        st.error(f"Failed to initialize vector store: {exc}")
        st.stop()
