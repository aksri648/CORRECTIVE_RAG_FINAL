import os
import tempfile
from typing import List
from dotenv import load_dotenv
import streamlit as st

import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
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


CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "rag-pdf-chroma")


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


def _load_and_split_pdf_files(uploaded_files) -> List[Document]:
    docs_list = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            pdf_path = os.path.join(temp_dir, uploaded_file.name)
            with open(pdf_path, "wb") as pdf_file:
                pdf_file.write(uploaded_file.getbuffer())

            pages = PyPDFLoader(pdf_path).load()
            for page in pages:
                page.metadata["source"] = uploaded_file.name
            docs_list.extend(pages)

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=750, chunk_overlap=100
    )
    return text_splitter.split_documents(docs_list)


def collection_exists() -> bool:
    client = get_chroma_cloud_client()
    return CHROMA_COLLECTION_NAME in [c.name for c in client.list_collections()]


def get_existing_retriever(progress_cb=None):
    if progress_cb:
        progress_cb("Connecting to Chroma Cloud...")

    client = get_chroma_cloud_client()
    existing_collections = [c.name for c in client.list_collections()]

    if CHROMA_COLLECTION_NAME not in existing_collections:
        return None

    vector_store = Chroma(
        client=client,
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=get_embeddings(),
    )
    return vector_store.as_retriever()


def index_uploaded_pdfs(uploaded_files, progress_cb=None):
    if not uploaded_files:
        raise ValueError("Upload at least one PDF before indexing.")

    if progress_cb:
        progress_cb("Connecting to Chroma Cloud...")

    client = get_chroma_cloud_client()
    reset_chroma_cloud_collection(show_warning=False)

    if progress_cb:
        progress_cb(f"Loading and splitting {len(uploaded_files)} PDF file(s)...")
    doc_splits = _load_and_split_pdf_files(uploaded_files)

    if not doc_splits:
        raise ValueError("No readable text was found in the uploaded PDFs.")

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

    return vector_store.as_retriever(), len(doc_splits)


def reset_chroma_cloud_collection(show_warning=True):
    client = get_chroma_cloud_client()
    try:
        client.delete_collection(CHROMA_COLLECTION_NAME)
    except Exception as exc:
        if show_warning:
            st.warning(f"Could not delete collection: {exc}")


@st.cache_resource(show_spinner=False)
def cached_build_vector_store():
    load_dotenv()
    status_box = st.status("Connecting to Chroma Cloud vector store...", expanded=True)
    progress_lines = []

    def progress_cb(msg: str):
        progress_lines.append(msg)
        status_box.update(label=msg)

    try:
        retriever = get_existing_retriever(progress_cb=progress_cb)
        if retriever is None:
            status_box.update(label="No indexed PDF collection found", state="error")
            return None
        status_box.update(label="Vector store ready", state="complete")
        return retriever
    except Exception as exc:
        status_box.update(label="Vector store failed to initialize", state="error")
        st.error(f"Failed to initialize vector store: {exc}")
        st.stop()
