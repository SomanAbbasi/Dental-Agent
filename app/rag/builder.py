import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from app.config.logger import get_logger

logger = get_logger(__name__)

POLICIES_DIR = Path("data/policies")
FAISS_INDEX_DIR = Path("data/faiss_index")

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50


def get_embeddings() -> HuggingFaceEmbeddings:
  
    logger.info("loading_embeddings_model")
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_policy_documents() -> list:
   
    if not POLICIES_DIR.exists():
        raise FileNotFoundError(
            f"Policies directory not found: {POLICIES_DIR}"
        )

    all_docs = []
    txt_files = list(POLICIES_DIR.glob("*.txt"))

    if not txt_files:
        raise ValueError(
            f"No .txt policy files found in {POLICIES_DIR}"
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "===", "Q:", ". ", " "],
    )

    for filepath in txt_files:
        logger.info("loading_policy_file", file=str(filepath))
        loader = TextLoader(str(filepath), encoding="utf-8")
        docs = loader.load()
        chunks = splitter.split_documents(docs)
        all_docs.extend(chunks)
        logger.info(
            "policy_file_chunked",
            file=filepath.name,
            chunks=len(chunks),
        )

    logger.info("total_chunks_loaded", count=len(all_docs))
    return all_docs


def build_faiss_index(force_rebuild: bool = False) -> FAISS:
    """
    Builds or loads the FAISS vector index.

    If the index already exists on disk, loads it (fast).
    If not, builds from policy documents and saves to disk.
    force_rebuild=True rebuilds even if index exists — use when
    you update policy documents.
    """
    embeddings = get_embeddings()

    if FAISS_INDEX_DIR.exists() and not force_rebuild:
        logger.info("loading_existing_faiss_index", path=str(FAISS_INDEX_DIR))
        return FAISS.load_local(
            str(FAISS_INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    logger.info("building_faiss_index_from_scratch")
    docs = load_policy_documents()
    vectorstore = FAISS.from_documents(docs, embeddings)

    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(FAISS_INDEX_DIR))
    logger.info("faiss_index_saved", path=str(FAISS_INDEX_DIR))

    return vectorstore