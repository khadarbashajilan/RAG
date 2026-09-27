"""Ingest: Meditations PDF -> chunked embeddings -> Pinecone.

Run once (or after swapping the source text). Re-running is safe: chunk IDs are
derived from a hash of the page content, so anything already stored is skipped.
Not needed at runtime.
"""

from __future__ import annotations

import hashlib
import re
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from .config import (
    PDF_DOWNLOAD_HINT,
    MissingKeysError,
    get_key,
    get_settings,
    require_keys,
)
from .retrieval import get_embeddings

_RETRY_DELAY_RE = re.compile(r"retryDelay'?:?\s*'?(\d+(?:\.\d+)?)s")


def chunk_id(doc) -> str:
    """Stable per-chunk ID, so re-ingest skips vectors that already exist."""
    digest = hashlib.sha1(doc.page_content.encode("utf-8")).hexdigest()
    return f"p{doc.metadata.get('page', 0)}-{digest[:16]}"


def load_chunks(pdf_path, chunk_size: int, chunk_overlap: int):
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Source PDF not found at {pdf_path}.\n{PDF_DOWNLOAD_HINT}"
        )
    docs = PyPDFLoader(str(pdf_path)).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)


def get_client() -> Pinecone:
    return Pinecone(api_key=get_key("PINECONE_API_KEY"))


def ensure_index(pc: Pinecone, index_name: str, dimension: int, metric: str,
                 cloud: str, region: str) -> None:
    if index_name in pc.list_indexes().names():
        print(f"Index '{index_name}' already exists.")
        return
    print(f"Creating index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(cloud=cloud, region=region),
    )
    print("Index created.")


def upsert_with_retry(vector_store, documents, ids, max_attempts: int) -> None:
    for attempt in range(1, max_attempts + 1):
        try:
            vector_store.add_documents(documents=documents, ids=ids)
            return
        except Exception as exc:
            if attempt == max_attempts:
                raise
            match = _RETRY_DELAY_RE.search(str(exc))
            wait = float(match.group(1)) + 2 if match else min(2 ** attempt, 60)
            print(f"  Retry {attempt}/{max_attempts} in {wait:.0f}s")
            time.sleep(wait)


def ingest() -> None:
    settings = get_settings()

    chunks = load_chunks(
        settings.pdf_path, settings.chunk_size, settings.chunk_overlap
    )
    print(f"Loaded {len(chunks)} chunks from {settings.pdf_path.name}.")

    pc = get_client()
    ensure_index(
        pc,
        settings.index_name,
        settings.index_dimension,
        settings.index_metric,
        settings.index_cloud,
        settings.index_region,
    )

    vector_store = PineconeVectorStore(
        index_name=settings.index_name,
        embedding=get_embeddings(),
    )

    all_ids = [chunk_id(doc) for doc in chunks]
    existing = set(pc.Index(settings.index_name).fetch(ids=all_ids).vectors.keys())
    pending = [(i, d) for i, d in zip(all_ids, chunks) if i not in existing]
    print(f"{len(existing)} chunks already stored, {len(pending)} to embed.")

    batch_size = settings.ingest_batch_size
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        upsert_with_retry(
            vector_store,
            [d for _, d in batch],
            [i for i, _ in batch],
            settings.ingest_max_attempts,
        )
        print(f"  {min(start + batch_size, len(pending))}/{len(pending)} upserted")

    total = pc.Index(settings.index_name).describe_index_stats().total_vector_count
    print(f"Index '{settings.index_name}' now holds {total} vectors.")


def main() -> None:
    try:
        require_keys()
    except MissingKeysError as exc:
        print(exc)
        raise SystemExit(1) from None
    ingest()


if __name__ == "__main__":
    main()
