"""Embeddings, vector store, retriever, and the agent's search tool.

Everything here is lazy: importing this module opens no network connections and
requires no API keys. The Pinecone client is built on first use.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.tools import tool
from langchain_mistralai import MistralAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from .config import (
    NO_PASSAGES_REPLY,
    OUT_OF_SCOPE,
    OUT_OF_SCOPE_REPLY,
    get_settings,
)
from .query import clean_query

SEARCH_TOOL_DESCRIPTION = (
    "Search Marcus Aurelius' Meditations for relevant teachings. "
    "Use this when the user asks about Stoic philosophy, Marcus Aurelius, "
    "or needs guidance on handling emotions, adversity, or life challenges."
)


@lru_cache(maxsize=1)
def get_embeddings() -> MistralAIEmbeddings:
    return MistralAIEmbeddings(model=get_settings().embedding_model)


@lru_cache(maxsize=1)
def get_vector_store() -> PineconeVectorStore:
    settings = get_settings()
    return PineconeVectorStore(
        index_name=settings.index_name,
        embedding=get_embeddings(),
    )


@lru_cache(maxsize=1)
def get_retriever():
    settings = get_settings()
    return get_vector_store().as_retriever(
        search_type=settings.search_type,
        search_kwargs={
            "k": settings.search_k,
            "fetch_k": settings.search_fetch_k,
            "lambda_mult": settings.search_lambda_mult,
        },
    )


def format_passages(docs) -> str:
    return "\n\n".join(
        f"[Teaching {i + 1}]\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )


@lru_cache(maxsize=1)
def get_search_tool():
    """The LangChain tool the agent calls to ground its answer.

    Query cleanup and the out-of-scope short circuit both happen here, locally,
    before anything is sent to Pinecone.
    """
    retriever = get_retriever()

    @tool
    def search_meditations(query: str) -> str:
        """Search Marcus Aurelius' Meditations for relevant teachings.

        Use this when the user asks about Stoic philosophy, Marcus Aurelius,
        or needs guidance on handling emotions, adversity, or life challenges.
        """
        clean = clean_query(query)
        if clean == OUT_OF_SCOPE:
            return OUT_OF_SCOPE_REPLY
        docs = retriever.invoke(clean)
        if not docs:
            return NO_PASSAGES_REPLY
        return format_passages(docs)

    return search_meditations
