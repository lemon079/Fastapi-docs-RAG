"""
FastAPI RAG Application - Retrieval Module
Uses Pinecone's inference API for embeddings (llama-text-embed-v2).
"""

from langchain.tools import tool
from pinecone import Pinecone
from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX,
    EMBEDDING_MODEL,
    RETRIEVAL_K,
)

# Initialize Pinecone
_pc = Pinecone(api_key=PINECONE_API_KEY)
_index = _pc.Index(PINECONE_INDEX)


def embed_text(text: str) -> list:
    """Generate embeddings using Pinecone's inference API."""
    response = _pc.inference.embed(
        model=EMBEDDING_MODEL, inputs=[text], parameters={"input_type": "query"}
    )
    return response.data[0].values


def embed_documents(texts: list) -> list:
    """Generate embeddings for multiple documents."""
    response = _pc.inference.embed(
        model=EMBEDDING_MODEL, inputs=texts, parameters={"input_type": "passage"}
    )
    return [item.values for item in response.data]


def search_documents(query: str, k: int = RETRIEVAL_K) -> list:
    """
    Search Pinecone for documents similar to the query.

    Args:
        query: The search query
        k: Number of results to return

    Returns:
        List of matching documents with scores and metadata
    """
    query_embedding = embed_text(query)
    results = _index.query(
        vector=query_embedding,
        top_k=k,
        include_metadata=True,
    )
    return results.matches


@tool
def retrieve_context(query: str) -> str:
    """
    Retrieve relevant information from the FastAPI documentation.
    Use this tool to find specific information before answering questions.
    """
    matches = search_documents(query, k=RETRIEVAL_K)

    if not matches:
        return "No relevant documentation found for this query."

    results = []
    for match in matches:
        text = match.metadata.get("text", "")
        score = match.score
        page = match.metadata.get("page", 0)

        if text:
            results.append(f"[Relevance: {score:.2f} | Page {page + 1}]\n{text}")

    if not results:
        return "No relevant documentation found for this query."

    return "\n\n---\n\n".join(results)


def get_retriever_tool():
    """Return the retriever tool for use in agents."""
    return retrieve_context
