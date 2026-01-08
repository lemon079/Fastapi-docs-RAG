"""
FastAPI RAG Application - Core Configuration
Supports both local (.env) and Streamlit Cloud (secrets) deployment.
"""

import os

# Try to load from .env for local development
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def get_config(key: str, default: str = None) -> str:
    """Get config from Streamlit secrets or environment variables."""
    try:
        import streamlit as st

        if key in st.secrets:
            return st.secrets[key]
    except:
        pass
    return os.getenv(key, default)


# Pinecone Configuration
PINECONE_API_KEY = get_config(
    "PINECONE_API_KEY",
    "pcsk_5z1jLP_QLUw6F4EAEbj3XSnmKq6Fs7mpPQgaFfko81uCwpGMCjf7G8XPYVCaiMCL9KKNHM",
)
PINECONE_INDEX = get_config("PINECONE_INDEX", "fastapi-rag")

# Gemini Configuration
GEMINI_API_KEY = get_config(
    "GEMINI_API_KEY",
    "AIzaSyDdKL4-geM9qex4HpaIpZ8OuOn1RwtwoCI",
)
LLM_MODEL = "gemini-2.5-flash"

# Embedding Model (Pinecone's inference API)
EMBEDDING_MODEL = "llama-text-embed-v2"

# RAG Configuration
CHUNK_SIZE = 600
CHUNK_OVERLAP = 50
RETRIEVAL_K = 4

# Document Path
PDF_PATH = "./fastapi_tutorial.pdf"

# System Prompt
SYSTEM_PROMPT = """You are a professional FastAPI documentation assistant. Your role is to provide accurate, helpful answers about FastAPI based solely on the provided documentation.

Guidelines:
1. ONLY use information from the retrieved context to answer questions
2. If the context doesn't contain relevant information, clearly state that
3. Provide code examples when appropriate
4. Be concise and professional in your responses
5. Cite specific concepts from the documentation when possible

When you don't have enough information, say: "Based on the available documentation, I don't have specific information about that topic. Please consult the official FastAPI documentation for more details."

Always use the retrieve_context tool before answering any question."""
