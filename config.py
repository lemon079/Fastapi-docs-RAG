"""
FastAPI RAG Application - Core Configuration
Centralized configuration for the RAG pipeline.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "fastapi-rag")

# Model Configuration
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-oss:20b-cloud")
EMBEDDING_MODEL = "llama-text-embed-v2"  # Pinecone's inference embedding model

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
