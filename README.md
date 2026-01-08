# FastAPI RAG Application

A professional Retrieval-Augmented Generation (RAG) application for querying FastAPI documentation using Ollama and Pinecone.

## Features

- 📚 **Document Indexing** - Index PDF documentation into Pinecone vector database
- 🔍 **Semantic Search** - Find relevant documentation chunks using embeddings
- 🤖 **AI Assistant** - Get accurate answers grounded in the documentation
- 📊 **RAGAS Evaluation** - Measure response faithfulness and detect hallucinations

## Project Structure

```
├── app.py          # Streamlit web interface
├── config.py       # Centralized configuration
├── indexer.py      # Document indexing script
├── retriever.py    # Vector search and retrieval
├── evaluation.py   # RAGAS evaluation metrics
└── fastapi_tutorial.pdf  # Source documentation
```

## Quick Start

### 1. Install Dependencies

```bash
uv sync
```

### 2. Index Documents (one-time)

```bash
uv run python indexer.py
```

### 3. Run the Application

```bash
uv run streamlit run app.py
```

Open <http://localhost:8501> in your browser.

## Configuration

Edit `config.py` to customize:

| Setting | Description | Default |
|---------|-------------|---------|
| `LLM_MODEL` | Ollama LLM model | `gpt-oss:20b-cloud` |
| `EMBEDDING_MODEL` | Embedding model | `mxbai-embed-large:latest` |
| `CHUNK_SIZE` | Characters per chunk | `600` |
| `RETRIEVAL_K` | Number of chunks to retrieve | `4` |

## Evaluation

The app includes RAGAS-powered evaluation:

- **Faithfulness** - Is the answer grounded in sources? (0-1)
- **Answer Relevancy** - Does it answer the question? (0-1)

A faithfulness score below 0.4 indicates high hallucination risk.

## Tech Stack

- **Streamlit** - Web interface
- **LangChain** - Agent framework
- **Ollama** - Local LLM and embeddings
- **Pinecone** - Vector database
- **RAGAS** - Evaluation metrics
