"""
FastAPI RAG Application - Document Indexer
Uses Pinecone's inference API for embeddings (llama-text-embed-v2).
Run this script once to index your PDF into Pinecone.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    PDF_PATH,
)


def load_and_split_pdf(pdf_path: str) -> list:
    """Load PDF and split into chunks."""
    print(f"📄 Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"   Loaded {len(docs)} pages")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(docs)
    print(f"   Split into {len(chunks)} chunks")
    return chunks


def embed_batch(pc: Pinecone, texts: list) -> list:
    """Generate embeddings using Pinecone's inference API."""
    response = pc.inference.embed(
        model=EMBEDDING_MODEL, inputs=texts, parameters={"input_type": "passage"}
    )
    return [item.values for item in response.data]


def index_documents(documents: list) -> None:
    """Index documents into Pinecone using batch embedding."""
    print(f"\n🌲 Connecting to Pinecone index: {PINECONE_INDEX}")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)

    print(f"🧠 Using embedding model: {EMBEDDING_MODEL}")
    print(f"\n📤 Indexing {len(documents)} documents...")

    batch_size = 50
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]

        # Get texts and embeddings for batch
        texts = [doc.page_content for doc in batch]
        embeddings = embed_batch(pc, texts)

        # Prepare vectors for upsert
        vectors = []
        for j, (doc, embedding) in enumerate(zip(batch, embeddings)):
            vectors.append(
                {
                    "id": f"doc_{i + j}",
                    "values": embedding,
                    "metadata": {
                        "text": doc.page_content,
                        "page": doc.metadata.get("page", 0),
                        "source": doc.metadata.get("source", ""),
                    },
                }
            )

        # Upsert batch
        index.upsert(vectors=vectors)
        print(f"   Progress: {min(i + batch_size, len(documents))}/{len(documents)}")

    print(f"\n✅ Successfully indexed {len(documents)} documents!")


def main():
    """Main indexing function."""
    print("=" * 50)
    print("FastAPI RAG - Document Indexer")
    print(f"Embedding Model: {EMBEDDING_MODEL}")
    print("=" * 50)

    chunks = load_and_split_pdf(PDF_PATH)

    confirm = input(f"\nIndex {len(chunks)} chunks to Pinecone? (y/n): ")
    if confirm.lower() == "y":
        index_documents(chunks)
    else:
        print("Indexing cancelled.")


if __name__ == "__main__":
    main()
