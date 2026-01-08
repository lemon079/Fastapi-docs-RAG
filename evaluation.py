"""
FastAPI RAG Application - RAGAS Evaluation
Measures faithfulness and answer relevancy.
"""

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset
from langchain_ollama import OllamaEmbeddings, ChatOllama
from config import LLM_MODEL, EMBEDDING_MODEL


def evaluate_response(
    question: str,
    answer: str,
    contexts: list[str],
) -> dict:
    """
    Evaluate a RAG response using RAGAS metrics.

    Args:
        question: The user's question
        answer: The generated answer
        contexts: List of retrieved context chunks

    Returns:
        Dict with faithfulness and answer_relevancy scores
    """
    dataset = Dataset.from_dict(
        {
            "user_input": [question],
            "response": [answer],
            "retrieved_contexts": [contexts],
        }
    )

    llm = LangchainLLMWrapper(ChatOllama(model=LLM_MODEL))
    embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=EMBEDDING_MODEL))

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=llm,
        embeddings=embeddings,
    )

    return result.to_pandas().iloc[0].to_dict()
