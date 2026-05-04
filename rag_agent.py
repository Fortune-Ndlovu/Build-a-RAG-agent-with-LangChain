"""Build a RAG agent with LangChain - tutorial scaffold.

Tutorial: https://docs.langchain.com/oss/python/langchain/rag

This file currently covers the Setup stage:
  1. Load environment variables (LangSmith + Google) from .env
  2. Initialize the chat model (Google Gemini)
  3. Initialize the embeddings model (Google Gemini)
  4. Initialize an in-memory vector store

Subsequent tutorial sections (Indexing -> Retrieval -> Generation) will
build on these objects.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

required = ["LANGSMITH_API_KEY", "GOOGLE_API_KEY"]
missing = [k for k in required if not os.environ.get(k)]
if missing:
    raise RuntimeError(
        f"Missing required environment variables: {', '.join(missing)}. "
        "Copy .env.example to .env and fill in your keys."
    )

os.environ.setdefault("LANGSMITH_TRACING", "true")

from langchain.chat_models import init_chat_model
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

model = init_chat_model("google_genai:gemini-2.5-flash-lite")

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vector_store = InMemoryVectorStore(embeddings)


def _smoke_test() -> None:
    """Quick check that all three components initialized correctly."""
    print(f"Chat model:    {model.__class__.__name__}")
    print(f"Embeddings:    {embeddings.__class__.__name__}")
    print(f"Vector store:  {vector_store.__class__.__name__}")
    print("\nPinging chat model...")
    response = model.invoke("Reply with a single word: ready")
    print(f"  -> {response.content!r}")
    print("\nSetup looks good. You're ready for the Indexing section.")


if __name__ == "__main__":
    _smoke_test()