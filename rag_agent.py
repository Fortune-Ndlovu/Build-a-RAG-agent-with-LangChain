"""Build a RAG agent with LangChain - tutorial scaffold.

Tutorial: https://docs.langchain.com/oss/python/langchain/rag

Indexing uses two of the author's own DEV posts instead of the tutorial's
sample URL. Covers: Setup, Components, and Indexing (load, split, store).

Optional env (defaults are conservative for Gemini embedding free-tier RPM):
  EMBED_BATCH_SIZE (default 20)
  EMBED_BATCH_PAUSE_SEC (default 15)
"""

from __future__ import annotations

import os
import time

os.environ.setdefault(
    "USER_AGENT",
    "rag-agent-tutorial/1.0 (https://docs.langchain.com/oss/python/langchain/rag)",
)

import bs4
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

required = ["LANGSMITH_API_KEY", "GOOGLE_API_KEY"]
missing = [k for k in required if not os.environ.get(k)]
if missing:
    raise RuntimeError(
        f"Missing required environment variables: {', '.join(missing)}. "
        "Copy .env.example to .env and fill in your keys."
    )

os.environ.setdefault("LANGSMITH_TRACING", "true")

model = init_chat_model("google_genai:gemini-2.5-flash-lite")

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vector_store = InMemoryVectorStore(embeddings)

# Personal content: Fortune Ndlovu on DEV Community
FORTUNE_DEV_POSTS = (
    "https://dev.to/fortune-ndlovu/intelligent-support-ticket-routing-with-natural-language-processing-nlp-57g1",
    "https://dev.to/fortune-ndlovu/image-classification-and-convolutional-neural-networks-cnns-4bdl",
)


if __name__ == "__main__":
    print(f"Chat model:    {model.__class__.__name__}")
    print(f"Embeddings:    {embeddings.__class__.__name__}")
    print(f"Vector store:  {vector_store.__class__.__name__}")

    # --- 1. Loading documents (DEV.to: main article HTML is #article-body) ---
    bs4_strainer = bs4.SoupStrainer(id="article-body")
    loader = WebBaseLoader(
        web_paths=FORTUNE_DEV_POSTS,
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()

    assert len(docs) == len(FORTUNE_DEV_POSTS)
    for i, doc in enumerate(docs):
        src = doc.metadata.get("source", "?")
        print(f"\n[{i + 1}] {src}")
        print(f"    Characters: {len(doc.page_content)}")
    print("\nPreview (first post, first 400 chars):")
    print(docs[0].page_content[:400])

    # --- 2. Splitting documents ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    all_splits = text_splitter.split_documents(docs)
    print(f"\nSplit into {len(all_splits)} sub-documents across {len(docs)} articles.")

    # --- 3. Storing documents (batched to stay under embedding RPM on free tier) ---
    batch_size = int(os.environ.get("EMBED_BATCH_SIZE", "20"))
    pause_sec = float(os.environ.get("EMBED_BATCH_PAUSE_SEC", "15"))
    document_ids: list[str] = []
    for start in range(0, len(all_splits), batch_size):
        batch = all_splits[start : start + batch_size]
        document_ids.extend(vector_store.add_documents(documents=batch))
        done = min(start + batch_size, len(all_splits))
        print(f"  Embedded {done}/{len(all_splits)} chunks...", flush=True)
        if start + batch_size < len(all_splits):
            time.sleep(pause_sec)

    print(document_ids[:3])
    print("\nIndexing complete. Vector store is ready for retrieval.")
