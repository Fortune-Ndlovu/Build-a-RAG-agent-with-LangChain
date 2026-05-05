"""DEV.to API ingest: markdown chunking, content-hash dedup, Chroma upsert."""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import requests
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langsmith import traceable

from rag_lab.config import Settings, get_vector_store

_log = logging.getLogger(__name__)

DEVTO_API = "https://dev.to/api"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@traceable(run_type="tool", name="devto_list_articles")
def list_article_summaries(settings: Settings) -> list[dict[str, Any]]:
    """List all articles for a user (summary rows, no body)."""
    url = f"{DEVTO_API}/articles"
    params: dict[str, Any] = {
        "username": settings.devto_username,
        "per_page": settings.devto_listing_per_page,
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        return []
    return data


@traceable(run_type="tool", name="devto_fetch_article_body")
def fetch_article_full(article_id: int) -> dict[str, Any]:
    r = requests.get(f"{DEVTO_API}/articles/{article_id}", timeout=60)
    r.raise_for_status()
    return r.json()


@traceable(run_type="chain", name="discover_articles")
def discover_articles(settings: Settings) -> list[dict[str, Any]]:
    """Full article records with body_markdown for each id in the user feed."""
    summaries = list_article_summaries(settings)
    out: list[dict[str, Any]] = []
    for s in summaries:
        aid = s.get("id")
        if aid is None:
            continue
        full = fetch_article_full(int(aid))
        out.append(
            {
                "id": int(aid),
                "title": full.get("title", s.get("title", "")),
                "url": full.get("url", s.get("url", "")),
                "tag_list": full.get("tag_list", s.get("tag_list", [])) or [],
                "published_at": full.get("published_at", s.get("published_at", "")),
                "body_markdown": full.get("body_markdown") or "",
            }
        )
    return out


def _article_chunks(article: dict[str, Any]) -> list[Document]:
    body = article.get("body_markdown") or ""
    h = _sha256(body)
    if not body.strip():
        return []

    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        add_start_index=True,
    )
    sections = md_splitter.split_text(body)
    combined: list[Document] = []
    for sec in sections:
        sub = char_splitter.split_documents([sec])
        for d in sub:
            d.metadata["title"] = article.get("title", "")
            d.metadata["url"] = article.get("url", "")
            d.metadata["tags"] = ",".join(article.get("tag_list", []) or [])
            d.metadata["published_at"] = str(article.get("published_at", ""))
            d.metadata["article_id"] = str(article["id"])
            d.metadata["content_hash"] = h
        combined.extend(sub)
    if not combined:
        d = Document(
            page_content=body[:1200],
            metadata={
                "title": article.get("title", ""),
                "url": article.get("url", ""),
                "tags": ",".join(article.get("tag_list", []) or []),
                "published_at": str(article.get("published_at", "")),
                "article_id": str(article["id"]),
                "content_hash": h,
            },
        )
        combined = char_splitter.split_documents([d])
    return combined


@traceable(run_type="chain", name="chunk_article")
def chunk_article(article: dict[str, Any]) -> list[Document]:
    return _article_chunks(article)


def _existing_hashes_by_article(vector_store) -> dict[str, str]:
    col = vector_store.get(include=["metadatas"], limit=10_000_000)
    metas = col.get("metadatas") or []
    by_aid: dict[str, str] = {}
    for m in metas:
        if not m:
            continue
        aid = m.get("article_id")
        ch = m.get("content_hash")
        if aid and ch:
            by_aid[str(aid)] = str(ch)
    return by_aid


@traceable(run_type="chain", name="compute_changes")
def compute_changes(
    articles: list[dict[str, Any]], existing: dict[str, str]
) -> tuple[list[dict[str, Any]], int]:
    """Return (articles to (re)ingest, skip_count)."""
    to_process: list[dict[str, Any]] = []
    skip = 0
    for art in articles:
        body = art.get("body_markdown") or ""
        h = _sha256(body)
        aid = str(art["id"])
        if existing.get(aid) == h:
            skip += 1
            continue
        to_process.append(art)
    return to_process, skip


@traceable(run_type="chain", name="upsert_article_chunks")
def upsert_article_chunks(
    vector_store, article: dict[str, Any], chunks: list[Document]
) -> int:
    aid = str(article["id"])
    got = vector_store.get(where={"article_id": aid}, include=[])
    old_ids = got.get("ids") or []
    if old_ids:
        vector_store.delete(ids=list(old_ids))
    if not chunks:
        return 0
    vector_store.add_documents(chunks)
    return len(chunks)


@traceable(run_type="chain", name="ingest_devto")
def ingest_devto(settings: Settings, *, rebuild: bool = False) -> dict[str, Any]:
    vs = get_vector_store(settings)
    if rebuild:
        _log.info("Rebuilding: resetting Chroma collection.")
        vs.delete_collection()
        vs = get_vector_store(settings)

    articles = discover_articles(settings)
    existing: dict[str, str] = {}
    if not rebuild:
        existing = _existing_hashes_by_article(vs)

    to_process, skip = compute_changes(articles, existing)
    total_chunks = 0
    for art in to_process:
        chunks = chunk_article(art)
        n = upsert_article_chunks(vs, art, chunks)
        total_chunks += n
        _log.info("Ingested article %s: %s chunks", art["id"], n)

    return {
        "articles_found": len(articles),
        "articles_skipped_unchanged": skip,
        "articles_updated": len(to_process),
        "chunks_written": total_chunks,
    }
