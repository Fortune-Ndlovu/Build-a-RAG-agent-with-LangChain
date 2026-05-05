"""RAG tools for the agent strategy (not used by chain/CRAG)."""

from __future__ import annotations

from typing import Any

import requests
from langchain.tools import tool
from langchain_chroma import Chroma

from rag_lab.config import Settings
from rag_lab.ingest import list_article_summaries

DEVTO_API = "https://dev.to/api"


def make_rag_tools(settings: Settings, vector_store: Chroma) -> list:
    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str, k: int = 4) -> tuple[str, list]:
        """Retrieve the most similar passages from the indexed DEV.to articles."""
        docs = vector_store.similarity_search(query, k=int(k))
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in docs
        )
        return serialized, docs

    @tool
    def list_articles(tag: str | None = None) -> str:
        """List the author's DEV.to article titles and URLs. Optional tag filter (substring)."""
        rows = list_article_summaries(settings)
        lines: list[str] = []
        for s in rows:
            title = s.get("title", "")
            url = s.get("url", "")
            tags = s.get("tag_list", []) or []
            if tag:
                tlow = tag.lower()
                if not any(tlow in (x or "").lower() for x in tags) and tlow not in (
                    title or ""
                ).lower():
                    continue
            lines.append(f"- {title}\n  {url}\n  tags: {', '.join(tags)}")
        return "\n".join(lines) if lines else "No articles matched."

    @tool
    def fetch_article(url: str) -> str:
        """Fetch full markdown body for a DEV.to article URL belonging to this author."""
        url = (url or "").strip()
        rows = list_article_summaries(settings)
        match: dict[str, Any] | None = None
        for s in rows:
            if s.get("url", "").rstrip("/") == url.rstrip("/"):
                match = s
                break
        if not match:
            for s in rows:
                if url and url in (s.get("url") or ""):
                    match = s
                    break
        if not match or match.get("id") is None:
            return "Could not resolve URL to an article in this author's feed."
        aid = int(match["id"])
        r = requests.get(f"{DEVTO_API}/articles/{aid}", timeout=60)
        r.raise_for_status()
        data: dict[str, Any] = r.json()
        username = (data.get("user") or {}).get("username", "")
        if username != settings.devto_username:
            return "URL does not belong to the configured author's DEV.to account."
        md = data.get("body_markdown") or ""
        return f"# {data.get('title', '')}\n\n{md}"

    return [retrieve_context, list_articles, fetch_article]
