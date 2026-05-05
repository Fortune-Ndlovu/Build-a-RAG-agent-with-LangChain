"""Always-retrieve chain strategy (single retrieval per turn via middleware)."""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt

from rag_lab.config import Settings
from rag_lab.strategies.prompts import CHAIN_SYSTEM


def build(
    model,
    vector_store,
    settings: Settings,
    *,
    checkpointer: Any = None,
    name: str | None = "strategy_chain",
):
    _ = settings

    @dynamic_prompt
    def prompt_with_context(request: ModelRequest) -> str:
        last = request.messages[-1]
        lc = getattr(last, "content", "")
        last_query = lc if isinstance(lc, str) else str(lc)
        docs = vector_store.similarity_search(last_query, k=4)
        docs_content = "\n\n".join(
            f"<passage source={doc.metadata!r}>\n{doc.page_content}\n</passage>"
            for doc in docs
        )
        return (
            CHAIN_SYSTEM
            + "\n<context>\n"
            + docs_content
            + "\n</context>"
        )

    return create_agent(
        model,
        tools=[],
        middleware=[prompt_with_context],
        checkpointer=checkpointer,
        name=name,
    )
