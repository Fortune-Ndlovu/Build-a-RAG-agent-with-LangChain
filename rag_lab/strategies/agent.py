"""Tool-calling agent strategy."""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent

from rag_lab.config import Settings
from rag_lab.strategies.prompts import AGENT_SYSTEM
from rag_lab.tools import make_rag_tools


def build(
    model,
    vector_store,
    settings: Settings,
    *,
    checkpointer: Any = None,
    name: str | None = "strategy_agent",
):
    tools = make_rag_tools(settings, vector_store)
    return create_agent(
        model,
        tools,
        system_prompt=AGENT_SYSTEM,
        checkpointer=checkpointer,
        name=name,
    )
