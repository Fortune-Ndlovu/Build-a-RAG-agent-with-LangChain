"""Pluggable RAG strategy builders (agent, chain, CRAG)."""

from __future__ import annotations

from rag_lab.strategies import agent, chain, crag

STRATEGIES: dict = {
    "agent": agent.build,
    "chain": chain.build,
    "crag": crag.build,
}

__all__ = ["STRATEGIES", "agent", "chain", "crag"]
