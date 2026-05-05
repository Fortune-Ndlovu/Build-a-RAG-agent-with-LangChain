"""Corrective RAG (retrieve → grade → rewrite loop → generate)."""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from rag_lab.config import Settings
from rag_lab.strategies.prompts import (
    CRAG_GENERATE_PROMPT,
    CRAG_GRADE_PROMPT,
    CRAG_REWRITE_PROMPT,
)


def _last_user_text(messages: list[BaseMessage]) -> str:
    for m in reversed(messages):
        t = getattr(m, "type", None)
        if t == "human" or m.__class__.__name__ == "HumanMessage":
            c = m.content
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                parts: list[str] = []
                for block in c:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(str(block.get("text", "")))
                return "\n".join(parts) if parts else str(c)
            return str(c)
    return ""


class CRAGState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    docs: list[Document]
    attempts: int
    sufficient: bool


def build(model, vector_store, settings: Settings, *, checkpointer: Any = None, name: str | None = None):
    _ = settings

    def prepare(state: CRAGState) -> dict[str, Any]:
        q = _last_user_text(state["messages"])
        return {"query": q, "attempts": 0, "docs": [], "sufficient": False}

    def retrieve(state: CRAGState) -> dict[str, Any]:
        docs = vector_store.similarity_search(state["query"], k=4)
        return {"docs": docs}

    def grade(state: CRAGState) -> dict[str, Any]:
        question = _last_user_text(state["messages"])
        docs = state.get("docs") or []
        passages = "\n\n".join(
            f"[{i}] meta={doc.metadata!r}\n{doc.page_content[:1200]}"
            for i, doc in enumerate(docs)
        )
        prompt = CRAG_GRADE_PROMPT.format(question=question, passages=passages)
        resp = model.invoke([HumanMessage(content=prompt)])
        text = (resp.content or "").strip().upper()
        sufficient = text.startswith("Y")
        return {"sufficient": sufficient}

    def rewrite(state: CRAGState) -> dict[str, Any]:
        question = _last_user_text(state["messages"])
        prompt = CRAG_REWRITE_PROMPT.format(
            question=question,
            query=state["query"],
            rationale="Graded as insufficient for answering.",
        )
        resp = model.invoke([HumanMessage(content=prompt)])
        new_q = (resp.content or "").strip().strip('"').strip("'")
        return {"query": new_q, "attempts": state.get("attempts", 0) + 1}

    def generate(state: CRAGState) -> dict[str, Any]:
        question = _last_user_text(state["messages"])
        docs = state.get("docs") or []
        passages = "\n\n".join(
            f"[{i}] meta={doc.metadata!r}\n{doc.page_content}"
            for i, doc in enumerate(docs)
        )
        prompt = CRAG_GENERATE_PROMPT.format(question=question, passages=passages)
        resp = model.invoke([HumanMessage(content=prompt)])
        return {"messages": [AIMessage(content=resp.content or "")]}

    def route_after_grade(state: CRAGState) -> str:
        if state.get("sufficient"):
            return "generate"
        if state.get("attempts", 0) >= 2:
            return "generate"
        return "rewrite"

    graph = StateGraph(CRAGState)
    graph.add_node("prepare", prepare)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade)
    graph.add_node("rewrite", rewrite)
    graph.add_node("generate", generate)

    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", route_after_grade, {"generate": "generate", "rewrite": "rewrite"})
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", END)

    return graph.compile(checkpointer=checkpointer, name=name or "strategy_crag")
