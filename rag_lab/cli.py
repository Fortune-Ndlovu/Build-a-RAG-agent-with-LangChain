"""CLI: ingest corpus, chat with strategy selection, run LangSmith benchmarks."""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
import uuid
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages import HumanMessage as LCHumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langsmith import traceable

from rag_lab.bench import run_benchmark
from rag_lab.config import (
    get_chat_model,
    get_vector_store,
    load_settings,
    setup_logging,
    setup_tracing,
)
from rag_lab.ingest import ingest_devto
from rag_lab.strategies import STRATEGIES

_log = logging.getLogger(__name__)


def _stdout_utf8() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


def _sqlite_conn(settings):
    path = Path(settings.sqlite_threads_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(path), check_same_thread=False)


def list_thread_ids(settings) -> list[str]:
    """Best-effort list of thread ids from LangGraph Sqlite checkpoints."""
    db_path = Path(settings.sqlite_threads_path)
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id DESC LIMIT 50"
        ).fetchall()
        return [str(r[0]) for r in rows if r and r[0] is not None]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def _format_tool_sources(messages: list) -> str:
    lines: list[str] = []
    n = 0
    for m in messages:
        if not isinstance(m, ToolMessage):
            continue
        art = getattr(m, "artifact", None)
        if not art or not isinstance(art, list) or not art:
            continue
        if not isinstance(art[0], Document):
            continue
        for doc in art:
            meta = getattr(doc, "metadata", {}) or {}
            u = meta.get("url", "")
            h2 = meta.get("h2") or meta.get("h1") or ""
            title = meta.get("title", "")
            n += 1
            label = f"{title} — {h2}" if h2 else (title or u)
            lines.append(f"  [{n}] {label}\n      {u}")
    if not lines:
        return "  (no retrieve_context tool calls in this turn — see LangSmith trace for internal retrieval.)"
    return "\n".join(lines)


@traceable(name="chat_turn", run_type="chain", tags=["chat"])
def _run_stream(
    runnable,
    user_text: str,
    config: dict,
) -> list:
    final_messages: list = []
    for event in runnable.stream(
        {"messages": [LCHumanMessage(content=user_text)]},
        config=config,
        stream_mode="values",
    ):
        if event and "messages" in event:
            final_messages = list(event["messages"])
    return final_messages


def cmd_ingest(settings, args: argparse.Namespace) -> None:
    stats = ingest_devto(settings, rebuild=args.rebuild)
    _log.info("Ingest complete: %s", stats)
    print(stats)


def cmd_bench(settings, args: argparse.Namespace) -> None:
    strategies = ["agent", "chain", "crag"] if args.strategy == "all" else [args.strategy]
    for name in strategies:
        print(f"\n=== strategy:{name} ===\n")
        res = run_benchmark(settings, strategy_name=name, limit=args.limit)
        print("feedback_stats:", getattr(res, "feedback_stats", res))
        print("run_stats:", getattr(res, "run_stats", ""))


def cmd_chat(settings, args: argparse.Namespace) -> None:
    model = get_chat_model(settings)
    vector_store = get_vector_store(settings)
    conn = _sqlite_conn(settings)
    checkpointer = SqliteSaver(conn)
    strategy = args.strategy
    if strategy not in STRATEGIES:
        raise SystemExit(f"Unknown strategy {strategy!r}. Choose: {list(STRATEGIES)}")

    def build_runnable(name: str):
        b = STRATEGIES[name]
        return b(
            model,
            vector_store,
            settings,
            checkpointer=checkpointer,
        )

    runnable = build_runnable(strategy)
    session_id = args.thread or str(uuid.uuid4())
    turn = 0
    print(
        f"Thread: {session_id}  |  strategy: {strategy}\n"
        "Commands: /help /sources /reset /rebuild /threads /resume <id> /strategy <name> /quit"
    )

    while True:
        try:
            raw = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw:
            continue
        if raw == "/quit":
            break
        if raw == "/help":
            print(
                "/help  /sources  /reset  /rebuild  /threads  /resume <id>  "
                "/strategy <agent|chain|crag>  /quit"
            )
            continue
        if raw == "/reset":
            session_id = str(uuid.uuid4())
            print(f"New thread: {session_id}")
            continue
        if raw == "/rebuild":
            print(ingest_devto(settings, rebuild=True))
            continue
        if raw == "/threads":
            tids = list_thread_ids(settings)
            print("Recent thread ids:" if tids else "No checkpoints yet.")
            for t in tids:
                print(" ", t)
            continue
        if raw.startswith("/resume "):
            new_id = raw.split(maxsplit=1)[1].strip()
            if new_id:
                session_id = new_id
                print(f"Resumed thread: {session_id}")
            continue
        if raw.startswith("/strategy "):
            new_s = raw.split(maxsplit=1)[1].strip()
            if new_s not in STRATEGIES:
                print("Unknown strategy:", new_s)
                continue
            strategy = new_s
            runnable = build_runnable(strategy)
            print(f"Strategy -> {strategy}")
            continue
        if raw == "/sources":
            print("Use /sources after an assistant reply (sources are from last turn).")
            continue

        turn += 1
        cfg = {
            "configurable": {"thread_id": session_id},
            "run_name": "chat_turn",
            "tags": ["chat", f"strategy:{strategy}", f"thread:{session_id}"],
            "metadata": {
                "thread_id": session_id,
                "turn": turn,
                "strategy": strategy,
                "query_preview": raw[:120],
            },
        }
        messages = _run_stream(runnable, raw, cfg)
        last = messages[-1] if messages else None
        if isinstance(last, AIMessage):
            print("\nAssistant>", last.content)
        else:
            print("\nAssistant>", last)
        print("\nSources:\n" + _format_tool_sources(messages))
        print(
            f"\nLangSmith project: https://smith.langchain.com (project={settings.langsmith_project})"
        )


def main(argv: list[str] | None = None) -> None:
    _stdout_utf8()
    parser = argparse.ArgumentParser(description="Agentic RAG strategy lab")
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Fetch DEV.to articles into Chroma")
    p_ingest.add_argument("--rebuild", action="store_true")

    p_chat = sub.add_parser("chat", help="Interactive REPL")
    p_chat.add_argument(
        "--strategy",
        choices=("agent", "chain", "crag"),
        default="agent",
    )
    p_chat.add_argument("--thread", default=None, help="Existing thread id to resume")

    p_bench = sub.add_parser("bench", help="Run LangSmith evaluation")
    p_bench.add_argument(
        "--strategy",
        choices=("agent", "chain", "crag", "all"),
        default="chain",
    )
    p_bench.add_argument("--limit", type=int, default=None)

    args = parser.parse_args(argv)
    setup_logging(verbose=args.verbose)
    settings = load_settings()
    setup_tracing(settings)

    if args.command == "ingest":
        cmd_ingest(settings, args)
    elif args.command == "bench":
        cmd_bench(settings, args)
    elif args.command == "chat":
        cmd_chat(settings, args)


if __name__ == "__main__":
    main()
