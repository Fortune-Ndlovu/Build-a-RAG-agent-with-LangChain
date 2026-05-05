"""LangSmith dataset + evaluate runner for strategy comparison."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langsmith import Client
from langsmith.evaluation.evaluator import EvaluationResult
from tabulate import tabulate

from rag_lab.config import (
    Settings,
    get_chat_model,
    get_judge_model,
    get_vector_store,
    load_settings,
    setup_logging,
    setup_tracing,
)
from rag_lab.strategies import STRATEGIES

_log = logging.getLogger(__name__)

# Repo root: .../project/  (parent of rag_lab package)
REPO_ROOT = Path(__file__).resolve().parent.parent
QA_PATH = REPO_ROOT / "evals" / "qa.yaml"


def load_qa_examples() -> list[dict[str, Any]]:
    raw = yaml.safe_load(QA_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("evals/qa.yaml must be a list of examples")
    return raw


def ensure_dataset(client: Client, settings: Settings) -> str:
    name = settings.eval_dataset_name
    examples = []
    for row in load_qa_examples():
        eid = row.get("id", "")
        examples.append(
            {
                "inputs": {"question": row["question"]},
                "outputs": {
                    "expected_answer_substring": row.get("expected_answer_substring", ""),
                    "expected_source_url_contains": row.get(
                        "expected_source_url_contains", ""
                    ),
                    "category": row.get("category", ""),
                    "id": eid,
                },
            }
        )
    if client.has_dataset(dataset_name=name):
        _log.info("Dataset %s already exists in LangSmith.", name)
        return name
    client.create_dataset(dataset_name=name)
    client.create_examples(dataset_name=name, examples=examples)
    _log.info("Created dataset %s with %s examples.", name, len(examples))
    return name


def _extract_answer(result: dict[str, Any]) -> str:
    msgs = result.get("messages") or []
    if not msgs:
        return ""
    last = msgs[-1]
    if isinstance(last, AIMessage):
        c = last.content
        return c if isinstance(c, str) else str(c)
    return str(getattr(last, "content", last))


def collect_retrieved_urls(
    result: dict[str, Any],
    question: str,
    vector_store: Chroma,
    *,
    k: int = 8,
) -> list[str]:
    urls: list[str] = []
    for m in result.get("messages") or []:
        if isinstance(m, ToolMessage):
            art = getattr(m, "artifact", None)
            if art:
                for doc in art:
                    u = doc.metadata.get("url") if hasattr(doc, "metadata") else None
                    if u:
                        urls.append(str(u))
    if not urls:
        try:
            for doc in vector_store.similarity_search(question, k=k):
                u = doc.metadata.get("url")
                if u:
                    urls.append(str(u))
        except Exception as exc:
            _log.debug("similarity_search fallback failed: %s", exc)
    out: list[str] = []
    seen: set[str] = set()
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def run_benchmark(
    settings: Settings,
    *,
    strategy_name: str,
    limit: int | None = None,
) -> Any:
    client = Client()
    dataset_name = ensure_dataset(client, settings)

    model = get_chat_model(settings)
    vs = get_vector_store(settings)
    judge = get_judge_model(settings)

    builder = STRATEGIES[strategy_name]
    runnable = builder(model, vs, settings, checkpointer=None)

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        q = inputs["question"]
        result = runnable.invoke(
            {"messages": [HumanMessage(content=q)]},
            config={
                "tags": ["bench", f"strategy:{strategy_name}"],
                "metadata": {"strategy": strategy_name},
            },
        )
        answer = _extract_answer(result)
        urls = collect_retrieved_urls(result, q, vs)
        return {"answer": answer, "retrieved_urls": urls}

    def retrieval_recall(run: Any, example: Any) -> EvaluationResult:
        ref = (example.outputs or {}) if example else {}
        expected_url = ref.get("expected_source_url_contains") or ""
        if not expected_url:
            return EvaluationResult(key="retrieval_recall", score=1.0)
        outs = run.outputs or {}
        urls = outs.get("retrieved_urls") or []
        hit = any(expected_url in u for u in urls)
        return EvaluationResult(key="retrieval_recall", score=1.0 if hit else 0.0)

    def llm_judge_correctness(run: Any, example: Any) -> EvaluationResult:
        ref = (example.outputs or {}) if example else {}
        expected = ref.get("expected_answer_substring") or ""
        question = (example.inputs or {}).get("question", "") if example else ""
        answer = (run.outputs or {}).get("answer") or ""
        if not expected:
            return EvaluationResult(key="correctness", score=1.0)
        prompt = (
            f"Question: {question}\n"
            f'Expected substring (case-insensitive match OK): {expected!r}\n'
            f"Model answer:\n{answer}\n\n"
            "Does the answer correctly reflect the expected substring or an equivalent claim? "
            'Reply with exactly "yes" or "no".'
        )
        verdict = judge.invoke([HumanMessage(content=prompt)])
        text = (verdict.content or "").strip().lower()
        ok = text.startswith("y")
        return EvaluationResult(key="correctness", score=1.0 if ok else 0.0)

    evaluators = [retrieval_recall, llm_judge_correctness]
    if limit is not None:
        data: Any = list(
            client.list_examples(dataset_name=dataset_name, limit=limit)
        )
    else:
        data = dataset_name
    return client.evaluate(
        target,
        data=data,
        evaluators=evaluators,
        experiment_prefix=f"strategy:{strategy_name}",
        metadata={
            "strategy": strategy_name,
            "embed_model": settings.ollama_embed_model,
            "chat_model": settings.chat_model,
            "embedding_provider": settings.embedding_provider,
        },
        max_concurrency=0,
    )


def format_results_md(prefix: str, results_obj: Any) -> str:
    lines = [
        f"# Benchmark {prefix}",
        "",
        f"- Time (UTC): {datetime.now(timezone.utc).isoformat()}",
        "",
    ]
    rs = getattr(results_obj, "run_stats", None)
    if rs:
        lines.append("## Run stats")
        lines.append(f"- Total runs: {getattr(rs, 'total_runs', '?')}")
        lines.append(f"- Latency (median): {getattr(rs, 'latency_median', '?')}")
        lines.append(f"- Total tokens: {getattr(rs, 'total_tokens', '?')}")
        lines.append("")
    fs = getattr(results_obj, "feedback_stats", None)
    if fs:
        lines.append("## Feedback stats")
        lines.append(f"```json\n{fs}\n```")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Run LangSmith eval for one or all strategies.")
    p.add_argument(
        "--strategy",
        choices=("agent", "chain", "crag", "all"),
        default="chain",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    setup_logging(verbose=args.verbose)
    settings = load_settings()
    setup_tracing(settings)

    strategies = ["agent", "chain", "crag"] if args.strategy == "all" else [args.strategy]
    summaries: list[list[Any]] = []

    for name in strategies:
        _log.info("Running benchmark for strategy=%s", name)
        res = run_benchmark(settings, strategy_name=name, limit=args.limit)
        fs = getattr(res, "feedback_stats", {}) or {}
        rs = getattr(res, "run_stats", None)
        lat = getattr(rs, "latency_median", None) if rs else None
        tok = getattr(rs, "total_tokens", None) if rs else None
        summaries.append([name, fs, lat, tok])
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        out = REPO_ROOT / "evals" / f"results-{name}-{ts}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(format_results_md(name, res), encoding="utf-8")
        _log.info("Wrote %s", out)

    print(
        tabulate(
            summaries,
            headers=["strategy", "feedback_stats", "latency_median", "total_tokens"],
            tablefmt="github",
        )
    )


if __name__ == "__main__":
    main()
