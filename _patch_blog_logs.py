"""One-off: rebuild ## Exact log output section from terminals/2.txt."""
from pathlib import Path

terminal = Path(
    r"C:\Users\ndlov\.cursor\projects\c-Users-ndlov-Documents-Build-a-RAG-agent-with-LangChain\terminals\2.txt"
)
blog = Path(__file__).resolve().parent / "BLOG_DRAFT.md"

raw = terminal.read_text(encoding="utf-8").splitlines()
sep = [i for i, line in enumerate(raw) if line.strip() == "---"]
start = sep[1] + 1 if len(sep) > 1 else 0
body = raw[start:]

try:
    i_chat = next(
        i
        for i, line in enumerate(body)
        if "python rag_agent.py chat --strategy agent" in line
    )
except StopIteration:
    i_chat = 0

try:
    i_chain = next(
        i for i, line in enumerate(body) if line.strip() == "You> /strategy chain"
    )
except StopIteration:
    i_chain = len(body)

try:
    i_crag = next(
        i
        for i, line in enumerate(body)
        if line.startswith("You>") and "/strategy crag" in line
    )
except StopIteration:
    i_crag = len(body)

setup = body[:i_chat]
agent = body[i_chat:i_chain]
chain = body[i_chain:i_crag]

try:
    i_bench_typo = next(
        i
        for i, line in enumerate(body)
        if "You> python rag_agent.py bench" in line
    )
    crag = body[i_crag:i_bench_typo]
    bench_typo_lines = body[i_bench_typo : i_bench_typo + 16]
except StopIteration:
    crag = body[i_crag:]
    bench_typo_lines = []


def fence(title: str, lines: list[str]) -> str:
    """Use 4-backtick fences so nested ```bash / ```python in transcripts don't break Markdown."""
    content = "\n".join(lines)
    return f"### {title}\n\n````text\n{content}\n````\n\n"


section_parts = [
    "## Exact log output (full transcript from one session)\n\n",
    "Below is the **complete** terminal output from my machine: first pulling models and ingesting the corpus, ",
    "then running the **same three questions** under each strategy (`agent`, then `chain`, then `crag`) in one REPL session. ",
    "I did not trim the Assistant replies or the `httpx` lines so you can see exactly what showed up.\n\n",
    fence("Setup: Ollama pulls + ingest", setup),
    fence("First strategy: `agent` (tool-calling)", agent),
    fence("Second strategy: `chain` (always-retrieve)", chain),
    fence("Third strategy: `crag` (corrective RAG)", crag),
    "### What you can already see in the logs (before any benchmark table)\n\n",
    "The three strategies are **not** the same execution pattern.\n\n",
    "- **`agent`:** You get multiple `POST .../api/chat` calls per question because the model decides what to do next, ",
    "and `POST .../api/embed` when it calls `retrieve_context`. The **Sources** panel lists real `ToolMessage` artifacts when retrieval ran — ",
    "and you can see noise too (for example, one retrieved hit points at an unrelated article URL in the CNN question). ",
    "That is normal top-k behavior; benchmarks quantify whether it hurts answers.\n\n",
    "- **`chain`:** Each question shows a simple rhythm: **one embed** (similarity search) and **one chat** completion. ",
    "There are no tool-call artifacts, so the CLI prints the “no retrieve_context tool calls” line and points you to LangSmith for internal retrieval. ",
    "Compare the **link** in the conda answer to the `agent` run: the URL slug can differ while the prose still sounds confident.\n\n",
    "- **`crag`:** Count the HTTP lines on the first question: extra `/chat` calls align with **grade** (and sometimes **rewrite**) before **generate**, ",
    "and you may see **two** embed rounds when retrieval runs again after a rewrite. CRAG can spend more time per question than `chain`.\n\n",
    "### Don’t run benchmarks from inside the chat REPL\n\n",
    "After the CRAG block below, I accidentally pasted `python rag_agent.py bench --strategy all` **as chat input**. ",
    "The model answered from posts — not what you want for LangSmith eval. Exit with `/quit`, then run bench from the shell; ",
    "**full benchmark logs go in the next section** once your run finishes.\n\n",
]

if bench_typo_lines:
    section_parts.append(
        fence(
            "What not to do (mistyped `bench` inside the REPL)",
            bench_typo_lines,
        )
    )

section_parts.extend(
    [
        "Optional screenshots for the published post: ",
        "`{{SCREENSHOT:langsmith_trace_chat}}`, `{{SCREENSHOT:langsmith_experiment_compare}}`.\n\n",
    ]
)

section = "".join(section_parts)

md = blog.read_text(encoding="utf-8")
start_marker = "## Exact log output"
end_marker = "## Benchmarks"
i0 = md.find(start_marker)
i1 = md.find(end_marker)
if i0 == -1 or i1 == -1 or i0 >= i1:
    raise SystemExit(f"markers not found: {i0=} {i1=}")

new_md = md[:i0] + section + "\n---\n\n" + md[i1:]
blog.write_text(new_md, encoding="utf-8")
print("Wrote section:", len(section), "chars")
