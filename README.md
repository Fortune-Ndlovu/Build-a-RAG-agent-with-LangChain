# DEV.to RAG lab

This repo answers questions **using your own DEV.to articles** as the source. It downloads your public posts, stores searchable chunks on disk, and runs a chat assistant that can look up passages before replying. You can also **compare three answer styles** (agent vs. chain vs. corrective RAG) with automated scoring in LangSmith.

---

## What you need

- **Python 3.11+** (what this repo was tested with)
- **Optional but recommended:** [Ollama](https://ollama.com) on your machine for local embeddings and/or local chat (no embedding quota from cloud APIs)
- **API keys** (see step 2): LangSmith for traces and evals; Google if you use Gemini for chat, judging, or embeddings

---

## Step 1 — Virtual environment and install

From the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks scripts:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Step 2 — Configure `.env`

```powershell
Copy-Item .env.example .env
notepad .env
```

Fill in at least:

| Variable | What it’s for |
|----------|----------------|
| `LANGSMITH_API_KEY` | See every run and every benchmark in [LangSmith](https://smith.langchain.com) (create a key under Settings → API Keys). |
| `GOOGLE_API_KEY` | Needed if **any** of chat, judge, or embeddings use Google (Gemini). Skip only if you use Ollama for all three. |

Match how you want to run:

- **Local chat + local embeddings (common):** set `CHAT_PROVIDER=ollama`, `CHAT_MODEL=qwen2.5:7b`, `EMBEDDING_PROVIDER=ollama`, `EMBEDDING_MODEL=nomic-embed-text`, keep `OLLAMA_BASE_URL` as `http://localhost:11434`.
- **Gemini for chat:** set `CHAT_PROVIDER=google` and `CHAT_MODEL=gemini-2.5-flash-lite` (or another Gemini name you use).
- **Benchmarks use a separate “judge” model:** `JUDGE_PROVIDER` / `JUDGE_MODEL` — often left on Google for stronger grading while chat stays on Ollama.

Set `DEVTO_USERNAME` to your DEV.to username so the app pulls **your** posts.

`.env` is gitignored so keys stay on your machine.

---

## Step 3 — Ollama models (if you use Ollama)

Start Ollama, then pull what your `.env` references. Typical pair:

```powershell
ollama pull nomic-embed-text
ollama pull qwen2.5:7b
```

---

## Step 4 — Build the article index (run once, then again when you publish edits)

This fetches your DEV.to feed, splits articles into chunks, and writes vectors under `./.chroma`.

```powershell
python rag_agent.py ingest
```

To wipe the index and rebuild from scratch:

```powershell
python rag_agent.py ingest --rebuild
```

You should see a short summary (how many articles, how many chunks). If something fails, check that Ollama is running and that `EMBEDDING_*` matches what you pulled.

---

## Step 5 — Chat with your articles

Start the interactive session (default strategy is the **tool agent** — it can search more than once per question):

```powershell
python rag_agent.py chat --strategy agent
```

Try other strategies:

```powershell
python rag_agent.py chat --strategy chain
python rag_agent.py chat --strategy crag
```

### What to type

Ask anything that should be answered from your posts, for example:

- “What conda environment name does the ticket routing article recommend?”
- “What class name is used for the CNN in the image classification post?”
- “Summarize how the MNIST example sets up the model.”

### Slash commands (inside the chat loop)

| Command | What it does |
|---------|----------------|
| `/help` | Lists commands |
| `/strategy agent` / `chain` / `crag` | Switch how answers are produced without restarting |
| `/reset` | New conversation thread |
| `/rebuild` | Re-run ingest (same as `ingest --rebuild` from disk) |
| `/threads` | Lists recent saved thread IDs |
| `/resume <id>` | Continue an old thread |
| `/quit` | Exit |

After each reply, the app prints **sources** when the agent used the retrieval tool. Open **LangSmith** (same project name as `LANGSMITH_PROJECT`) to see the full trace: tool calls, retrieved text, and latency.

---

## Step 6 — Run the evaluation suite

**Important:** Benchmarks are **shell commands**, not chat messages. If you are inside `python rag_agent.py chat`, type **`/quit`** first (or press Ctrl+C), then run the commands below in PowerShell from the project folder. If you paste `python rag_agent.py bench ...` at the `You>` prompt, the assistant will try to answer it like a normal question.

Questions and expected checks live in `evals/qa.yaml`. The benchmark:

1. Sends each question through your chosen strategy.
2. Scores **retrieval** (did the right article URL show up in retrieved context) and **answer quality** (a judge model checks the answer against a short expected phrase).

Run one strategy:

```powershell
python rag_agent.py bench --strategy chain --limit 5
```

Run all three and compare:

```powershell
python rag_agent.py bench --strategy all
```

Or run the bench module directly:

```powershell
python bench.py --strategy chain --limit 5
```

Results tables print in the terminal; markdown summaries are written under `evals/results-*.md`. In LangSmith, open the **Experiments** view for your dataset (`EVAL_DATASET_NAME` in `.env`) to compare runs side by side.

Tune `evals/qa.yaml` if your posts use different wording than the expected substrings.

---

## Other ways to start the app

```powershell
python -m rag_lab chat --strategy agent
python -m rag_lab ingest
python -m rag_lab bench --strategy chain
```

Always run these from the **project root** so `.env` and `evals/` are found.

---

## Repo layout (short)

| Path | Purpose |
|------|--------|
| `rag_agent.py` | Main CLI entry |
| `rag_lab/` | App code (settings, ingest, tools, strategies, bench runner) |
| `evals/qa.yaml` | Benchmark questions |
| `.chroma/` | Local vector index (created after ingest; gitignored) |
| `.threads.sqlite` | Chat memory for resumed threads (gitignored) |

---

## Troubleshooting

- **`Unable to infer model provider`** — For Ollama chat you must set `CHAT_PROVIDER=ollama` and use a plain model tag like `qwen2.5:7b`, not only the model name without provider.
- **Embedding / 429 errors from Google** — Switch `EMBEDDING_PROVIDER=ollama` and use `nomic-embed-text` with Ollama running.
- **Empty or wrong answers** — Re-run ingest; confirm `DEVTO_USERNAME` is correct; try `/rebuild` in chat or `ingest --rebuild`.
