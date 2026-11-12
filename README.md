# CRAG — Corrective RAG with a local LLM

A small, single-file implementation of **Corrective Retrieval-Augmented Generation (CRAG)** that runs fully on your machine with [Ollama](https://ollama.com). Ask questions about your own PDFs, text or Markdown files; when your documents don't contain the answer, it falls back to a web search.

## How it works

1. **Retrieve.** BM25 keyword search finds the chunks of your documents that best match the question.
2. **Evaluate.** The LLM grades each chunk as `relevant`, `ambiguous` or `irrelevant`.
3. **Correct.** The grades decide the next step:
   - **CORRECT** (at least one relevant chunk): keep only the facts that answer the question.
   - **AMBIGUOUS** (nothing clearly relevant, something on-topic): use those facts *and* search the web.
   - **INCORRECT** (nothing useful): discard the chunks and search the web (DuckDuckGo).
4. **Generate.** The LLM answers from that knowledge and cites its sources (e.g. `[report.pdf p3#1]`, meaning page 3, chunk 1).

## Setup

1. Install Ollama: https://ollama.com/download
2. Pull a model (the default is Qwen 2.5 7B, ~4.7 GB):
   ```bash
   ollama pull qwen2.5:7b
   ```
3. Install the Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

```bash
# Ask a question about a PDF
python crag.py "How many days of paid leave do staff get?" --docs handbook.pdf

# Interactive Q&A about a PDF (leave out the question)
python crag.py --docs handbook.pdf

# A whole folder of .pdf / .txt / .md files
python crag.py "your question" --docs ./my_docs

# Answer only from your documents, never the web
python crag.py "your question" --docs handbook.pdf --no-web

# Use a smaller, faster model
python crag.py "your question" --docs handbook.pdf --model qwen2.5:3b
```

Without `--docs`, it uses a small built-in sample corpus.

| Option | Meaning |
|---|---|
| `--docs PATH` | A `.pdf`, `.txt` or `.md` file, or a folder of them |
| `--model NAME` | Ollama model to use (default `qwen2.5:7b`, or set `CRAG_MODEL`) |
| `-k N` | Chunks retrieved per question (default 4) |
| `--no-web` | Never fall back to web search |

## Notes

- Scanned PDFs (images of text) have no extractable text; run OCR on them first.
- On a machine without a dedicated GPU, expect 1–2 minutes per question with the 7B model.
- Retrieval is keyword-based (BM25), so questions that use the same words as the document work best.
