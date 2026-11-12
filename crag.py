"""
Basic CRAG (Corrective Retrieval-Augmented Generation) with a local Ollama model.

Pipeline:
  1. Retrieve   - BM25 over a local document store (pure Python, no deps).
  2. Evaluate   - The LLM grades all retrieved docs in one call: relevant / ambiguous / irrelevant.
  3. Correct    - Decide an action from the grades:
                    CORRECT   (any doc relevant)      -> keep the relevant + ambiguous docs
                    INCORRECT (all docs irrelevant)   -> discard them, search the web
                    AMBIGUOUS (otherwise)             -> keep the ambiguous docs AND search the web
                  With --refine, kept docs are first condensed to key facts (slower).
  4. Generate   - The LLM answers from the corrected knowledge (streamed as it's written).

Setup:
    1. Install Ollama: https://ollama.com/download
    2. ollama pull qwen2.5:7b
    3. pip install -r requirements.txt

Usage:
    python crag.py "your question"                          # built-in sample corpus
    python crag.py "your question" --docs report.pdf        # answer from a PDF
    python crag.py --docs report.pdf                        # interactive Q&A about a PDF
    python crag.py "your question" --docs ./my_docs         # folder of .pdf/.txt/.md files
    python crag.py "your question" --docs report.pdf --no-web   # never use the web
    python crag.py "your question" --model qwen2.5:3b       # use another Ollama model
    python crag.py "your question" --docs report.pdf --refine   # condense to facts first
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import ollama
from ddgs import DDGS

# Override with --model or the CRAG_MODEL environment variable.
MODEL = os.environ.get("CRAG_MODEL", "qwen2.5:7b")


# --------------------------------------------------------------------------- #
# 1. Retriever (BM25)
# --------------------------------------------------------------------------- #

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass
class Doc:
    id: str
    text: str


class BM25Retriever:
    def __init__(self, docs: list[Doc], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(d.text)) for d in docs]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avgdl = sum(self.lens) / max(len(docs), 1)
        df = Counter(term for tf in self.tfs for term in tf)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 3) -> list[Doc]:
        q = tokenize(query)
        scores = []
        for i, tf in enumerate(self.tfs):
            s = 0.0
            for t in q:
                if t not in tf:
                    continue
                f = tf[t]
                denom = f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avgdl)
                s += self.idf[t] * f * (self.k1 + 1) / denom
            scores.append((s, i))
        scores.sort(reverse=True)
        return [self.docs[i] for s, i in scores[:k] if s > 0]


# --------------------------------------------------------------------------- #
# LLM helper (Ollama)
# --------------------------------------------------------------------------- #

KEEP_ALIVE = "30m"  # keep the model loaded between questions (Ollama's default is 5 min)


def _chat(messages: list[dict], *, schema: dict | None, temperature: float, stream: bool):
    try:
        return ollama.chat(
            model=MODEL,
            messages=messages,
            format=schema,  # JSON schema -> Ollama constrains the output to it
            options={"temperature": temperature},
            keep_alive=KEEP_ALIVE,
            stream=stream,
        )
    except ollama.ResponseError as e:
        if e.status_code == 404:
            sys.exit(f"Model '{MODEL}' not found. Run:  ollama pull {MODEL}")
        raise
    except ConnectionError:
        sys.exit("Can't reach Ollama. Make sure it's installed and running (http://localhost:11434).")


def _messages(prompt: str, system: str) -> list[dict]:
    msgs = [{"role": "system", "content": system}] if system else []
    return msgs + [{"role": "user", "content": prompt}]


def llm(prompt: str, *, system: str = "", schema: dict | None = None,
        temperature: float = 0.0) -> str:
    resp = _chat(_messages(prompt, system), schema=schema, temperature=temperature, stream=False)
    return (resp.message.content or "").strip()


def llm_stream(prompt: str, *, system: str = "", temperature: float = 0.0) -> Iterator[str]:
    """Yield the reply piece by piece as the model writes it."""
    for part in _chat(_messages(prompt, system), schema=None, temperature=temperature, stream=True):
        if part.message.content:
            yield part.message.content


# --------------------------------------------------------------------------- #
# 2. Retrieval evaluator
# --------------------------------------------------------------------------- #

GRADES = ("relevant", "ambiguous", "irrelevant")


def grade_docs(question: str, docs: list[Doc]) -> dict[str, str]:
    """Grade all retrieved docs in ONE model call (much faster than one call per doc)."""
    if not docs:
        return {}
    numbered = "\n\n".join(f"Document {i}:\n{d.text}" for i, d in enumerate(docs, start=1))
    schema = {
        "type": "object",
        "properties": {
            "grades": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "document": {"type": "integer"},
                        "grade": {"type": "string", "enum": list(GRADES)},
                    },
                    "required": ["document", "grade"],
                },
            }
        },
        "required": ["grades"],
    }
    out = llm(
        f"Question:\n{question}\n\n{numbered}\n\n"
        "For EACH document, grade whether it contains information that helps answer the question:\n"
        "- relevant: it directly contains (part of) the answer\n"
        "- ambiguous: it is on-topic but may not contain the answer\n"
        "- irrelevant: it does not help answer the question\n"
        f"Return one entry for each of the {len(docs)} documents, e.g. "
        '{"grades": [{"document": 1, "grade": "relevant"}, {"document": 2, "grade": "irrelevant"}]}',
        system="You are a strict retrieval evaluator.",
        schema=schema,
    )
    try:
        entries = json.loads(out).get("grades", [])
    except (json.JSONDecodeError, AttributeError):
        entries = []
    by_num = {e.get("document"): e.get("grade") for e in entries if isinstance(e, dict)}
    # Small models occasionally skip a document or misbehave; treat those as ambiguous.
    return {d.id: (by_num.get(i) if by_num.get(i) in GRADES else "ambiguous")
            for i, d in enumerate(docs, start=1)}


# --------------------------------------------------------------------------- #
# 3. Corrective actions
# --------------------------------------------------------------------------- #

def as_knowledge(docs: list[Doc]) -> str:
    """Pass chunks through unchanged, labelled with their source ids."""
    return "\n\n".join(f"[{d.id}]\n{d.text}" for d in docs)


def refine(question: str, docs: list[Doc]) -> str:
    """Knowledge refinement: keep only the facts that matter for the question (one extra call)."""
    if not docs:
        return ""
    out = llm(
        f"Question:\n{question}\n\nDocuments:\n{as_knowledge(docs)}\n\n"
        "Copy out the facts from these documents that help answer the question.\n"
        "Write one bullet point per fact, in this exact format:\n"
        f"- <the fact, written as a full sentence> [{docs[0].id}]\n"
        "Use the id of the document the fact came from. Output only the bullet points. "
        "If nothing is relevant, output NONE.",
    )
    return "" if out.upper().startswith("NONE") else out


def rewrite_query(question: str) -> str:
    """Turn the question into a short keyword query for web search."""
    out = llm(
        f"Rewrite this question as a short web search query (keywords only, max 10 words). "
        f"Output only the query.\n\nQuestion: {question}"
    )
    return out.strip().strip('"').splitlines()[0] if out else question


def web_search(question: str, max_results: int = 5, refine_results: bool = False) -> str:
    """External knowledge: DuckDuckGo search results, optionally refined."""
    query = rewrite_query(question)
    print(f"[web]      searching: {query}")
    try:
        results = DDGS().text(query, max_results=max_results)
    except Exception as e:  # network errors, rate limits
        print(f"[web]      search failed: {e}")
        return ""
    if not results:
        return ""
    web_docs = [Doc(r.get("href", f"web{i}"), f"{r.get('title', '')}\n{r.get('body', '')}")
                for i, r in enumerate(results)]
    return refine(question, web_docs) if refine_results else as_knowledge(web_docs)


# --------------------------------------------------------------------------- #
# 4. Generation + the full pipeline
# --------------------------------------------------------------------------- #

def _answer_prompt(question: str, knowledge: str) -> str:
    # Use a real source label as the example, so small models don't copy a made-up one.
    first_label = re.search(r"^\[[^\]\n]+\]", knowledge, re.M)
    example = first_label.group(0) if first_label else "[source]"
    return (f"Knowledge:\n{knowledge or '(none)'}\n\nQuestion:\n{question}\n\n"
            "Answer the question using only the knowledge above. After each point, cite where it "
            "came from by copying the label in square brackets that appears above that passage, "
            f"exactly as written (for example {example}). Never invent a label. "
            "If the knowledge is insufficient, say so.")


ANSWER_SYSTEM = "You are a helpful assistant that answers grounded in the provided knowledge."


def generate_stream(question: str, knowledge: str) -> Iterator[str]:
    return llm_stream(_answer_prompt(question, knowledge), system=ANSWER_SYSTEM, temperature=0.2)


def generate(question: str, knowledge: str) -> str:
    return "".join(generate_stream(question, knowledge)).strip()


@dataclass
class CRAGResult:
    answer: str
    action: str
    grades: dict[str, str] = field(default_factory=dict)
    knowledge: str = ""


def prepare(question: str, retriever: BM25Retriever, k: int = 4, verbose: bool = True,
            use_web: bool = True, refine_docs: bool = False) -> tuple[str, dict[str, str], str]:
    """Steps 1-3 (retrieve, evaluate, correct). Returns (action, grades, knowledge)."""
    log = print if verbose else (lambda *a, **kw: None)

    docs = retriever.search(question, k=k)
    log(f"[retrieve] {len(docs)} docs: {[d.id for d in docs]}")

    grades = grade_docs(question, docs)
    log(f"[evaluate] {grades}")

    relevant = [d for d in docs if grades[d.id] == "relevant"]
    ambiguous = [d for d in docs if grades[d.id] == "ambiguous"]

    if relevant:
        action = "CORRECT"
    elif ambiguous:
        action = "AMBIGUOUS"
    else:
        action = "INCORRECT"
    log(f"[action]   {action}")

    # Keep ambiguous chunks alongside relevant ones: small graders are often over-cautious.
    use = relevant + ambiguous
    parts = [refine(question, use) if refine_docs else as_knowledge(use)]
    if action != "CORRECT" and use_web:
        parts.append(web_search(question, refine_results=refine_docs))

    knowledge = "\n\n".join(p for p in parts if p)
    log(f"[knowledge]\n{knowledge}\n")
    return action, grades, knowledge


def crag(question: str, retriever: BM25Retriever, k: int = 4, verbose: bool = True,
         use_web: bool = True, refine_docs: bool = False) -> CRAGResult:
    action, grades, knowledge = prepare(question, retriever, k, verbose, use_web, refine_docs)
    answer = generate(question, knowledge)
    return CRAGResult(answer=answer, action=action, grades=grades, knowledge=knowledge)


# --------------------------------------------------------------------------- #
# Sample corpus + CLI
# --------------------------------------------------------------------------- #

SAMPLE_DOCS = [
    Doc("python", "Python is a high-level programming language created by Guido van Rossum "
                  "and first released in 1991. It emphasizes code readability."),
    Doc("rag", "Retrieval-Augmented Generation (RAG) combines a retriever that fetches documents "
               "with a generator model that writes answers grounded in those documents."),
    Doc("crag", "Corrective RAG (CRAG) adds a lightweight retrieval evaluator that scores retrieved "
                "documents. Based on confidence it triggers Correct, Incorrect, or Ambiguous actions; "
                "incorrect retrievals are replaced by web search results."),
    Doc("bm25", "BM25 is a ranking function used by search engines to estimate the relevance of "
                "documents to a query, based on term frequency and inverse document frequency."),
    Doc("eiffel", "The Eiffel Tower is a wrought-iron lattice tower in Paris, France, completed in 1889."),
]


def chunk_text(text: str, size: int = 1000, overlap: int = 200) -> list[str]:
    """Split long text into overlapping chunks, preferring to break at sentence ends."""
    text = re.sub(r"\s+", " ", text).strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(". ", start + size // 2, end)
            if cut != -1:
                end = cut + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def load_pdf(path: Path) -> list[Doc]:
    """Read a PDF page by page and chunk it. Chunk ids look like 'report.pdf p3#1'."""
    from pypdf import PdfReader  # pip install pypdf

    docs = []
    for page_no, page in enumerate(PdfReader(path).pages, start=1):
        text = page.extract_text() or ""
        for i, chunk in enumerate(chunk_text(text), start=1):
            docs.append(Doc(f"{path.name} p{page_no}#{i}", chunk))
    if not docs:
        print(f"[warn] no text found in {path.name} (scanned PDFs need OCR first)")
    return docs


def load_path(path_str: str) -> list[Doc]:
    """Load a single .pdf/.txt/.md file, or every such file in a folder."""
    path = Path(path_str)
    if not path.exists():
        sys.exit(f"Not found: {path}")
    files = [path] if path.is_file() else sorted(path.rglob("*"))
    docs = []
    for p in files:
        suffix = p.suffix.lower()
        if suffix == ".pdf":
            docs.extend(load_pdf(p))
        elif suffix in {".txt", ".md"}:
            text = p.read_text(encoding="utf-8", errors="ignore")
            for i, chunk in enumerate(chunk_text(text), start=1):
                docs.append(Doc(f"{p.name}#{i}", chunk))
    return docs


def main() -> None:
    global MODEL
    ap = argparse.ArgumentParser(description="Basic Corrective RAG with Ollama")
    ap.add_argument("question", nargs="?",
                    help="question to ask (omit to start an interactive Q&A session)")
    ap.add_argument("--docs", nargs="+", help="one or more .pdf/.txt/.md files or folders "
                                              "(default: built-in sample corpus)")
    ap.add_argument("--model", default=MODEL, help=f"Ollama model (default: {MODEL})")
    ap.add_argument("-k", type=int, default=4, help="chunks to retrieve per question")
    ap.add_argument("--no-web", action="store_true",
                    help="never search the web; answer only from your documents")
    ap.add_argument("--refine", action="store_true",
                    help="condense retrieved text to key facts before answering (slower)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="also print the knowledge passed to the answer step")
    args = ap.parse_args()
    MODEL = args.model

    docs = [d for p in args.docs for d in load_path(p)] if args.docs else SAMPLE_DOCS
    if not docs:
        sys.exit("No documents found.")
    print(f"[load]     {len(docs)} chunks indexed")
    retriever = BM25Retriever(docs)

    def ask(q: str) -> None:
        action, grades, knowledge = prepare(q, retriever, k=args.k, verbose=False,
                                            use_web=not args.no_web, refine_docs=args.refine)
        print(f"[evaluate] {grades}\n[action]   {action}")
        if args.verbose:
            print(f"[knowledge]\n{knowledge}\n")
        print("=" * 60)
        for piece in generate_stream(q, knowledge):
            print(piece, end="", flush=True)
        print("\n" + "=" * 60)

    if args.question:
        ask(args.question)
        return
    print("Ask questions about your documents (blank line or Ctrl+C to quit).")
    try:
        while q := input("\n> ").strip():
            ask(q)
    except (KeyboardInterrupt, EOFError):
        pass


if __name__ == "__main__":
    main()
