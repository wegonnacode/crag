"""
Basic CRAG (Corrective Retrieval-Augmented Generation) with a local Ollama model.

Pipeline:
  1. Retrieve   - BM25 over a local document store (pure Python, no deps).
  2. Evaluate   - The LLM grades each retrieved doc: relevant / ambiguous / irrelevant.
  3. Correct    - Decide an action from the grades:
                    CORRECT   (any doc relevant)      -> refine the relevant docs
                    INCORRECT (all docs irrelevant)   -> discard them, search the web
                    AMBIGUOUS (otherwise)             -> refine docs AND search the web
  4. Generate   - The LLM answers from the corrected knowledge.

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

def llm(prompt: str, *, system: str = "", schema: dict | None = None,
        temperature: float = 0.0) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        resp = ollama.chat(
            model=MODEL,
            messages=messages,
            format=schema,  # JSON schema -> Ollama constrains the output to it
            options={"temperature": temperature},
        )
    except ollama.ResponseError as e:
        if e.status_code == 404:
            sys.exit(f"Model '{MODEL}' not found. Run:  ollama pull {MODEL}")
        raise
    except ConnectionError:
        sys.exit("Can't reach Ollama. Make sure it's installed and running (http://localhost:11434).")
    return resp.message.content.strip()


# --------------------------------------------------------------------------- #
# 2. Retrieval evaluator
# --------------------------------------------------------------------------- #

GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "grade": {"type": "string", "enum": ["relevant", "ambiguous", "irrelevant"]},
        "reason": {"type": "string"},
    },
    "required": ["grade", "reason"],
}


def grade_doc(question: str, doc: Doc) -> str:
    out = llm(
        f"Question:\n{question}\n\nDocument:\n{doc.text}\n\n"
        "Grade whether this document contains information that helps answer the question.\n"
        "- relevant: it directly contains (part of) the answer\n"
        "- ambiguous: it is on-topic but may not contain the answer\n"
        "- irrelevant: it does not help answer the question\n"
        'Respond in JSON: {"grade": "...", "reason": "..."}',
        system="You are a strict retrieval evaluator.",
        schema=GRADE_SCHEMA,
    )
    try:
        grade = json.loads(out).get("grade", "")
    except json.JSONDecodeError:
        grade = ""
    # Small models occasionally misbehave; treat anything unexpected as ambiguous.
    return grade if grade in {"relevant", "ambiguous", "irrelevant"} else "ambiguous"


# --------------------------------------------------------------------------- #
# 3. Corrective actions
# --------------------------------------------------------------------------- #

def refine(question: str, docs: list[Doc]) -> str:
    """Knowledge refinement: keep only the facts that matter for the question."""
    if not docs:
        return ""
    joined = "\n\n".join(f"[{d.id}]\n{d.text}" for d in docs)
    out = llm(
        f"Question:\n{question}\n\nDocuments:\n{joined}\n\n"
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


def web_search(question: str, max_results: int = 5) -> str:
    """External knowledge: DuckDuckGo search, then refine the snippets."""
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
    return refine(question, web_docs)


# --------------------------------------------------------------------------- #
# 4. Generation + the full pipeline
# --------------------------------------------------------------------------- #

def generate(question: str, knowledge: str) -> str:
    return llm(
        f"Knowledge:\n{knowledge or '(none)'}\n\nQuestion:\n{question}\n\n"
        "Answer the question using only the knowledge above. Cite sources in brackets. "
        "If the knowledge is insufficient, say so.",
        system="You are a helpful assistant that answers grounded in the provided knowledge.",
        temperature=0.2,
    )


@dataclass
class CRAGResult:
    answer: str
    action: str
    grades: dict[str, str] = field(default_factory=dict)
    knowledge: str = ""


def crag(question: str, retriever: BM25Retriever, k: int = 4, verbose: bool = True,
         use_web: bool = True) -> CRAGResult:
    log = print if verbose else (lambda *a, **kw: None)

    docs = retriever.search(question, k=k)
    log(f"[retrieve] {len(docs)} docs: {[d.id for d in docs]}")

    grades = {d.id: grade_doc(question, d) for d in docs}
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

    parts = []
    if action == "CORRECT":
        parts.append(refine(question, relevant))
    elif action == "AMBIGUOUS":
        parts.append(refine(question, ambiguous))
        if use_web:
            parts.append(web_search(question))
    elif use_web:
        parts.append(web_search(question))

    knowledge = "\n\n".join(p for p in parts if p)
    log(f"[knowledge]\n{knowledge}\n")

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
    ap.add_argument("--docs", help="a .pdf/.txt/.md file, or a folder of them "
                                   "(default: built-in sample corpus)")
    ap.add_argument("--model", default=MODEL, help=f"Ollama model (default: {MODEL})")
    ap.add_argument("-k", type=int, default=4, help="chunks to retrieve per question")
    ap.add_argument("--no-web", action="store_true",
                    help="never search the web; answer only from your documents")
    args = ap.parse_args()
    MODEL = args.model

    docs = load_path(args.docs) if args.docs else SAMPLE_DOCS
    if not docs:
        sys.exit("No documents found.")
    print(f"[load]     {len(docs)} chunks indexed")
    retriever = BM25Retriever(docs)

    def ask(q: str) -> None:
        result = crag(q, retriever, k=args.k, use_web=not args.no_web)
        print("=" * 60)
        print(result.answer)
        print("=" * 60)

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
