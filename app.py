"""
Web interface for the CRAG system (Gradio).

    python app.py              # http://localhost:7860 (this computer only)
    python app.py --lan        # also reachable from other devices on your Wi-Fi
    python app.py --share      # temporary public https link anyone can open

Optional login (recommended with --share):
    set CRAG_PASSWORD=...      (PowerShell: $env:CRAG_PASSWORD = "...")
    set CRAG_USER=...          (default user name: "user")
"""

from __future__ import annotations

import argparse
import os

import gradio as gr

import crag

def installed_models() -> list[str]:
    """Models already pulled into Ollama, falling back to Qwen 2.5 sizes if Ollama is offline."""
    try:
        names = sorted(m.model for m in crag.ollama.list().models)
    except Exception:
        names = []
    return names or ["qwen2.5:7b", "qwen2.5:3b", "qwen2.5:14b"]


MODELS = installed_models()
DEFAULT_MODEL = crag.MODEL if crag.MODEL in MODELS else MODELS[0]
SAMPLE_RETRIEVER = crag.BM25Retriever(crag.SAMPLE_DOCS)


def build_index(files):
    """Load uploaded files into a retriever (stored per browser session)."""
    if not files:
        return None, "No files uploaded. Using the built-in sample documents."
    docs, names = [], []
    for path in files:
        try:
            docs.extend(crag.load_path(path))
            names.append(os.path.basename(path))
        except Exception as e:  # corrupt or unreadable file
            return None, f"Could not read {os.path.basename(path)}: {e}"
    if not docs:
        return None, "No text found in those files (scanned PDFs need OCR first)."
    return crag.BM25Retriever(docs), f"Indexed {len(docs)} chunks from: {', '.join(names)}"


def respond(question, history, retriever, use_web, model):
    question = (question or "").strip()
    if not question:
        yield "", history
        return
    history = history + [{"role": "user", "content": question}]
    yield "", history + [{"role": "assistant", "content": "_Thinking… (this can take a minute)_"}]

    crag.MODEL = model
    try:
        result = crag.crag(question, retriever or SAMPLE_RETRIEVER, verbose=False, use_web=use_web)
        grades = ", ".join(f"`{doc}`: {g}" for doc, g in result.grades.items()) or "none retrieved"
        answer = (f"{result.answer}\n\n---\n"
                  f"<sub>Path: **{result.action}** · Retrieved: {grades}</sub>")
    except SystemExit as e:  # crag.llm exits with a readable message when Ollama is unreachable
        answer = f"⚠️ {e}"
    except Exception as e:
        answer = f"⚠️ Error: {e}"

    yield "", history + [{"role": "assistant", "content": answer}]


with gr.Blocks(title="CRAG Q&A") as demo:
    gr.Markdown("# CRAG Q&A\nUpload PDFs, text or Markdown files, then ask questions about them. "
                "If your documents don't contain the answer, it can search the web.")
    retriever_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1, min_width=260):
            files = gr.File(label="Your documents", file_count="multiple",
                            file_types=[".pdf", ".txt", ".md"])
            status = gr.Markdown("No files uploaded. Using the built-in sample documents.")
            use_web = gr.Checkbox(value=True, label="Search the web when documents don't help")
            model = gr.Dropdown(MODELS, value=DEFAULT_MODEL, label="Ollama model",
                                allow_custom_value=True)
        with gr.Column(scale=3):
            chat = gr.Chatbot(height=520, label="Chat")
            with gr.Row():
                box = gr.Textbox(placeholder="Ask a question…", show_label=False, scale=5)
                send = gr.Button("Ask", variant="primary", scale=1)
            clear = gr.Button("Clear chat", size="sm")

    files.change(build_index, files, [retriever_state, status])
    for trigger in (box.submit, send.click):
        trigger(respond, [box, chat, retriever_state, use_web, model], [box, chat])
    clear.click(lambda: [], None, chat)


def main() -> None:
    ap = argparse.ArgumentParser(description="CRAG web interface")
    ap.add_argument("--share", action="store_true", help="create a temporary public link")
    ap.add_argument("--lan", action="store_true", help="listen on your local network")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    password = os.environ.get("CRAG_PASSWORD")
    auth = (os.environ.get("CRAG_USER", "user"), password) if password else None
    if args.share and not auth:
        print("Warning: public link without a password. Set CRAG_PASSWORD to require a login.")

    demo.queue(default_concurrency_limit=1)  # one question at a time: the LLM runs locally
    demo.launch(share=args.share, server_name="0.0.0.0" if args.lan else "127.0.0.1",
                server_port=args.port, auth=auth)


if __name__ == "__main__":
    main()
