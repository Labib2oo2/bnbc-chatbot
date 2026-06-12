#!/usr/bin/env python3
"""BNBC AI Chatbot - Terminal UI (TUI)
RAG over Bangladesh National Building Code 2020 using BM25 retrieval + Claude API.
"""

import os
import pickle
import re
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rank_bm25 import BM25Okapi
import anthropic

console = Console()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def tok(s):
    return re.findall(r"[a-zA-Z0-9]+", s.lower())


def load_data():
    with open(os.path.join(BASE_DIR, "chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)
    with open(os.path.join(BASE_DIR, "bm25.pkl"), "rb") as f:
        bm25 = pickle.load(f)
    return chunks, bm25


def retrieve(query, chunks, bm25, k=10):
    scores = bm25.get_scores(tok(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [(chunks[i], scores[i]) for i in ranked if scores[i] > 0]


def build_context(results):
    parts = []
    for c, score in results:
        sec = c["section"] or "N/A"
        parts.append(f"[Page {c['page']} | Section {sec}]\n{c['text']}")
    return "\n\n---\n\n".join(parts)


SYSTEM_PROMPT = """You are an expert assistant on the Bangladesh National Building Code (BNBC) 2020.
Answer the user's question ONLY using the provided context excerpts from BNBC.
Always cite the relevant Section number(s) and Page number(s) in your answer.
If the answer is in Bangla in the question, you may answer in Bangla, otherwise answer in the language of the question.
If the context does not contain enough information to answer confidently, say so clearly and suggest which BNBC part/chapter might be relevant instead of guessing.
Be concise and precise, especially with numeric requirements (widths, heights, ratios, loads, etc.)."""


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[red]ANTHROPIC_API_KEY environment variable not set.[/red]")
        console.print("Run: export ANTHROPIC_API_KEY=your_key_here")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    console.print(Panel.fit(
        "[bold cyan]BNBC AI Chatbot[/bold cyan]\n"
        "Bangladesh National Building Code 2020 — RAG Assistant\n"
        "[dim]Type your question (English or Bangla). Type 'exit' or 'quit' to leave.[/dim]",
        border_style="cyan"
    ))

    console.print("[dim]Loading index...[/dim]")
    chunks, bm25 = load_data()
    console.print(f"[green]Loaded {len(chunks)} indexed sections from BNBC 2020.[/green]\n")

    history = []

    while True:
        try:
            query = Prompt.ask("[bold yellow]আপনার প্রশ্ন[/bold yellow]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[cyan]বিদায়![/cyan]")
            break

        if not query.strip():
            continue
        if query.strip().lower() in ("exit", "quit", "q", "বের"):
            console.print("[cyan]বিদায়![/cyan]")
            break

        with console.status("[bold green]খুঁজছি ও উত্তর তৈরি করছি..."):
            results = retrieve(query, chunks, bm25, k=10)
            context = build_context(results)

            user_msg = f"Context from BNBC 2020:\n\n{context}\n\n---\n\nQuestion: {query}"

            try:
                response = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=1200,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_msg}],
                )
                answer = response.content[0].text
            except Exception as e:
                console.print(f"[red]API error: {e}[/red]")
                continue

        console.print(Panel(Markdown(answer), title="[bold green]উত্তর", border_style="green"))

        with console.status("", spinner="dots") if False else _nullcontext():
            pass

        if results:
            srcs = ", ".join(
                f"Sec {c['section'] or '-'} (p.{c['page']})" for c, _ in results[:4]
            )
            console.print(f"[dim]📚 সূত্র: {srcs}[/dim]\n")
        else:
            console.print("[dim]📚 কোনো সরাসরি মিল পাওয়া যায়নি — Claude সাধারণ জ্ঞান থেকে উত্তর দিয়েছে।[/dim]\n")


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    main()
