import os
import pickle
import re

import streamlit as st
from rank_bm25 import BM25Okapi
from groq import Groq

st.set_page_config(page_title="BNBC AI Chatbot", page_icon="🏗️", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def tok(s):
    return re.findall(r"[a-zA-Z0-9]+", s.lower())


@st.cache_resource
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
If the question is in Bangla, answer in Bangla; otherwise answer in the language of the question.
If the context does not contain enough information to answer confidently, say so clearly and suggest which BNBC part/chapter might be relevant instead of guessing.
Be concise and precise, especially with numeric requirements (widths, heights, ratios, loads, etc.)."""


st.sidebar.title("🏗️ BNBC AI Chatbot")
st.sidebar.markdown(
    "Bangladesh National Building Code 2020-এর উপর একটি RAG-based সহায়ক।\n\n"
    "প্রশ্ন করো (বাংলা/English), উত্তর সাথে Section ও Page reference পাবে।\n\n"
    "**Powered by Groq (Free) — Llama 3.3 70B**"
)

api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
if not api_key:
    api_key = st.sidebar.text_input("Groq API Key (console.groq.com — free)", type="password")

if st.sidebar.button("🗑️ Clear chat"):
    st.session_state.messages = []

st.title("BNBC AI সহায়ক")

chunks, bm25 = load_data()
st.sidebar.success(f"{len(chunks)} sections ইনডেক্স করা আছে")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption("📚 সূত্র: " + msg["sources"])

query = st.chat_input("আপনার প্রশ্ন লিখুন... (যেমন: ৬ তলা ভবনে staircase-এর minimum width কত?)")

if query:
    if not api_key:
        st.error("দয়া করে sidebar-এ Groq API Key দিন (console.groq.com থেকে ফ্রি)।")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("খুঁজছি ও উত্তর তৈরি করছি..."):
            results = retrieve(query, chunks, bm25, k=10)
            context = build_context(results)
            user_msg = f"Context from BNBC 2020:\n\n{context}\n\n---\n\nQuestion: {query}"

            client = Groq(api_key=api_key)
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    max_tokens=1200,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                )
                answer = response.choices[0].message.content
            except Exception as e:
                answer = f"⚠️ API error: {e}"

        st.markdown(answer)

        sources = ""
        if results:
            sources = ", ".join(
                f"Sec {c['section'] or '-'} (p.{c['page']})" for c, _ in results[:4]
            )
            st.caption("📚 সূত্র: " + sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
