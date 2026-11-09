"""Chat with your PDF — a RAG app built from scratch (no LangChain).

Pipeline: PDF -> text chunks -> embeddings -> FAISS index -> top-k chunks -> LLM answer with page citations.
"""
import os

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Model is read from an environment variable so it can be changed without editing code
# (Groq retired Llama 3.1 in Aug 2026; the app was migrated to GPT-OSS-20B).
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
CHUNK_SIZE = 500   # characters per chunk
OVERLAP = 50       # characters shared between neighboring chunks
TOP_K = 3          # chunks sent to the LLM for each question


@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embed_model = load_embed_model()
api_key = os.environ.get("GROQ_API_KEY") or st.secrets["GROQ_API_KEY"]
client = Groq(api_key=api_key)


def build_rag(uploaded_file):
    """Split the PDF into overlapping chunks (remembering each chunk's page) and index them in FAISS."""
    reader = PdfReader(uploaded_file)
    chunks = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        for j in range(0, len(text), CHUNK_SIZE - OVERLAP):
            piece = text[j:j + CHUNK_SIZE].strip()
            if piece:
                chunks.append({"text": piece, "page": i + 1})

    if not chunks:
        st.error("No readable text found in this PDF (it may be scanned images).")
        st.stop()

    embeddings = np.array(
        embed_model.encode([c["text"] for c in chunks]),
        dtype="float32",
    )

    if embeddings.ndim != 2:
        st.error(f"Embedding failed — got shape {embeddings.shape}.")
        st.stop()

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return chunks, index


def ask(question, chunks, index):
    """Find the most relevant chunks and ask the LLM to answer using only them."""
    q_embed = embed_model.encode([question])
    _, ids = index.search(np.array(q_embed, dtype="float32"), k=min(TOP_K, len(chunks)))
    context = "\n".join(chunks[i]["text"] for i in ids[0])
    pages = sorted(set(chunks[i]["page"] for i in ids[0]))
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": f"Answer using only this context:\n{context}"},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content, pages


st.title("Chat with your PDF")
st.caption("A RAG app built from scratch — no LangChain.")

uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded:
    if st.session_state.get("filename") != uploaded.name:
        with st.spinner("Reading and indexing your PDF..."):
            st.session_state.chunks, st.session_state.index = build_rag(uploaded)
            st.session_state.filename = uploaded.name
        st.success(f"Indexed {len(st.session_state.chunks)} chunks")

    question = st.text_input("Ask a question about the document")
    if question:
        with st.spinner("Thinking..."):
            answer, pages = ask(question, st.session_state.chunks, st.session_state.index)
        st.write(answer)
        source = ", ".join(f"Page {p}" for p in pages)
        st.caption(f"Source: {source}")
