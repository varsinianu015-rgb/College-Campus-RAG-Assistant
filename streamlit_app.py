import os
from pathlib import Path

import requests
import streamlit as st

st.set_page_config(
    page_title="College RAG Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_API_URL = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8001").rstrip("/")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1100px;}
      .hero {padding: 1.25rem 1.5rem; border-radius: 16px; background: linear-gradient(120deg,#172554,#312e81); color: white; margin-bottom: 1.2rem;}
      .hero h1 {color: white; margin: 0 0 .35rem 0;}
      .hero p {color: #dbeafe; margin: 0;}
      .source-card {border: 1px solid rgba(128,128,128,.35); border-radius: 10px; padding: .7rem .9rem; margin: .4rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

st.markdown(
    """
    <div class="hero">
      <h1>🎓 College Campus RAG Assistant</h1>
      <p>Ask questions about college regulations, student handbooks, and HR policies. Answers are grounded in uploaded PDFs.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Settings")
    api_url = st.text_input("FastAPI backend URL", value=DEFAULT_API_URL).rstrip("/")
    st.caption("Start the FastAPI backend before using this interface.")
    if st.button("Check backend", use_container_width=True):
        try:
            response = requests.get(f"{api_url}/health", timeout=8)
            response.raise_for_status()
            st.success("Backend is online")
        except requests.RequestException as exc:
            st.error(f"Backend is unavailable: {exc}")

    st.divider()
    st.header("📄 Upload college PDFs")
    pdf_files = st.file_uploader(
        "Choose one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Files are sent to FastAPI, processed into chunks, embedded locally on your computer, and stored in ChromaDB.",
    )
    if st.button("Upload and index PDFs", type="primary", use_container_width=True, disabled=not pdf_files):
        files_payload = []
        try:
            for uploaded in pdf_files:
                files_payload.append(("files", (uploaded.name, uploaded.getvalue(), "application/pdf")))
            with st.spinner("Uploading PDFs and creating local embeddings (first run downloads the model)..."):
                response = requests.post(
                    f"{api_url}/documents/upload",
                    files=files_payload,
                    timeout=300,
                )
            if response.ok:
                result = response.json()
                st.success(result.get("message", "Upload completed."))
                for item in result.get("documents", []):
                    st.write(f"**{item.get('filename', item.get('source_document', 'PDF'))}**")
                    if item.get("chunks") is not None:
                        st.caption(f"Indexed chunks: {item['chunks']}")
                    elif item.get("chunk_count") is not None:
                        st.caption(f"Indexed chunks: {item['chunk_count']}")
                st.session_state.uploaded_files.extend([f.name for f in pdf_files])
            else:
                try:
                    detail = response.json().get("detail", response.text)
                except ValueError:
                    detail = response.text
                st.error(f"Upload failed ({response.status_code}): {detail}")
        except requests.RequestException as exc:
            st.error(f"Could not connect to the backend: {exc}")

    if st.session_state.uploaded_files:
        st.divider()
        st.caption("Files uploaded in this session")
        for name in dict.fromkeys(st.session_state.uploaded_files):
            st.write(f"📘 {name}")

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.subheader("💬 Ask your college documents")
if not st.session_state.messages:
    st.info("Start by uploading a PDF in the sidebar, then ask a question below.")
    with st.expander("Try a sample question"):
        st.markdown(
            "- What is the attendance requirement for students?\n"
            "- What is the hostel curfew?\n"
            "- How many casual leave days are available to eligible employees?\n"
            "- What is the library fine for overdue books?"
        )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📚 View source documents and pages"):
                for source in message["sources"]:
                    filename = source.get("source_document", "Unknown document")
                    page = source.get("page_number", "?")
                    score = source.get("relevance_score")
                    score_text = f" · relevance: {score:.2f}" if isinstance(score, (int, float)) else ""
                    st.markdown(
                        f'<div class="source-card"><b>{filename}</b><br/>Page {page}{score_text}</div>',
                        unsafe_allow_html=True,
                    )

question = st.chat_input("Ask a question about your uploaded PDFs...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching locally embedded PDFs and asking GPT-4o mini..."):
                response = requests.post(
                    f"{api_url}/ask",
                    json={"question": question},
                    timeout=300,
                )
            if response.ok:
                result = response.json()
                answer = result.get("answer", "No answer was returned.")
                sources = result.get("sources", [])
                st.markdown(answer)
                if sources:
                    with st.expander("📚 View source documents and pages", expanded=True):
                        for source in sources:
                            filename = source.get("source_document", "Unknown document")
                            page = source.get("page_number", "?")
                            score = source.get("relevance_score")
                            score_text = f" · relevance: {score:.2f}" if isinstance(score, (int, float)) else ""
                            st.markdown(
                                f'<div class="source-card"><b>{filename}</b><br/>Page {page}{score_text}</div>',
                                unsafe_allow_html=True,
                            )
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            else:
                try:
                    detail = response.json().get("detail", response.text)
                except ValueError:
                    detail = response.text
                error_message = f"The backend returned an error ({response.status_code}): {detail}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
        except requests.RequestException as exc:
            error_message = f"Cannot reach FastAPI at {api_url}. Make sure the backend is running. Details: {exc}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
