"""
app.py
======
Streamlit Web Application for Technical Documentation Assistant with Source References.
Features a modern, high-contrast, responsive UI with glassmorphism touches and clear attribution.
"""

import os
import streamlit as st
from document_processor import DocumentProcessor
from retriever import DocumentRetriever
from answer_generator import AnswerGenerator

# Page configuration
st.set_page_config(
    page_title="Technical Documentation Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .hero-container {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #1E3A8A 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(30, 27, 75, 0.25);
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.4rem;
        color: #FFFFFF;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #C7D2FE;
        font-weight: 400;
        line-height: 1.5;
    }
    .status-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1.2rem;
    }
    .answer-card {
        background: #FFFFFF;
        border-radius: 12px;
        border: 1px solid #E0E7FF;
        border-left: 6px solid #4F46E5;
        padding: 1.4rem 1.6rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .badge-page {
        background-color: #EEF2FF;
        color: #4338CA;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        border: 1px solid #C7D2FE;
        display: inline-block;
    }
    .badge-score {
        background-color: #ECFDF5;
        color: #065F46;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        border: 1px solid #A7F3D0;
        display: inline-block;
        margin-left: 0.4rem;
    }
    .source-passage-box {
        background-color: #F8FAFC;
        border-left: 4px solid #6366F1;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        margin-bottom: 0.8rem;
        font-size: 0.95rem;
        line-height: 1.6;
        color: #334155;
    }
    .stButton>button {
        background: linear-gradient(135deg, #4F46E5 0%, #3B82F6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 1.4rem;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.35);
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "processor" not in st.session_state:
    st.session_state.processor = DocumentProcessor(chunk_size=350, chunk_overlap=70)

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "generator" not in st.session_state:
    st.session_state.generator = None

if "current_file" not in st.session_state:
    st.session_state.current_file = None

if "indexed_chunks" not in st.session_state:
    st.session_state.indexed_chunks = 0

if "page_count" not in st.session_state:
    st.session_state.page_count = 0

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# --- Sidebar ---
with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    st.markdown("### Document Ingestion")

    uploaded_file = st.file_uploader(
        "Upload a technical PDF",
        type=["pdf"],
        help="Upload documentation, manuals, datasheets, or academic papers."
    )

    if uploaded_file is not None:
        if st.session_state.current_file != uploaded_file.name:
            with st.spinner("Extracting text and building vector index..."):
                file_bytes = uploaded_file.read()
                extraction = st.session_state.processor.extract_text_from_pdf(file_bytes, uploaded_file.name)

                if extraction["error"]:
                    st.error(f"❌ {extraction['error']}")
                elif extraction["is_scanned_likely"]:
                    st.warning(
                        "⚠️ **Notice: Scanned PDF Detected!**\n\n"
                        "This document appears to contain scanned images without digital selectable text. "
                        "Standard PDF parsers cannot read images directly. "
                        "To process scanned documents, an OCR (Optical Character Recognition) engine like Tesseract would be required."
                    )
                else:
                    chunks = st.session_state.processor.chunk_documents(extraction["pages"], uploaded_file.name)
                    if not chunks:
                        st.error("❌ No text could be split into chunks.")
                    else:
                        if st.session_state.retriever is None:
                            st.session_state.retriever = DocumentRetriever()

                        num_chunks = st.session_state.retriever.build_index(chunks)
                        st.session_state.current_file = uploaded_file.name
                        st.session_state.indexed_chunks = num_chunks
                        st.session_state.page_count = extraction["total_pages"]
                        st.session_state.chat_history = []
                        st.success(f"✅ Indexed {extraction['total_pages']} pages into {num_chunks} searchable chunks!")

    if st.session_state.current_file:
        st.markdown("---")
        st.markdown("### 📄 Active Document")
        st.markdown(f"""
        <div class="status-card">
            <strong>File:</strong> <code>{st.session_state.current_file}</code><br>
            <strong>Pages:</strong> {st.session_state.page_count}<br>
            <strong>Indexed Chunks:</strong> {st.session_state.indexed_chunks}
        </div>
        """, unsafe_allow_html=True)

        if st.button("🗑️ Clear Active Document", use_container_width=True):
            st.session_state.retriever = None
            st.session_state.current_file = None
            st.session_state.indexed_chunks = 0
            st.session_state.page_count = 0
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("---")
    st.markdown("### 🛡️ Architecture & Guardrails")
    st.markdown(
        "- **Parser:** PyMuPDF (`pymupdf`)\n"
        "- **Embeddings:** `all-MiniLM-L6-v2` (384-dim)\n"
        "- **Vector DB:** FAISS CPU (Flat Inner Product / Cosine)\n"
        "- **QA Generator:** HuggingFace `google/flan-t5-base`\n"
        "- **Model Cost:** **$0.00** (100% Free & Open-Source)"
    )


# --- Main Dashboard ---
st.markdown("""
<div class="hero-container">
    <div class="hero-title">⚡ Technical Documentation Assistant</div>
    <div class="hero-subtitle">Production-grade Retrieval-Augmented Generation (RAG) system with strict source references, exact page attribution, and anti-hallucination guardrails.</div>
</div>
""", unsafe_allow_html=True)

# Generative AI Disclaimer
st.warning(
    "⚠️ **Generative AI Notice:** Retrieval-Augmented Generation (RAG) significantly reduces false assertions "
    "by grounding responses in source documents, but it **does not completely eliminate hallucinations**. "
    "Always verify critical technical decisions against the cited page references below."
)

if not st.session_state.current_file:
    st.info("👈 **Get Started:** Please upload a technical PDF document using the sidebar to begin asking questions.")
else:
    # Query input form
    with st.form("query_form", clear_on_submit=False):
        user_question = st.text_input(
            "Enter your technical question about the uploaded document:",
            placeholder="e.g., What are the core hardware prerequisites and system requirements?"
        )
        col_btn, col_extra = st.columns([1, 4])
        with col_btn:
            submit_button = st.form_submit_button("🔍 Ask Assistant", use_container_width=True)

    if submit_button:
        if not user_question.strip():
            st.warning("⚠️ Please enter a question before clicking Ask.")
        else:
            with st.spinner("Retrieving relevant passages and synthesizing verified answer..."):
                retrieved_chunks = st.session_state.retriever.retrieve(
                    user_question,
                    top_k=3,
                    min_similarity_threshold=0.25
                )

                if st.session_state.generator is None:
                    st.session_state.generator = AnswerGenerator()

                result = st.session_state.generator.generate_answer(user_question, retrieved_chunks)

                st.session_state.chat_history.insert(0, {
                    "question": user_question,
                    "answer": result["answer"],
                    "is_supported": result["is_supported"],
                    "sources": result["sources"],
                    "passages": retrieved_chunks
                })

    # Display Response History
    if st.session_state.chat_history:
        st.markdown("### 💬 Answers & Source Verifications")

        for idx, entry in enumerate(st.session_state.chat_history):
            with st.container():
                st.markdown(f"#### ❓ {entry['question']}")

                if entry["is_supported"]:
                    st.markdown(f"""
                    <div class="answer-card">
                        <strong style="color: #4F46E5; font-size: 1.1rem;">Verified Answer:</strong><br>
                        <p style="font-size: 1.05rem; margin-top: 0.5rem; line-height: 1.6; color: #1E293B;">{entry['answer']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="answer-card" style="border-left-color: #EF4444;">
                        <strong style="color: #DC2626; font-size: 1.1rem;">Context Status:</strong><br>
                        <p style="font-size: 1.05rem; margin-top: 0.5rem; line-height: 1.6; color: #7F1D1D;">{entry['answer']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Source references section
                if entry["sources"]:
                    st.markdown("**📑 Source References:**")
                    cols = st.columns(len(entry["sources"]))
                    for c_idx, src in enumerate(entry["sources"]):
                        with cols[c_idx]:
                            st.markdown(
                                f"<span class='badge-page'>📄 Page {src['page_number']}</span> "
                                f"<span class='badge-score'>Similarity: {src.get('similarity_score', 'N/A')}</span>",
                                unsafe_allow_html=True
                            )
                            st.caption(f"Source: `{src['source_file']}`")

                # Expandable supporting passages
                if entry["passages"]:
                    with st.expander(f"🔍 Inspect Supporting Passages for Query #{len(st.session_state.chat_history) - idx}"):
                        for p_idx, chunk in enumerate(entry["passages"], 1):
                            st.markdown(
                                f"<div class='source-passage-box'>"
                                f"<strong>Passage {p_idx} (Page {chunk['page_number']} | Relevance: {chunk.get('similarity_score', 'N/A')}):</strong><br>"
                                f"<em>\"{chunk['text']}\"</em>"
                                f"</div>",
                                unsafe_allow_html=True
                            )

                st.markdown("---")
