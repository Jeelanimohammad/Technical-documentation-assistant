"""
app.py
======
Streamlit Web Application for Technical Documentation Assistant with Source References.

Features:
- Responsive, modern UI with clear status indicators and dark/light compatibility.
- Technical PDF upload and live processing telemetry.
- PyMuPDF text extraction with scanned PDF detection.
- Sentence Transformers embeddings & FAISS similarity search.
- Open-source Flan-T5 local inference with anti-hallucination guardrails.
- Page-by-page source provenance viewer with similarity scores.
- Safety disclaimers on Generative AI and hallucination risks.
"""

import os
import streamlit as st
from document_processor import DocumentProcessor
from retriever import DocumentRetriever
from answer_generator import AnswerGenerator

# Page configuration
st.set_page_config(
    page_title="Technical Documentation Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for polished UI
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .source-box {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 0.8rem 1.2rem;
        border-radius: 6px;
        margin-bottom: 0.8rem;
    }
    .badge-page {
        background-color: #DBEAFE;
        color: #1D4ED8;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-sim {
        background-color: #E0E7FF;
        color: #4338CA;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.85rem;
        margin-left: 0.5rem;
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
    st.title("⚙️ System Control")
    st.markdown("### Document Ingestion")

    uploaded_file = st.file_uploader(
        "Upload a technical PDF",
        type=["pdf"],
        help="Upload documentation, manuals, datasheets, or academic papers."
    )

    if uploaded_file is not None:
        if st.session_state.current_file != uploaded_file.name:
            # Process newly uploaded file
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
                        st.success(f"✅ Processed {extraction['total_pages']} pages into {num_chunks} searchable chunks!")

    if st.session_state.current_file:
        st.markdown("---")
        st.markdown("### 📄 Active Document")
        st.info(
            f"**File:** `{st.session_state.current_file}`\n\n"
            f"**Total Pages:** `{st.session_state.page_count}`\n\n"
            f"**Indexed Chunks:** `{st.session_state.indexed_chunks}`"
        )

        if st.button("🗑️ Clear Active Document"):
            st.session_state.retriever = None
            st.session_state.current_file = None
            st.session_state.indexed_chunks = 0
            st.session_state.page_count = 0
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("---")
    st.markdown("### 🛡️ Architecture & Guardrails")
    st.markdown(
        "- **Parser:** PyMuPDF (`fitz`)\n"
        "- **Embeddings:** `all-MiniLM-L6-v2`\n"
        "- **Vector DB:** FAISS CPU (Flat Inner Product / Cosine)\n"
        "- **Generator:** HuggingFace `google/flan-t5-base`\n"
        "- **API Costs:** **$0.00** (100% Free & Local)"
    )


# --- Main Dashboard ---
st.markdown('<div class="main-title">📚 Technical Documentation Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Retrieval-Augmented Generation (RAG) system with strict source references & page attribution</div>',
    unsafe_allow_html=True
)

# Crucial RAG Disclaimer required by project specifications
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
                # 1. Retrieve passages from FAISS
                retrieved_chunks = st.session_state.retriever.retrieve(
                    user_question,
                    top_k=3,
                    min_similarity_threshold=0.25
                )

                # 2. Instantiate generator if needed
                if st.session_state.generator is None:
                    st.session_state.generator = AnswerGenerator()

                # 3. Generate context-grounded answer
                result = st.session_state.generator.generate_answer(user_question, retrieved_chunks)

                # Store in history
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
                st.markdown(f"#### Q: {entry['question']}")

                if entry["is_supported"]:
                    st.success(f"**Answer:** {entry['answer']}")
                else:
                    st.error(f"**Answer:** {entry['answer']}")

                # Source references section
                if entry["sources"]:
                    st.markdown("**📑 Source References:**")
                    cols = st.columns(len(entry["sources"]))
                    for c_idx, src in enumerate(entry["sources"]):
                        with cols[c_idx]:
                            st.markdown(
                                f"<span class='badge-page'>📄 Page {src['page_number']}</span> "
                                f"<span class='badge-sim'>Score: {src.get('similarity_score', 'N/A')}</span>",
                                unsafe_allow_html=True
                            )
                            st.caption(f"Source: `{src['source_file']}`")

                # Expandable supporting passages
                if entry["passages"]:
                    with st.expander(f"🔍 Inspect Supporting Passages for Query #{len(st.session_state.chat_history) - idx}"):
                        for p_idx, chunk in enumerate(entry["passages"], 1):
                            st.markdown(
                                f"<div class='source-box'>"
                                f"<strong>Passage {p_idx} (Page {chunk['page_number']} | Similarity: {chunk.get('similarity_score', 'N/A')}):</strong><br>"
                                f"<em>\"{chunk['text']}\"</em>"
                                f"</div>",
                                unsafe_allow_html=True
                            )

                st.markdown("---")
