"""
test_pipeline.py
================
Automated verification script to test:
1. Technical PDF generation with known ground truth across multiple pages.
2. Text extraction and scanned/empty page validation with PyMuPDF.
3. Word-level chunking and page metadata integrity.
4. Sentence Transformers embedding computation and FAISS index construction.
5. In-domain question retrieval with correct page references.
6. Out-of-domain / unsupported question refusal (anti-hallucination guard).
7. Empty/corrupt PDF handling.
"""

import sys
import os
import pymupdf
from document_processor import DocumentProcessor
from retriever import DocumentRetriever
from answer_generator import AnswerGenerator


def create_sample_technical_pdf(filepath: str) -> None:
    """Generates a synthetic technical manual PDF with verified multi-page facts."""
    doc = pymupdf.open()

    # Page 1: Architecture & Hardware Requirements
    p1 = doc.new_page()
    p1_text = (
        "Project Architecture and System Requirements\n\n"
        "The Technical Documentation Assistant operates as a modular Retrieval-Augmented Generation (RAG) system.\n"
        "Minimum hardware specifications require a dual-core 64-bit x86 or ARM CPU, 8 GB of RAM, and 5 GB of free disk space.\n"
        "The primary embedding model utilized is all-MiniLM-L6-v2 which maps text chunks into a 384-dimensional dense vector space.\n"
        "For vector similarity indexing, FAISS IndexFlatIP is configured with L2 normalization to compute exact cosine similarity.\n"
    )
    p1.insert_text((50, 50), p1_text, fontsize=11)

    # Page 2: Safety, Security, and Storage
    p2 = doc.new_page()
    p2_text = (
        "Security, Guardrails, and Data Storage Policies\n\n"
        "All uploaded documents and generated vector embeddings are stored in-memory during active sessions.\n"
        "The system enforces a strict anti-hallucination policy: if the query cannot be answered from retrieved passages, "
        "the language model must explicitly output that the information was not found in the available document context.\n"
        "Optical Character Recognition (OCR) using Tesseract is recommended if scanned image PDFs without digital text are ingested.\n"
    )
    p2.insert_text((50, 50), p2_text, fontsize=11)

    doc.save(filepath)
    doc.close()
    print(f"Created sample technical PDF at: {filepath}")


def create_empty_pdf(filepath: str) -> None:
    """Generates a PDF with an empty page (no text)."""
    doc = pymupdf.open()
    doc.new_page()  # Blank page
    doc.save(filepath)
    doc.close()
    print(f"Created blank PDF at: {filepath}")


def run_tests():
    print("=" * 60)
    print("RUNNING AUTOMATED RAG PIPELINE & INTEGRATION TESTS")
    print("=" * 60)

    sample_pdf_path = "sample_technical_manual.pdf"
    blank_pdf_path = "blank_test.pdf"

    create_sample_technical_pdf(sample_pdf_path)
    create_empty_pdf(blank_pdf_path)

    processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)

    # --- Test 1: Empty PDF / Scanned detection ---
    print("\n[TEST 1] Testing empty/scanned PDF handling...")
    with open(blank_pdf_path, "rb") as f:
        blank_bytes = f.read()
    blank_res = processor.extract_text_from_pdf(blank_bytes, "blank_test.pdf")
    assert blank_res["is_scanned_likely"] == True, "Failed to flag image/blank PDF as scanned/empty!"
    print("  -> Passed: Blank/scanned PDF accurately flagged.")

    # --- Test 2: Multi-page Text Extraction & Metadata Tracking ---
    print("\n[TEST 2] Testing PDF text extraction & page-level chunking...")
    with open(sample_pdf_path, "rb") as f:
        sample_bytes = f.read()
    extract_res = processor.extract_text_from_pdf(sample_bytes, sample_pdf_path)
    assert extract_res["total_pages"] == 2, f"Expected 2 pages, got {extract_res['total_pages']}"
    assert extract_res["error"] is None, f"Encountered error: {extract_res['error']}"

    chunks = processor.chunk_documents(extract_res["pages"], sample_pdf_path)
    assert len(chunks) >= 2, f"Expected at least 2 chunks, got {len(chunks)}"

    pages_in_chunks = {c["page_number"] for c in chunks}
    assert 1 in pages_in_chunks and 2 in pages_in_chunks, "Missing page numbers in chunk metadata!"
    print(f"  -> Passed: Extracted {extract_res['total_pages']} pages into {len(chunks)} chunks with accurate page IDs.")

    # --- Test 3: Embedding Generation and FAISS Indexing ---
    print("\n[TEST 3] Testing Sentence Transformers embedding & FAISS indexing...")
    retriever = DocumentRetriever(model_name="all-MiniLM-L6-v2")
    indexed_count = retriever.build_index(chunks)
    assert indexed_count == len(chunks), f"Expected {len(chunks)} indexed, got {indexed_count}"
    print(f"  -> Passed: FAISS successfully indexed {indexed_count} vector embeddings.")

    # --- Test 4: Semantic Search & Page Citation Accuracy ---
    print("\n[TEST 4] Testing semantic search on known facts...")
    q1 = "What is the minimum RAM and hardware requirement?"
    retrieved_q1 = retriever.retrieve(q1, top_k=2)
    assert len(retrieved_q1) > 0, "Failed to retrieve any passage for known question!"
    top_chunk = retrieved_q1[0]
    assert top_chunk["page_number"] == 1, f"Expected fact to be found on Page 1, but retrieved Page {top_chunk['page_number']}"
    assert "8 GB of RAM" in top_chunk["text"], f"Expected RAM text in passage, got: {top_chunk['text']}"
    print(f"  -> Passed: Correctly retrieved Page {top_chunk['page_number']} with similarity score: {top_chunk['similarity_score']}")

    # --- Test 5: Grounded Answer Generation (Present Fact) ---
    print("\n[TEST 5] Testing answer generation on present factual question...")
    generator = AnswerGenerator(model_name="google/flan-t5-base")
    ans_res = generator.generate_answer(q1, retrieved_q1)
    print(f"  Question: {q1}")
    print(f"  Generated Answer: {ans_res['answer']}")
    print(f"  Is Supported: {ans_res['is_supported']}")
    print(f"  Sources: {ans_res['sources']}")
    assert ans_res["is_supported"] == True, "Model incorrectly flagged supported question as unsupported!"
    assert ans_res["sources"][0]["page_number"] == 1, "Source reference page number mismatch!"
    print("  -> Passed: Generated factual answer grounded in Page 1.")

    # --- Test 6: Rejection of Absent / Unsupported Questions (Anti-Hallucination) ---
    print("\n[TEST 6] Testing rejection of absent/unsupported questions...")
    q_absent = "What is the capital city of Australia and what is its population?"
    retrieved_absent = retriever.retrieve(q_absent, top_k=2, min_similarity_threshold=0.35)
    ans_absent = generator.generate_answer(q_absent, retrieved_absent)
    print(f"  Question: {q_absent}")
    print(f"  Generated Answer: {ans_absent['answer']}")
    print(f"  Is Supported: {ans_absent['is_supported']}")
    assert ans_absent["is_supported"] == False, "Model hallucinated an answer for out-of-context query!"
    assert "not found" in ans_absent["answer"].lower(), "Missing standard rejection message!"
    print("  -> Passed: Refused hallucination and properly declared absence of context.")

    # Clean up test artifacts
    if os.path.exists(blank_pdf_path):
        os.remove(blank_pdf_path)

    print("\n" + "=" * 60)
    print("ALL 6 AUTOMATED RAG TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
