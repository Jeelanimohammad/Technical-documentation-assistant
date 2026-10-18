"""
document_processor.py
=====================
Handles PDF document loading, text extraction using PyMuPDF,
graceful handling of scanned/empty documents, and sliding-window
chunking with strict page-level source tracking.
"""

from typing import List, Dict, Any, Optional
import pymupdf  # Modern PyMuPDF API


class DocumentProcessor:
    """Extracts text from PDF documents and splits it into searchable chunks with metadata."""

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 80):
        """
        Initialize the DocumentProcessor.

        Args:
            chunk_size (int): Target maximum number of words per chunk.
            chunk_overlap (int): Number of words to overlap between consecutive chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text_from_pdf(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """
        Extract text page by page from raw PDF bytes.

        Args:
            file_bytes (bytes): The raw byte content of the uploaded PDF file.
            filename (str): Name of the file being processed.

        Returns:
            Dict[str, Any]: Contains:
                - 'pages': list of dicts with {'page_number': int, 'text': str}
                - 'total_pages': int
                - 'is_scanned_likely': bool
                - 'error': Optional[str]
        """
        if not file_bytes:
            return {
                "pages": [],
                "total_pages": 0,
                "is_scanned_likely": False,
                "error": "The uploaded PDF file is empty (0 bytes)."
            }

        try:
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        except Exception as e:
            return {
                "pages": [],
                "total_pages": 0,
                "is_scanned_likely": False,
                "error": f"Failed to open PDF document: {str(e)}"
            }

        total_pages = len(doc)
        if total_pages == 0:
            doc.close()
            return {
                "pages": [],
                "total_pages": 0,
                "is_scanned_likely": False,
                "error": "The PDF contains no pages."
            }

        pages_data = []
        total_extracted_chars = 0

        for page_idx in range(total_pages):
            page = doc[page_idx]
            text = page.get_text("text").strip()
            total_extracted_chars += len(text)
            pages_data.append({
                "page_number": page_idx + 1,  # 1-indexed for human readability
                "text": text
            })

        doc.close()

        # Check if the document has virtually no extractable digital text (e.g. scanned images)
        is_scanned_likely = (total_extracted_chars < 50 and total_pages > 0)

        return {
            "pages": pages_data,
            "total_pages": total_pages,
            "total_chars": total_extracted_chars,
            "is_scanned_likely": is_scanned_likely,
            "error": None
        }

    def chunk_documents(self, pages_data: List[Dict[str, Any]], filename: str) -> List[Dict[str, Any]]:
        """
        Split page text into chunks while preserving page number and source metadata.

        Args:
            pages_data (List[Dict[str, Any]]): Extracted pages with page numbers and text.
            filename (str): Source document filename.

        Returns:
            List[Dict[str, Any]]: List of chunks, each containing:
                - 'chunk_id': int
                - 'text': str
                - 'page_number': int
                - 'source_file': str
                - 'word_count': int
        """
        chunks: List[Dict[str, Any]] = []
        chunk_id = 1

        for page in pages_data:
            page_num = page["page_number"]
            page_text = page["text"].strip()

            if not page_text:
                continue

            # Word-level tokenization for clean semantic chunking
            words = page_text.split()
            if not words:
                continue

            # If page text is within chunk_size, create a single chunk
            if len(words) <= self.chunk_size:
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": " ".join(words),
                    "page_number": page_num,
                    "source_file": filename,
                    "word_count": len(words)
                })
                chunk_id += 1
            else:
                # Sliding window chunking with overlap
                start = 0
                step = max(1, self.chunk_size - self.chunk_overlap)
                while start < len(words):
                    end = min(start + self.chunk_size, len(words))
                    chunk_words = words[start:end]
                    
                    if chunk_words:
                        chunks.append({
                            "chunk_id": chunk_id,
                            "text": " ".join(chunk_words),
                            "page_number": page_num,
                            "source_file": filename,
                            "word_count": len(chunk_words)
                        })
                        chunk_id += 1
                    
                    if end >= len(words):
                        break
                    start += step

        return chunks
