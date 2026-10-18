"""
answer_generator.py
===================
Context-grounded answer generation using a lightweight, open-source
sequence-to-sequence language model (google/flan-t5-base or flan-t5-small).

Ensures:
- Strict grounding in retrieved context
- Rejection of unsupported questions (anti-hallucination guard)
- Clean provenance and source citation linking
"""

from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class AnswerGenerator:
    """Generates grounded answers from retrieved context using an open-source model."""

    DEFAULT_MODEL = "google/flan-t5-base"
    FALLBACK_MODEL = "google/flan-t5-small"

    def __init__(self, model_name: str = "google/flan-t5-base"):
        """
        Initialize the generation model and tokenizer.
        Falls back to flan-t5-small if system memory is constrained.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

    def load_model(self) -> None:
        """Loads tokenizer and model weights into memory if not already loaded."""
        if self.model is not None:
            return

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
        except Exception as primary_err:
            # Fallback to flan-t5-small if base model fails (e.g. disk/memory constraint)
            if self.model_name != self.FALLBACK_MODEL:
                self.model_name = self.FALLBACK_MODEL
                self.tokenizer = AutoTokenizer.from_pretrained(self.FALLBACK_MODEL)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.FALLBACK_MODEL)
                self.model.to(self.device)
                self.model.eval()
            else:
                raise primary_err

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
        max_context_length: int = 1500
    ) -> Dict[str, Any]:
        """
        Generate an answer strictly grounded in retrieved passages.

        Args:
            question (str): The user's query.
            retrieved_chunks (List[Dict[str, Any]]): Retrieved passages from FAISS.
            max_context_length (int): Character ceiling to fit within token limit.

        Returns:
            Dict[str, Any]: Contains:
                - 'answer': Generated answer string.
                - 'is_supported': bool indicating whether context had sufficient info.
                - 'sources': list of {'source_file': str, 'page_number': int, 'score': float}.
        """
        if not retrieved_chunks:
            return {
                "answer": "The information requested was not found in the available document context.",
                "is_supported": False,
                "sources": []
            }

        self.load_model()

        # Build clean concatenated context while compiling unique sources
        context_parts = []
        sources = []
        seen_sources = set()

        for chunk in retrieved_chunks:
            source_key = (chunk.get("source_file", "document.pdf"), chunk.get("page_number", 1))
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                sources.append({
                    "source_file": source_key[0],
                    "page_number": source_key[1],
                    "similarity_score": chunk.get("similarity_score", 0.0)
                })
            context_parts.append(f"[Page {chunk.get('page_number', 1)}]: {chunk.get('text', '')}")

        context_str = "\n\n".join(context_parts)
        if len(context_str) > max_context_length:
            context_str = context_str[:max_context_length] + "..."

        # Prompt engineering for strict context-grounded reasoning & refusing unsupported queries
        prompt = (
            "Read the following technical document context carefully and answer the question.\n"
            "Rules:\n"
            "1. Answer ONLY using facts directly stated in the context.\n"
            "2. If the context does not contain enough information to answer the question, "
            "respond exactly: 'The information was not found in the available document context.'\n"
            "3. Do not invent or assume any facts.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {question.strip()}\n\n"
            "Answer:"
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                num_beams=2,
                early_stopping=True,
                do_sample=False
            )

        raw_answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Check if the model expressed absence of information or empty response
        unsupported_markers = [
            "not found",
            "not mentioned",
            "does not contain",
            "no information",
            "not stated",
            "not available",
            "unanswerable"
        ]

        is_unsupported = (
            not raw_answer or
            any(marker in raw_answer.lower() for marker in unsupported_markers)
        )

        if is_unsupported:
            final_answer = "The information was not found in the available document context."
            return {
                "answer": final_answer,
                "is_supported": False,
                "sources": sources
            }

        return {
            "answer": raw_answer,
            "is_supported": True,
            "sources": sources
        }
