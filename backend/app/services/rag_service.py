import math
import re
import logging
from typing import List, Dict, Any, Optional
import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        self.enabled = settings.ENABLE_RAG

    def chunk_document(self, raw_text: str, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Splits document text into manageable contextual chunks tagged with page numbers."""
        chunks = []
        chunk_idx = 0

        for page in pages_data:
            page_num = page.get("page", 1)
            text = page.get("text", "")
            # Split by double newline or sentences
            paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 20]
            if not paragraphs:
                paragraphs = [text.strip()] if text.strip() else []

            for p in paragraphs:
                chunks.append({
                    "chunk_id": chunk_idx,
                    "page": page_num,
                    "content": p
                })
                chunk_idx += 1

        return chunks

    def search_chunks(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Calculates BM25 / token-overlap relevance score between query and document chunks.
        Provides robust, fast, local semantic retrieval without requiring heavy GPU models.
        """
        query_tokens = set(re.findall(r"\w+", query.lower()))
        if not query_tokens or not chunks:
            return []

        scored_chunks = []
        for chunk in chunks:
            content = chunk["content"]
            chunk_tokens = re.findall(r"\w+", content.lower())
            if not chunk_tokens:
                continue

            # Term frequency matching
            score = 0.0
            for qt in query_tokens:
                tf = chunk_tokens.count(qt)
                if tf > 0:
                    score += (1.0 + math.log(tf))

            # Bonus for exact phrase match
            if query.lower() in content.lower():
                score += 5.0

            if score > 0:
                scored_chunks.append({
                    **chunk,
                    "relevance_score": round(score, 3)
                })

        scored_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)
        return scored_chunks[:top_k]

    async def answer_question(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        document_filename: str
    ) -> Dict[str, Any]:
        """Answers a question about a document citing source page and text."""
        top_chunks = self.search_chunks(query, chunks, top_k=4)
        if not top_chunks:
            return {
                "query": query,
                "answer": "No relevant context found in this document to answer the question.",
                "citations": []
            }

        citations = [
            {
                "page": c["page"],
                "content": c["content"][:250] + ("..." if len(c["content"]) > 250 else ""),
                "score": c["relevance_score"]
            }
            for c in top_chunks
        ]

        # If LLM configured, generate synthesis
        if settings.LLM_BASE_URL and settings.LLM_API_KEY and settings.LLM_PROVIDER != "none":
            try:
                context_str = "\n\n".join([f"[Page {c['page']}]: {c['content']}" for c in top_chunks])
                prompt = (
                    f"Document: {document_filename}\n"
                    f"Question: {query}\n\n"
                    f"Context Excerpts:\n{context_str}\n\n"
                    "Answer the user's question concisely based strictly on the excerpts above. "
                    "Always mention the relevant page number(s) in your answer."
                )

                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.post(
                        f"{settings.LLM_BASE_URL.rstrip('/')}/chat/completions",
                        headers={"Authorization": f"Bearer {settings.LLM_API_KEY}"},
                        json={
                            "model": settings.LLM_MODEL,
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0.2
                        }
                    )
                    if resp.status_code == 200:
                        ans_text = resp.json()["choices"][0]["message"]["content"].strip()
                        return {
                            "query": query,
                            "answer": ans_text,
                            "citations": citations
                        }
            except Exception as e:
                logger.warning(f"RAG LLM synthesis failed: {e}. Falling back to chunk citation.")

        # Fallback direct citation
        top_ans = f"Based on Page {top_chunks[0]['page']}: {top_chunks[0]['content']}"
        return {
            "query": query,
            "answer": top_ans,
            "citations": citations
        }

rag_service = RAGService()
