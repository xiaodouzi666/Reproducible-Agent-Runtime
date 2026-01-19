"""
LocalSearchTool - BM25-based local corpus search.
"""

import os
import time
import hashlib
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from .base import BaseTool, ToolResult, EvidenceAnchor

# Try to import rank_bm25, fall back to simple keyword matching
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False


@dataclass
class Document:
    """A document in the corpus."""
    doc_id: str
    title: str
    content: str
    path: str
    paragraphs: list  # List of (paragraph_idx, text) tuples


class LocalSearchTool(BaseTool):
    """
    Search tool using BM25 over a local corpus.
    Falls back to keyword matching if rank_bm25 is not installed.
    """

    name = "local_search"
    description = "Search local corpus for relevant documents and passages"

    def __init__(
        self,
        corpus_dir: str,
        seed: Optional[int] = None,
        top_k: int = 5
    ):
        super().__init__(seed=seed)
        self.corpus_dir = Path(corpus_dir)
        self.top_k = top_k
        self.documents: list[Document] = []
        self.bm25 = None
        self._load_corpus()

    def _load_corpus(self):
        """Load all documents from the corpus directory."""
        if not self.corpus_dir.exists():
            return

        for file_path in self.corpus_dir.glob("**/*"):
            if file_path.suffix.lower() in [".txt", ".md", ".markdown"]:
                try:
                    content = file_path.read_text(encoding="utf-8")
                    doc_id = file_path.stem
                    title = self._extract_title(content, file_path.name)

                    # Split into paragraphs
                    paragraphs = []
                    for i, para in enumerate(content.split("\n\n")):
                        para = para.strip()
                        if para and len(para) > 20:  # Skip very short paragraphs
                            paragraphs.append((i, para))

                    self.documents.append(Document(
                        doc_id=doc_id,
                        title=title,
                        content=content,
                        path=str(file_path),
                        paragraphs=paragraphs
                    ))
                except Exception as e:
                    print(f"Warning: Failed to load {file_path}: {e}")

        # Build BM25 index if available
        if HAS_BM25 and self.documents:
            # Tokenize all paragraphs
            corpus = []
            self._para_index = []  # (doc_idx, para_idx)

            for doc_idx, doc in enumerate(self.documents):
                for para_idx, para_text in doc.paragraphs:
                    tokens = self._tokenize(para_text)
                    corpus.append(tokens)
                    self._para_index.append((doc_idx, para_idx, para_text))

            if corpus:
                self.bm25 = BM25Okapi(corpus)

    def _extract_title(self, content: str, filename: str) -> str:
        """Extract title from content or use filename."""
        lines = content.strip().split("\n")
        for line in lines[:5]:
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
            if line and not line.startswith("#") and len(line) > 10:
                return line[:100]
        return filename

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization."""
        # Basic tokenization - lowercase and split on whitespace/punctuation
        import re
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def _keyword_search(self, query: str, top_k: int) -> list[tuple[int, int, str, float]]:
        """Fallback keyword search when BM25 is not available."""
        query_tokens = set(self._tokenize(query))
        results = []

        for doc_idx, doc in enumerate(self.documents):
            for para_idx, para_text in doc.paragraphs:
                para_tokens = set(self._tokenize(para_text))
                # Simple overlap score
                overlap = len(query_tokens & para_tokens)
                if overlap > 0:
                    score = overlap / len(query_tokens)
                    results.append((doc_idx, para_idx, para_text, score))

        results.sort(key=lambda x: x[3], reverse=True)
        return results[:top_k]

    def execute(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> ToolResult:
        """
        Search the corpus for relevant passages.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            ToolResult with matching passages and evidence anchors
        """
        start_time = time.time()
        top_k = top_k or self.top_k

        if not self.documents:
            return ToolResult(
                success=False,
                output=[],
                error="No documents loaded in corpus",
                tool_name=self.name,
                input_data={"query": query},
                latency_ms=(time.time() - start_time) * 1000
            )

        # Perform search
        if HAS_BM25 and self.bm25:
            query_tokens = self._tokenize(query)
            scores = self.bm25.get_scores(query_tokens)

            # Get top-k results
            top_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:top_k]

            results = []
            for idx in top_indices:
                if scores[idx] > 0:
                    doc_idx, para_idx, para_text = self._para_index[idx]
                    results.append((doc_idx, para_idx, para_text, scores[idx]))
        else:
            results = self._keyword_search(query, top_k)

        # Build evidence anchors
        evidence_anchors = []
        search_results = []

        for doc_idx, para_idx, para_text, score in results:
            doc = self.documents[doc_idx]

            # Create evidence anchor
            anchor = EvidenceAnchor(
                doc_id=doc.doc_id,
                doc_title=doc.title,
                location=f"paragraph_{para_idx}",
                content_hash=hashlib.md5(para_text.encode()).hexdigest()[:12],
                snippet=para_text[:500] + ("..." if len(para_text) > 500 else ""),
                relevance_score=float(score)
            )
            evidence_anchors.append(anchor)

            search_results.append({
                "doc_id": doc.doc_id,
                "title": doc.title,
                "paragraph": para_idx,
                "snippet": para_text[:500],
                "score": float(score)
            })

        latency_ms = (time.time() - start_time) * 1000

        return ToolResult(
            success=True,
            output=search_results,
            tool_name=self.name,
            input_data={"query": query, "top_k": top_k},
            latency_ms=latency_ms,
            evidence_anchors=[a.to_dict() for a in evidence_anchors]
        )

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        for doc in self.documents:
            if doc.doc_id == doc_id:
                return doc
        return None

    def get_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "query": {"type": "string", "description": "Search query"},
                "top_k": {"type": "integer", "description": "Number of results", "default": 5}
            }
        }
