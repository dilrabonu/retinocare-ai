"""Hybrid retrieval: dense (ChromaDB) + sparse (BM25), combined via
Reciprocal Rank Fusion (RRF).

Why hybrid: dense/semantic search finds text that means the same thing
even with different words ("high risk of vision loss" ~ "severe
retinopathy"). Sparse/keyword search (BM25) is better at exact term
matches (specific thresholds, named criteria). Combining catches more
relevant guideline text than either alone.
"""

import re
from pathlib import Path
from typing import List

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi


def _chunk_markdown(text: str, source: str) -> List[dict]:
    """Splits a markdown file into paragraph-level chunks (blank-line
    separated), dropping empty chunks and headings-only lines."""
    raw_chunks = re.split(r"\n\s*\n", text)
    chunks = []
    for chunk in raw_chunks:
        chunk = chunk.strip()
        if len(chunk) < 20:  # skip empty/near-empty fragments
            continue
        chunks.append({"text": chunk, "source": source})
    return chunks


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


class HybridRetriever:
    def __init__(self, knowledge_base_dir: str | Path, persist_dir: str | None = None):
        self.knowledge_base_dir = Path(knowledge_base_dir)

        md_files = sorted(self.knowledge_base_dir.rglob("*.md"))
        if not md_files:
            raise FileNotFoundError(
                f"No .md files found in {self.knowledge_base_dir} -- "
                f"add guideline documents before building the retriever."
            )

        self.chunks: List[dict] = []
        for path in md_files:
            text = path.read_text(encoding="utf-8")
            self.chunks.extend(_chunk_markdown(text, source=path.name))

        # --- Dense index (ChromaDB) ---
        client = (
            chromadb.PersistentClient(path=persist_dir)
            if persist_dir
            else chromadb.EphemeralClient()
        )
        self.collection = client.get_or_create_collection(name="retinocare_guidelines")
        # Refresh contents each build so re-running this doesn't duplicate entries.
        existing_ids = self.collection.get()["ids"]
        if existing_ids:
            self.collection.delete(ids=existing_ids)
        self.collection.add(
            documents=[c["text"] for c in self.chunks],
            metadatas=[{"source": c["source"]} for c in self.chunks],
            ids=[f"chunk_{i}" for i in range(len(self.chunks))],
        )

        # --- Sparse index (BM25) ---
        tokenized_corpus = [_tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, k: int = 5) -> List[dict]:
        """Returns top-k chunks as [{"text":..., "source":..., "score":...}],
        fused from dense + sparse rankings via Reciprocal Rank Fusion.
        """
        n = len(self.chunks)
        k_dense = min(k * 2, n)
        k_sparse = min(k * 2, n)

        # Dense ranking
        dense_result = self.collection.query(query_texts=[query], n_results=k_dense)
        dense_ids = [int(i.split("_")[1]) for i in dense_result["ids"][0]]

        # Sparse ranking
        bm25_scores = self.bm25.get_scores(_tokenize(query))
        sparse_ids = list(np.argsort(bm25_scores)[::-1][:k_sparse]) if k_sparse else []

        # Reciprocal Rank Fusion: score = sum(1 / (rank + fusion_k)) across rankers
        fusion_k = 60
        rrf_scores: dict[int, float] = {}
        for rank, idx in enumerate(dense_ids):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (fusion_k + rank + 1)
        for rank, idx in enumerate(sparse_ids):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (fusion_k + rank + 1)

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        return [
            {
                "text": self.chunks[idx]["text"],
                "source": self.chunks[idx]["source"],
                "score": round(score, 4),
            }
            for idx, score in ranked
        ]