import json
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
POLICY_PATH = ROOT / "policy" / "documents.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def _split_sentences(text):
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]


class PolicyRetriever:
    def __init__(self):
        with POLICY_PATH.open(encoding="utf-8") as file:
            documents = json.load(file)

        self.chunks = []

        for document in documents:
            for position, sentence in enumerate(
                _split_sentences(document["text"])
            ):
                self.chunks.append({
                    "document_id": document["id"],
                    "title": document["title"],
                    "sentence_number": position + 1,
                    "text": sentence,
                })

        self.encoder = SentenceTransformer(EMBEDDING_MODEL)

        texts = [chunk["text"] for chunk in self.chunks]
        embeddings = self.encoder.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, query, k=3):
        query_embedding = self.encoder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        scores, indices = self.index.search(
            query_embedding,
            min(k, len(self.chunks)),
        )

        results = []

        for score, index in zip(scores[0], indices[0]):
            chunk = dict(self.chunks[int(index)])
            chunk["score"] = round(float(score), 4)
            results.append(chunk)

        return results


if __name__ == "__main__":
    retriever = PolicyRetriever()

    for result in retriever.search(
        "How long do I have to return an electronics product?"
    ):
        print(result)