import json
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer


class Retriever:

    def __init__(self):

        # Load the same Sentence-BERT model
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    def search(
        self,
        query,
        index_path,
        metadata_path,
        top_k=10
    ):

        # ------------------------
        # Load FAISS index
        # ------------------------

        index = faiss.read_index(index_path)

        # ------------------------
        # Query Embedding
        # ------------------------

        query_vector = self.model.encode(

            [query],

            normalize_embeddings=True,

            convert_to_numpy=True

        )

        query_vector = query_vector.astype("float32")

        # ------------------------
        # Search
        # ------------------------

        scores, indices = index.search(

            query_vector,

            top_k

        )

        # ------------------------
        # Load Metadata
        # ------------------------

        with open(

            metadata_path,

            encoding="utf-8"

        ) as f:

            metadata = json.load(f)

        results = []

        # ------------------------
        # Match vectors with clauses
        # ------------------------

        for score, idx in zip(

            scores[0],

            indices[0]

        ):

            if idx < len(metadata):

                results.append({

                    "clause_id": metadata[idx]["clause_id"],

                    "document": metadata[idx]["document"],

                    "text": metadata[idx]["text"],

                    "score": float(score)

                })

        return results