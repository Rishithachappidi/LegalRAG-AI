import json
import faiss
import numpy as np

from models.sentence_bert import load_model


class Retriever:

    def __init__(self):

        # Reuse the same Sentence-BERT model for query embedding
        self.model = load_model()
        self._index_cache = {}
        self._metadata_cache = {}

    def search(
        self,
        query,
        index_path,
        metadata_path,
        top_k=10
    ):

        # ------------------------
        # Load FAISS index and metadata once per path
        # ------------------------

        if index_path not in self._index_cache:
            self._index_cache[index_path] = faiss.read_index(index_path)

        if metadata_path not in self._metadata_cache:
            with open(metadata_path, encoding="utf-8") as f:
                self._metadata_cache[metadata_path] = json.load(f)

        index = self._index_cache[index_path]
        metadata = self._metadata_cache[metadata_path]

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