import os
import json
import numpy as np
import streamlit as st

# --------------------------------------------------
# Load Model Only Once
# --------------------------------------------------

@st.cache_resource
def load_model():
    from sentence_transformers import SentenceTransformer

    os.environ.setdefault(
        "TOKENIZERS_PARALLELISM",
        "false"
    )

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


class SentenceBERT:

    def __init__(self, output_folder="embeddings"):

        self.model = load_model()

        self.output_folder = output_folder

        os.makedirs(
            self.output_folder,
            exist_ok=True
        )

    # --------------------------------------------------
    # Generate Embeddings
    # --------------------------------------------------

    def generate(self, chunk_file):

        filename = os.path.basename(
            chunk_file
        ).replace(
            "_chunks.json",
            "_embeddings.npy"
        )

        save_path = os.path.join(
            self.output_folder,
            filename
        )

        # ------------------------------------------
        # If Embeddings Already Exist
        # ------------------------------------------

        if os.path.exists(save_path):

            embeddings = np.load(save_path)

            return {

                "embedding_file": save_path,

                "embeddings": embeddings,

                "total_chunks": embeddings.shape[0],

                "dimension": embeddings.shape[1]

            }

        # ------------------------------------------
        # Read Clauses
        # ------------------------------------------

        with open(
            chunk_file,
            "r",
            encoding="utf-8"
        ) as f:

            clauses = json.load(f)

        texts = [

            clause["text"]

            for clause in clauses

        ]

        # ------------------------------------------
        # Generate Embeddings
        # ------------------------------------------

        embeddings = self.model.encode(

            texts,

            convert_to_numpy=True,

            normalize_embeddings=True,

            show_progress_bar=False

        )

        np.save(

            save_path,

            embeddings

        )

        return {

            "embedding_file": save_path,

            "embeddings": embeddings,

            "total_chunks": len(texts),

            "dimension": embeddings.shape[1]

        }