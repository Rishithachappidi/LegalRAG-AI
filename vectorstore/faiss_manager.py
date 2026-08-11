import os
import json
import faiss
import numpy as np


class FAISSManager:

    def __init__(self, output_folder="faiss_index"):

        self.output_folder = output_folder

        os.makedirs(
            self.output_folder,
            exist_ok=True
        )

    # --------------------------------------------------
    # Build or Load FAISS Index
    # --------------------------------------------------

    def build_index(

        self,

        embedding_file,

        chunk_file,

        document_name

    ):

        index_path = os.path.join(

            self.output_folder,

            document_name + ".index"

        )

        metadata_path = os.path.join(

            self.output_folder,

            document_name + "_metadata.json"

        )

        # =====================================
        # If Index Already Exists
        # =====================================

        if os.path.exists(index_path) and os.path.exists(metadata_path):

            index = faiss.read_index(index_path)

            return {

                "index": index,

                "vectors": index.ntotal,

                "dimension": index.d,

                "index_path": index_path,

                "metadata_path": metadata_path

            }

        # =====================================
        # Load Embeddings
        # =====================================

        embeddings = np.load(

            embedding_file

        ).astype("float32")

        dimension = embeddings.shape[1]

        # =====================================
        # Build Index
        # =====================================

        index = faiss.IndexFlatIP(

            dimension

        )

        index.add(

            embeddings

        )

        # =====================================
        # Save Index
        # =====================================

        faiss.write_index(

            index,

            index_path

        )

        # =====================================
        # Save Metadata
        # =====================================

        with open(

            chunk_file,

            encoding="utf-8"

        ) as f:

            metadata = json.load(f)

        with open(

            metadata_path,

            "w",

            encoding="utf-8"

        ) as f:

            json.dump(

                metadata,

                f,

                indent=4,

                ensure_ascii=False

            )

        return {

            "index": index,

            "vectors": embeddings.shape[0],

            "dimension": dimension,

            "index_path": index_path,

            "metadata_path": metadata_path

        }