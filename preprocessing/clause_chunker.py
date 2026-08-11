import os
import re
import json


class ClauseChunker:

    def __init__(self, output_folder="chunks"):

        self.output_folder = output_folder

        os.makedirs(self.output_folder, exist_ok=True)

    # ----------------------------------------------------
    # Split legal document into clauses
    # ----------------------------------------------------

    def split_into_clauses(self, text):

        text = text.replace("\r", "")

        # Common legal heading patterns
        pattern = (
            r'(?='
            r'\n\s*(?:'
            r'\d+(?:\.\d+)*\.|'          # 1. 2.1 3.2.1
            r'ARTICLE\s+[IVXLC]+|'       # ARTICLE I
            r'SECTION\s+\d+|'            # SECTION 3
            r'CLAUSE\s+\d+'              # CLAUSE 5
            r')'
            r')'
        )

        raw_clauses = re.split(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        clauses = []

        for clause in raw_clauses:

            if clause is None:
                continue

            clause = clause.strip()

            if len(clause) > 40:

                clauses.append(clause)

        return clauses

    # ----------------------------------------------------
    # Create chunks
    # ----------------------------------------------------

    def chunk_document(
        self,
        text,
        document_name
    ):

        clauses = self.split_into_clauses(text)

        output = []

        for idx, clause in enumerate(clauses):

            output.append({

                "document": document_name,

                "clause_id": idx + 1,

                "text": clause

            })

        save_path = os.path.join(

            self.output_folder,

            document_name + "_chunks.json"

        )

        with open(

            save_path,

            "w",

            encoding="utf-8"

        ) as f:

            json.dump(

                output,

                f,

                indent=4,

                ensure_ascii=False

            )

        return output