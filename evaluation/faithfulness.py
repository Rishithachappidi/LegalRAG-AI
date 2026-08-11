import numpy as np


class FaithfulnessEvaluator:

    def __init__(self, embedding_model):

        self.embedding_model = embedding_model

    def evaluate(
        self,
        answer,
        selected_clauses
    ):

        # -----------------------------------------
        # Check Answer
        # -----------------------------------------

        if not answer or not answer.strip():

            return {
                "faithfulness_score": 0.0,
                "hallucination_rate": 100.0,
                "semantic_similarity": 0.0,
                "answer_quality": "No answer generated"
            }

        # -----------------------------------------
        # Check Retrieved Context
        # -----------------------------------------

        if not selected_clauses:

            return {
                "faithfulness_score": 0.0,
                "hallucination_rate": 100.0,
                "semantic_similarity": 0.0,
                "answer_quality": "No supporting legal clauses"
            }

        # -----------------------------------------
        # Combine ALL selected clauses
        # -----------------------------------------

        context = "\n\n".join(

            clause["text"]

            for clause in selected_clauses

        )

        # -----------------------------------------
        # Encode Complete LLaMA Answer
        # -----------------------------------------

        answer_embedding = self.embedding_model.model.encode(

            [answer],

            convert_to_numpy=True,

            normalize_embeddings=True,

            show_progress_bar=False

        )

        # -----------------------------------------
        # Encode Complete Legal Context
        # -----------------------------------------

        context_embedding = self.embedding_model.model.encode(

            [context],

            convert_to_numpy=True,

            normalize_embeddings=True,

            show_progress_bar=False

        )

        # -----------------------------------------
        # Calculate Cosine Similarity
        # -----------------------------------------

        similarity = float(

            np.dot(

                answer_embedding[0],

                context_embedding[0]

            )

        )

        # -----------------------------------------
        # Convert to Percentage
        # -----------------------------------------

        faithfulness_score = max(

            0.0,

            min(

                similarity * 100,

                100.0

            )

        )

        # -----------------------------------------
        # Estimated Hallucination Risk
        # -----------------------------------------

        hallucination_rate = (

            100.0 - faithfulness_score

        )

        # -----------------------------------------
        # Overall Quality
        # -----------------------------------------

        if faithfulness_score >= 85:

            quality = "Excellent grounding"

        elif faithfulness_score >= 70:

            quality = "Good grounding"

        elif faithfulness_score >= 50:

            quality = "Moderate grounding"

        else:

            quality = "Low grounding / High hallucination risk"

        # -----------------------------------------
        # Return Evaluation
        # -----------------------------------------

        return {

            "faithfulness_score":
                round(
                    faithfulness_score,
                    2
                ),

            "hallucination_rate":
                round(
                    hallucination_rate,
                    2
                ),

            "semantic_similarity":
                round(
                    similarity,
                    4
                ),

            "answer_quality":
                quality

        }