import re
import numpy as np


class FaithfulnessEvaluator:

    def __init__(self, embedding_model):

        self.embedding_model = embedding_model

        # BERTScore is optional.
        # Install using:
        # pip install bert-score

        try:

            from bert_score import score

            self._bert_score = score

            self.bert_score_available = True

        except ImportError:

            self._bert_score = None

            self.bert_score_available = False


    # ============================================================
    # JACCARD TOKENIZATION
    # ============================================================

    def _tokenize_for_jaccard(self, text):

        words = re.findall(

            r"\b[a-zA-Z0-9]+\b",

            text.lower()

        )

        return set(words)


    # ============================================================
    # JACCARD SIMILARITY
    # ============================================================

    def _calculate_jaccard(

        self,

        answer,

        reference_answer

    ):

        answer_words = self._tokenize_for_jaccard(

            answer

        )

        reference_words = self._tokenize_for_jaccard(

            reference_answer

        )

        if not answer_words or not reference_words:

            return 0.0

        intersection = (

            answer_words

            &

            reference_words

        )

        union = (

            answer_words

            |

            reference_words

        )

        if not union:

            return 0.0

        similarity = (

            len(intersection)

            /

            len(union)

        )

        return similarity


    # ============================================================
    # MAIN EVALUATION
    # ============================================================

    def evaluate(

        self,

        answer,

        selected_clauses,

        reference_answer=None

    ):

        # ========================================================
        # CHECK ANSWER
        # ========================================================

        if not answer or not answer.strip():

            return {

                "faithfulness_score": 0.0,

                "hallucination_rate": 100.0,

                "semantic_similarity": 0.0,

                "bert_precision": 0.0,

                "bert_recall": 0.0,

                "bert_f1": 0.0,

                "jaccard_similarity": 0.0,

                "answer_quality":
                    "No answer generated"

            }


        # ========================================================
        # CHECK RETRIEVED CONTEXT
        # ========================================================

        if not selected_clauses:

            return {

                "faithfulness_score": 0.0,

                "hallucination_rate": 100.0,

                "semantic_similarity": 0.0,

                "bert_precision": 0.0,

                "bert_recall": 0.0,

                "bert_f1": 0.0,

                "jaccard_similarity": 0.0,

                "answer_quality":
                    "No supporting legal clauses"

            }


        # ========================================================
        # COMBINE ALL SELECTED CLAUSES
        # ========================================================

        context = "\n\n".join(

            clause.get(

                "text",

                ""

            )

            for clause in selected_clauses

        )


        # ========================================================
        # 1. COSINE SIMILARITY
        # ========================================================

        answer_embedding = (

            self.embedding_model.model.encode(

                [answer],

                convert_to_numpy=True,

                normalize_embeddings=True,

                show_progress_bar=False

            )

        )


        context_embedding = (

            self.embedding_model.model.encode(

                [context],

                convert_to_numpy=True,

                normalize_embeddings=True,

                show_progress_bar=False

            )

        )


        # Because both embeddings are normalized,
        # their dot product is cosine similarity.

        similarity = float(

            np.dot(

                answer_embedding[0],

                context_embedding[0]

            )

        )


        # ========================================================
        # COSINE SCORE AS PERCENTAGE
        # ========================================================

        faithfulness_score = max(

            0.0,

            min(

                similarity * 100,

                100.0

            )

        )


        # Estimated hallucination risk based on
        # cosine similarity.

        hallucination_rate = (

            100.0

            -

            faithfulness_score

        )


        # ========================================================
        # 2. BERTSCORE
        # ========================================================

        bert_precision = 0.0

        bert_recall = 0.0

        bert_f1 = 0.0


        # BERTScore requires a reference answer.

        if (

            reference_answer

            and

            reference_answer.strip()

            and

            self.bert_score_available

        ):

            try:

                precision, recall, f1 = (

                    self._bert_score(

                        [answer],

                        [reference_answer],

                        lang="en",

                        verbose=False

                    )

                )


                bert_precision = float(

                    precision[0]

                )

                bert_recall = float(

                    recall[0]

                )

                bert_f1 = float(

                    f1[0]

                )


            except Exception:

                # Keep zero if BERTScore fails.

                bert_precision = 0.0

                bert_recall = 0.0

                bert_f1 = 0.0


        # ========================================================
        # 3. JACCARD SIMILARITY
        # ========================================================

        jaccard_similarity = 0.0


        if (

            reference_answer

            and

            reference_answer.strip()

        ):

            jaccard_similarity = (

                self._calculate_jaccard(

                    answer,

                    reference_answer

                )

            )


        # ========================================================
        # ANSWER QUALITY
        # ========================================================

        if faithfulness_score >= 85:

            quality = "Excellent grounding"

        elif faithfulness_score >= 70:

            quality = "Good grounding"

        elif faithfulness_score >= 50:

            quality = "Moderate grounding"

        else:

            quality = (

                "Low grounding / "

                "High hallucination risk"

            )


        # ========================================================
        # RETURN ALL EVALUATION RESULTS
        # ========================================================

        return {

            # -----------------------------------------------
            # COSINE SIMILARITY
            # -----------------------------------------------

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


            # -----------------------------------------------
            # BERTSCORE
            # -----------------------------------------------

            "bert_precision":

                round(

                    bert_precision,

                    4

                ),

            "bert_recall":

                round(

                    bert_recall,

                    4

                ),

            "bert_f1":

                round(

                    bert_f1,

                    4

                ),


            # -----------------------------------------------
            # JACCARD SIMILARITY
            # -----------------------------------------------

            "jaccard_similarity":

                round(

                    jaccard_similarity,

                    4

                ),


            # -----------------------------------------------
            # QUALITY
            # -----------------------------------------------

            "answer_quality":

                quality

        }