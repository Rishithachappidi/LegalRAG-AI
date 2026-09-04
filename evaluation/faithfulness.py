import re
import numpy as np


class FaithfulnessEvaluator:

    def __init__(
        self,
        embedding_model,
        bert_model_type="distilbert-base-uncased"
    ):

        self.embedding_model = embedding_model

        # ------------------------------------------------------
        # BERTScore is optional.
        # Install using:
        # pip install bert-score
        #
        # PERFORMANCE NOTE:
        # A persistent BERTScorer object is created ONCE here
        # (at evaluator construction time) instead of calling the
        # `bert_score.score()` convenience function on every
        # question. The convenience function reloads the full
        # transformer model from disk on every single call, which
        # is what caused the multi-second/minute delays and the
        # repeated "downloading/reconstructing model" behaviour.
        #
        # A smaller model ("distilbert-base-uncased") is used by
        # default instead of BERTScore's usual default
        # ("roberta-large", ~1.4GB) so the one-time download and
        # every subsequent scoring pass are both much faster, at a
        # small cost in BERTScore's own precision.
        # ------------------------------------------------------

        self.bert_model_type = bert_model_type

        self._bert_scorer = None
        self.bert_score_available = False

        try:

            from bert_score import BERTScorer

            self._bert_scorer = BERTScorer(
                model_type=self.bert_model_type,
                lang="en",
                rescale_with_baseline=False
            )

            self.bert_score_available = True

        except Exception:

            self._bert_scorer = None
            self.bert_score_available = False


    # ============================================================
    # SENTENCE SPLITTING (for per-sentence cosine faithfulness)
    # ============================================================

    def _split_sentences(self, text):
        """
        Split the answer into sentences so each one can be checked
        individually against the retrieved clauses, instead of
        collapsing the whole answer into a single embedding.

        Very short fragments (e.g. a lone "[Clause 3]" citation) are
        dropped since they carry little semantic content of their
        own and would otherwise drag the average down artificially.
        """

        raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())

        sentences = [
            s.strip()
            for s in raw_sentences
            if len(re.findall(r"[a-zA-Z]{2,}", s)) >= 3
        ]

        if not sentences:
            sentences = [text.strip()] if text.strip() else []

        return sentences


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

    def _calculate_jaccard(self, answer, reference_answer):

        answer_words = self._tokenize_for_jaccard(answer)
        reference_words = self._tokenize_for_jaccard(reference_answer)

        if not answer_words or not reference_words:
            return 0.0

        intersection = answer_words & reference_words
        union = answer_words | reference_words

        if not union:
            return 0.0

        return len(intersection) / len(union)


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
                "bert_hallucination_rate": 100.0,
                "jaccard_similarity": 0.0,
                "jaccard_hallucination_rate": 100.0,
                "answer_quality": "No answer generated"
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
                "bert_hallucination_rate": 100.0,
                "jaccard_similarity": 0.0,
                "jaccard_hallucination_rate": 100.0,
                "answer_quality": "No supporting legal clauses"
            }

        # ========================================================
        # COMBINE ALL SELECTED CLAUSES
        # ========================================================

        context = "\n\n".join(
            clause.get("text", "")
            for clause in selected_clauses
        )

        # ========================================================
        # 1. COSINE SIMILARITY -- PER-SENTENCE, BEST-CLAUSE MATCH
        # ========================================================
        # Previously this compared ONE embedding for the entire
        # answer against ONE embedding for all clauses squashed
        # together. General-purpose sentence embeddings rarely
        # exceed ~0.75-0.85 cosine similarity even for a genuinely
        # faithful paraphrase, and averaging a whole multi-sentence
        # answer into a single vector dilutes it further -- so a
        # well-grounded answer could still show 25-35% "hallucination"
        # purely as a measurement artifact, not real fabrication.
        #
        # Instead: split the answer into sentences, embed each
        # sentence and each individual clause, and for every answer
        # sentence take its similarity to the SINGLE BEST-MATCHING
        # clause (not the blended context). A genuinely supported
        # sentence should closely match at least one specific
        # clause, even if it doesn't closely match the document as
        # a whole. The final score is the average of these
        # per-sentence best-match similarities.

        answer_sentences = self._split_sentences(answer)

        if not answer_sentences:
            answer_sentences = [answer]

        clause_texts = [
            clause.get("text", "")
            for clause in selected_clauses
            if clause.get("text", "").strip()
        ]

        if not clause_texts:
            clause_texts = [context]

        sentence_embeddings = self.embedding_model.model.encode(
            answer_sentences,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        clause_embeddings = self.embedding_model.model.encode(
            clause_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        # Vectors are normalized, so a plain matmul gives cosine
        # similarity directly: shape (num_sentences, num_clauses).
        similarity_matrix = np.matmul(
            sentence_embeddings,
            clause_embeddings.T
        )

        # Best-matching clause per answer sentence.
        best_match_per_sentence = similarity_matrix.max(axis=1)

        similarity = float(np.mean(best_match_per_sentence))

        # Weakest-supported sentence -- useful as a diagnostic for
        # which part of the answer is least grounded, even though
        # the overall score is an average.
        weakest_sentence_similarity = float(np.min(best_match_per_sentence))

        faithfulness_score = max(
            0.0,
            min(similarity * 100, 100.0)
        )

        hallucination_rate = 100.0 - faithfulness_score

        # ========================================================
        # 2. BERTSCORE
        # ========================================================
        # Uses the reference_answer if one is supplied by the
        # caller, otherwise falls back to comparing the answer
        # against the retrieved clause context itself, so this
        # metric always has something to compare against.

        bert_precision = 0.0
        bert_recall = 0.0
        bert_f1 = 0.0

        bert_reference = (
            reference_answer
            if reference_answer and reference_answer.strip()
            else context
        )

        if self.bert_score_available and bert_reference.strip():

            try:

                precision, recall, f1 = self._bert_scorer.score(
                    [answer],
                    [bert_reference]
                )

                bert_precision = float(precision[0])
                bert_recall = float(recall[0])
                bert_f1 = float(f1[0])

            except Exception:

                bert_precision = 0.0
                bert_recall = 0.0
                bert_f1 = 0.0

        bert_hallucination_rate = max(
            0.0,
            min(100.0 - (bert_f1 * 100.0), 100.0)
        )

        # ========================================================
        # 3. JACCARD SIMILARITY
        # ========================================================
        # Same fallback behaviour as BERTScore above.

        jaccard_reference = (
            reference_answer
            if reference_answer and reference_answer.strip()
            else context
        )

        jaccard_similarity = self._calculate_jaccard(
            answer,
            jaccard_reference
        )

        jaccard_hallucination_rate = max(
            0.0,
            min(100.0 - (jaccard_similarity * 100.0), 100.0)
        )

        # ========================================================
        # ANSWER QUALITY (based on the cosine faithfulness score)
        # ========================================================

        if faithfulness_score >= 85:
            quality = "Excellent grounding"
        elif faithfulness_score >= 70:
            quality = "Good grounding"
        elif faithfulness_score >= 50:
            quality = "Moderate grounding"
        else:
            quality = "Low grounding / High hallucination risk"

        # ========================================================
        # RETURN ALL EVALUATION RESULTS
        # ========================================================

        return {

            # -----------------------------------------------
            # METHOD 1: COSINE SIMILARITY (per-sentence, best-clause)
            # -----------------------------------------------
            "faithfulness_score": round(faithfulness_score, 2),
            "hallucination_rate": round(hallucination_rate, 2),
            "semantic_similarity": round(similarity, 4),
            "weakest_sentence_similarity": round(weakest_sentence_similarity, 4),

            # -----------------------------------------------
            # METHOD 2: BERTSCORE
            # -----------------------------------------------
            "bert_precision": round(bert_precision, 4),
            "bert_recall": round(bert_recall, 4),
            "bert_f1": round(bert_f1, 4),
            "bert_hallucination_rate": round(bert_hallucination_rate, 2),

            # -----------------------------------------------
            # METHOD 3: JACCARD SIMILARITY
            # -----------------------------------------------
            "jaccard_similarity": round(jaccard_similarity, 4),
            "jaccard_hallucination_rate": round(jaccard_hallucination_rate, 2),

            # -----------------------------------------------
            # OVERALL QUALITY LABEL
            # -----------------------------------------------
            "answer_quality": quality
        }