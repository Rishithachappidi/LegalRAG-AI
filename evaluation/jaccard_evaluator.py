import re


class JaccardEvaluator:

    def __init__(self):
        pass

    def _tokenize(self, text):

        words = re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower()
        )

        return set(words)

    def evaluate(self, answer, reference_answer):

        if not answer or not answer.strip():
            return {
                "jaccard_similarity": 0.0
            }

        if not reference_answer or not reference_answer.strip():
            return {
                "jaccard_similarity": 0.0
            }

        answer_words = self._tokenize(answer)

        reference_words = self._tokenize(
            reference_answer
        )

        if not answer_words or not reference_words:

            return {
                "jaccard_similarity": 0.0
            }

        intersection = (
            answer_words & reference_words
        )

        union = (
            answer_words | reference_words
        )

        similarity = (
            len(intersection) /
            len(union)
        )

        return {
            "jaccard_similarity": round(
                similarity,
                4
            )
        }