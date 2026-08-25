from bert_score import score


class BERTScoreEvaluator:

    def __init__(self, model_type="microsoft/deberta-xlarge-mnli"):
        self.model_type = model_type

    def evaluate(self, answer, reference_answer):

        if not answer or not answer.strip():
            return {
                "bert_precision": 0.0,
                "bert_recall": 0.0,
                "bert_f1": 0.0
            }

        if not reference_answer or not reference_answer.strip():
            return {
                "bert_precision": 0.0,
                "bert_recall": 0.0,
                "bert_f1": 0.0
            }

        precision, recall, f1 = score(
            [answer],
            [reference_answer],
            model_type=self.model_type,
            lang="en",
            verbose=False
        )

        return {
            "bert_precision": round(
                float(precision[0]), 4
            ),

            "bert_recall": round(
                float(recall[0]), 4
            ),

            "bert_f1": round(
                float(f1[0]), 4
            )
        }