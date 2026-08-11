class RelevanceValidator:
    
    def __init__(
        self,
        min_top_score=0.30,
        min_average_score=0.20
    ):
        self.min_top_score = min_top_score
        self.min_average_score = min_average_score

    def validate(self, retrieval_results):

        if len(retrieval_results) == 0:
            return False, "No clauses retrieved."

        top_score = retrieval_results[0]["score"]

        avg_score = sum(
            item["score"]
            for item in retrieval_results
        ) / len(retrieval_results)

        print("\n==============================")
        print("Relevance Validator")
        print("==============================")
        print(f"Top Score     : {top_score:.4f}")
        print(f"Average Score : {avg_score:.4f}")
        print("==============================\n")

        if top_score < self.min_top_score:
            return False, (
                f"Query not related to uploaded documents "
                f"(Top Score = {top_score:.4f})"
            )

        if avg_score < self.min_average_score:
            return False, (
                f"Retrieved clauses are too weak "
                f"(Average Score = {avg_score:.4f})"
            )

        return True, "Relevant query"