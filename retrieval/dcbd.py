class DCBD:
    def __init__(self, coverage_threshold=0.90, min_clauses=3):
        self.coverage_threshold = coverage_threshold
        self.min_clauses = min_clauses

    def select_clauses(self, retrieval_results):
        if len(retrieval_results) == 0:
            return []

        # If retrieved fewer clauses than min_clauses, return all
        if len(retrieval_results) <= self.min_clauses:
            return retrieval_results

        total_similarity = sum(item["score"] for item in retrieval_results)

        if total_similarity == 0:
            return retrieval_results[:self.min_clauses]

        selected = []
        cumulative_similarity = 0

        for idx, item in enumerate(retrieval_results):
            selected.append(item)
            cumulative_similarity += item["score"]
            coverage = cumulative_similarity / total_similarity

            # Stop ONLY if threshold is met AND we have at least min_clauses
            if coverage >= self.coverage_threshold and len(selected) >= self.min_clauses:
                break

        return selected