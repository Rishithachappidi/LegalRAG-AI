class DCBD:
    
    def __init__(
        self,
        coverage_threshold=0.75,
        min_clauses=3,
        max_clauses=6
    ):

        self.coverage_threshold = coverage_threshold
        self.min_clauses = min_clauses
        self.max_clauses = max_clauses

    # =====================================================
    # SELECT CLAUSES USING CUMULATIVE SIMILARITY
    # =====================================================

    def select_clauses(
        self,
        retrieval_results
    ):

        # -----------------------------------------
        # No retrieval results
        # -----------------------------------------

        if not retrieval_results:

            return []

        # -----------------------------------------
        # Sort by similarity
        # -----------------------------------------

        results = sorted(
            retrieval_results,
            key=lambda x: x["score"],
            reverse=True
        )

        # -----------------------------------------
        # If fewer than minimum clauses
        # -----------------------------------------

        if len(results) <= self.min_clauses:

            return results

        # -----------------------------------------
        # Calculate total similarity
        # -----------------------------------------

        total_similarity = sum(
            max(float(item["score"]), 0.0)
            for item in results
        )

        if total_similarity <= 0:

            return results[
                :self.min_clauses
            ]

        # -----------------------------------------
        # Cumulative selection
        # -----------------------------------------

        selected = []

        cumulative_similarity = 0.0

        for item in results:

            score = max(
                float(item["score"]),
                0.0
            )

            selected.append(item)

            cumulative_similarity += score

            coverage = (
                cumulative_similarity /
                total_similarity
            )

            # -------------------------------------
            # Stop when coverage is reached
            # -------------------------------------

            if (
                coverage >= self.coverage_threshold
                and
                len(selected) >= self.min_clauses
            ):

                break

            # -------------------------------------
            # Safety limit
            # -------------------------------------

            if len(selected) >= self.max_clauses:

                break

        # -----------------------------------------
        # Guarantee minimum number of clauses
        # -----------------------------------------

        if len(selected) < self.min_clauses:

            selected = results[
                :min(
                    self.min_clauses,
                    len(results)
                )
            ]

        return selected