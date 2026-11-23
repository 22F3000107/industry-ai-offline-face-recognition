
import numpy as np


class FaceMatcher:
    """Match face embeddings against a reference database."""

    def __init__(self, identities, threshold=0.363):
        if not identities:
            raise ValueError("Reference database is empty.")

        self.identities = identities
        self.threshold = threshold

    def match(self, query_embedding):
        """
        Return the best matching identity or Unknown.

        The threshold is an initial SFace reference value.
        It must be calibrated using independent validation data.
        """

        query = np.asarray(
            query_embedding,
            dtype=np.float32
        ).reshape(-1)

        query_norm = np.linalg.norm(query)

        if query_norm < 1e-12:
            raise ValueError("Query embedding has zero norm.")

        query = query / query_norm

        scores = {}

        for name, reference_embedding in self.identities.items():
            reference = np.asarray(
                reference_embedding,
                dtype=np.float32
            ).reshape(-1)

            if reference.shape != query.shape:
                raise ValueError(
                    f"Embedding dimension mismatch for {name}"
                )

            reference_norm = np.linalg.norm(reference)

            if reference_norm < 1e-12:
                continue

            reference = reference / reference_norm

            similarity = float(np.dot(query, reference))
            scores[name] = similarity

        if not scores:
            raise ValueError("No valid reference embeddings.")

        best_name = max(scores, key=scores.get)
        best_score = scores[best_name]

        if best_score >= self.threshold:
            identity = best_name
        else:
            identity = "Unknown"

        return {
            "identity": identity,
            "best_match": best_name,
            "similarity": best_score,
            "threshold": self.threshold,
            "all_scores": scores,
        }
