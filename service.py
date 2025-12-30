import bentoml
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the embedding model ONCE
model = SentenceTransformer("all-MiniLM-L6-v2")

@bentoml.service(name="document_change_analyzer")
class DocumentChangeService:

    @bentoml.api
    def analyze(self, payload: dict) -> dict:
        doc_v1 = payload["doc_v1"]
        doc_v2 = payload["doc_v2"]

        # Generate embeddings
        emb_v1 = model.encode([doc_v1])
        emb_v2 = model.encode([doc_v2])

        # Compute cosine similarity
        similarity = cosine_similarity(emb_v1, emb_v2)[0][0]

        # Simple decision rule
        if similarity > 0.85:
            change_type = "minor_or_no_change"
        else:
            change_type = "major_change"

        return {
            "similarity_score": float(similarity),
            "change_type": change_type
        }
