from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
import torch


class EmbeddingModel:

    def __init__(self, model_name):

        self.model = None

        try:
            print(f"\n[MODEL] Loading embedding model: {model_name}")

            # GPU support
            device = "cuda" if torch.cuda.is_available() else "cpu"

            print(f"[MODEL] Using device: {device}")

            self.model = SentenceTransformer(
                model_name,
                device=device
            )

            print("[MODEL] Embedding model loaded successfully")

        except Exception as e:

            print("[MODEL ERROR]", str(e))

            raise RuntimeError(
                "Embedding model initialization failed"
            )

    # --------------------------------------------------
    # ENCODE
    # --------------------------------------------------
    def encode(self, texts: List[str]) -> np.ndarray:

        """
        Convert text into embeddings.

        Supports:
        - single string
        - list of strings

        Returns:
        numpy float32 embeddings
        """

        # -------------------------
        # SAFETY CHECKS
        # -------------------------

        if texts is None:
            raise ValueError("encode() received None")

        if self.model is None:
            raise RuntimeError("Embedding model not initialized")

        # Convert single string -> list
        if isinstance(texts, str):
            texts = [texts]

        if not isinstance(texts, list):
            raise ValueError(
                "Input must be string or list"
            )

        # Remove empty texts
        texts = [
            str(t).strip()
            for t in texts
            if str(t).strip()
        ]

        if not texts:
            raise ValueError("No valid text provided")

        # -------------------------
        # BGE IMPORTANT REQUIREMENT
        # -------------------------

        """
        BGE models perform MUCH better when queries
        are prefixed with instruction text.
        """

        processed = []

        for text in texts:

            # Detect short query vs document chunk
            if len(text.split()) <= 15:

                # Query embedding
                processed.append(
                    f"Represent this sentence for searching relevant passages: {text}"
                )

            else:
                # Document chunk embedding
                processed.append(text)

        # -------------------------
        # GENERATE EMBEDDINGS
        # -------------------------

        embeddings = self.model.encode(
            processed,

            convert_to_numpy=True,

            normalize_embeddings=True,

            batch_size=32,

            show_progress_bar=False
        )

        return np.array(
            embeddings,
            dtype=np.float32
        )