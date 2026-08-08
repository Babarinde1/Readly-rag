from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingPipeline:
    """
    Handles:
    1. Document chunking
    2. Text embedding

    The embedding model is loaded once and reused.
    """

    _model = None
    _model_name = None

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Load the model only once per process
        if (
            EmbeddingPipeline._model is None
            or EmbeddingPipeline._model_name != model_name
        ):
            print(
                f"[INFO] Loading embedding model: {model_name}"
            )

            EmbeddingPipeline._model = SentenceTransformer(
                model_name
            )

            EmbeddingPipeline._model_name = model_name

            print(
                "[INFO] Embedding model loaded successfully."
            )

        self.model = EmbeddingPipeline._model

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",
                "\n",
                " ",
                ""
            ],
        )

    # --------------------------------------------------
    # Chunk documents
    # --------------------------------------------------

    def chunk_documents(
        self,
        documents: List[Any]
    ) -> List[Any]:

        chunks = self.text_splitter.split_documents(
            documents
        )

        print(
            f"[DEBUG] {len(documents)} documents "
            f"split into {len(chunks)} chunks."
        )

        return chunks

    # --------------------------------------------------
    # Embed documents
    # --------------------------------------------------

    def embed_documents(
        self,
        documents: List[Any]
    ) -> np.ndarray:

        if not documents:
            return np.empty(
                (0, 384),
                dtype=np.float32
            )

        texts = [
            doc.page_content
            for doc in documents
        ]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        embeddings = embeddings.astype(
            np.float32
        )

        print(
            f"[DEBUG] Generated embeddings "
            f"of shape: {embeddings.shape}"
        )

        return embeddings

    # --------------------------------------------------
    # Embed a single query
    # --------------------------------------------------

    def embed_query(
        self,
        query: str
    ) -> np.ndarray:

        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embedding.astype(
            np.float32
        )


# ------------------------------------------------------
# Test
# ------------------------------------------------------

if __name__ == "__main__":

    try:

        from src.data_loader import load_all_documents

        documents = load_all_documents(
            "../data/"
        )

        pipeline = EmbeddingPipeline()

        chunks = pipeline.chunk_documents(
            documents
        )

        embeddings = pipeline.embed_documents(
            chunks
        )

        print(
            "[INFO] Number of chunks:",
            len(chunks)
        )

        print(
            "[INFO] Embedding shape:",
            embeddings.shape
        )

        if len(embeddings) > 0:
            print(
                "[INFO] First embedding:",
                embeddings[0]
            )

    except Exception as e:

        print(
            f"[ERROR] An error occurred: {e}"
        )