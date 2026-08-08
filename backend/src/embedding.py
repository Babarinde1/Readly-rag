from typing import List, Any
import os
import numpy as np
import cohere
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


class EmbeddingPipeline:
    """
    Handles:
    1. Document chunking
    2. Text embedding via Cohere API (no local model, no torch)
    """

    _client = None

    def __init__(
        self,
        model_name: str = "embed-english-v3.0",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if EmbeddingPipeline._client is None:
            cohere_api_key = os.getenv("COHERE_API_KEY")

            if not cohere_api_key:
                raise ValueError("COHERE_API_KEY is not configured.")

            print("[INFO] Initializing Cohere client.")
            EmbeddingPipeline._client = cohere.Client(cohere_api_key)

        self.client = EmbeddingPipeline._client

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        chunks = self.text_splitter.split_documents(documents)
        print(f"[DEBUG] {len(documents)} documents split into {len(chunks)} chunks.")
        return chunks

    def _embed(self, texts: List[str], input_type: str) -> np.ndarray:
        batch_size = 96
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embed(
                texts=batch, model=self.model_name, input_type=input_type
            )
            all_embeddings.extend(response.embeddings)

        return np.asarray(all_embeddings, dtype=np.float32)

    def embed_documents(self, documents: List[Any]) -> np.ndarray:
        if not documents:
            return np.empty((0, 1024), dtype=np.float32)

        texts = [doc.page_content for doc in documents]
        embeddings = self._embed(texts, input_type="search_document")

        print(f"[DEBUG] Generated embeddings of shape: {embeddings.shape}")
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        return self._embed([query], input_type="search_query")


if __name__ == "__main__":
    try:
        from src.data_loader import load_all_documents

        documents = load_all_documents("../data/")
        pipeline = EmbeddingPipeline()
        chunks = pipeline.chunk_documents(documents)
        embeddings = pipeline.embed_documents(chunks)

        print("[INFO] Number of chunks:", len(chunks))
        print("[INFO] Embedding shape:", embeddings.shape)

        if len(embeddings) > 0:
            print("[INFO] First embedding:", embeddings[0])

    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")