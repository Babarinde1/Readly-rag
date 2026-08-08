import os
import time
import faiss
import numpy as np
import pickle
from typing import List, Any

import cohere
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


class FaissVectorStore:

    def __init__(
        self,
        persist_dir: str = "../data/faiss_store",
        embedding_model: str = "embed-english-v3.0"
    ):
        self.persist_dir = persist_dir

        os.makedirs(self.persist_dir, exist_ok=True)

        self.embedding_model = embedding_model

        # Cohere client -- no local model, no torch, huge memory savings
        cohere_api_key = os.getenv("COHERE_API_KEY")

        if not cohere_api_key:
            raise ValueError("COHERE_API_KEY is not configured.")

        self.co = cohere.Client(cohere_api_key)

        self.index = None
        self.metadata = []

        self.faiss_path = os.path.join(self.persist_dir, "faiss.index")
        self.meta_path = os.path.join(self.persist_dir, "metadata.pkl")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    # ---------------------------------------------------------
    # Embed helper (batches + retries to survive flaky connections)
    # ---------------------------------------------------------

    def _embed(self, texts: List[str], input_type: str, max_retries: int = 3) -> np.ndarray:

        batch_size = 96  # Cohere embed endpoint limit per call
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            for attempt in range(1, max_retries + 1):
                try:
                    response = self.co.embed(
                        texts=batch,
                        model=self.embedding_model,
                        input_type=input_type
                    )
                    all_embeddings.extend(response.embeddings)
                    break  # success -- move to next batch

                except Exception as e:
                    if attempt == max_retries:
                        print(f"[ERROR] Embedding batch {i // batch_size + 1} failed after {max_retries} attempts: {e}")
                        raise

                    wait_time = 2 ** attempt  # 2s, 4s, 8s
                    print(
                        f"[WARN] Embedding batch {i // batch_size + 1} failed "
                        f"(attempt {attempt}/{max_retries}): {e}. Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)

        return np.asarray(all_embeddings, dtype="float32")

    # ---------------------------------------------------------
    # Build vector store
    # ---------------------------------------------------------

    def build_from_documents(self, documents: List[Any]):

        print(f"[INFO] Building vector store from {len(documents)} raw documents...")

        chunks = self.text_splitter.split_documents(documents)

        print(f"[DEBUG] {len(documents)} documents split into {len(chunks)} chunks.")

        if not chunks:
            raise ValueError("No text chunks were generated from the documents.")

        texts = [chunk.page_content for chunk in chunks]

        # Documents get "search_document" input type
        embeddings = self._embed(texts, input_type="search_document")

        print(f"[DEBUG] Generated embeddings of shape: {embeddings.shape}")

        metadatas = []

        for chunk in chunks:
            metadata = getattr(chunk, "metadata", {}) or {}
            metadatas.append({
                "text": chunk.page_content,
                "source": metadata.get("source"),
                "page": metadata.get("page")
            })

        self.add_embeddings(embeddings, metadatas)
        self.save()

        print(f"[INFO] Vector store built and saved to {self.persist_dir}")

    # ---------------------------------------------------------
    # Add embeddings
    # ---------------------------------------------------------

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D NumPy array.")

        dim = embeddings.shape[1]

        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)

        if self.index.d != dim:
            raise ValueError(
                f"Embedding dimension mismatch. "
                f"FAISS expects {self.index.d}, but received {dim}."
            )

        self.index.add(embeddings)

        if metadatas:
            if len(metadatas) != embeddings.shape[0]:
                raise ValueError("Number of metadata entries must match number of embeddings.")
            self.metadata.extend(metadatas)

        print(f"[INFO] Added {embeddings.shape[0]} vectors to Faiss index.")

    # ---------------------------------------------------------
    # Save / Load (unchanged)
    # ---------------------------------------------------------

    def save(self):
        if self.index is None:
            raise ValueError("Cannot save an empty FAISS index.")

        faiss.write_index(self.index, self.faiss_path)

        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

        print(f"[INFO] Saved Faiss index and metadata to {self.persist_dir}")

    def load(self):
        if not os.path.exists(self.faiss_path):
            raise FileNotFoundError(f"FAISS index not found: {self.faiss_path}")

        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"Metadata file not found: {self.meta_path}")

        self.index = faiss.read_index(self.faiss_path)

        with open(self.meta_path, "rb") as f:
            self.metadata = pickle.load(f)

        print(f"[INFO] Loaded Faiss index and metadata from {self.persist_dir}")

    # ---------------------------------------------------------
    # Search (unchanged)
    # ---------------------------------------------------------

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        if self.index is None:
            raise ValueError("FAISS index has not been initialized.")

        if self.index.ntotal == 0:
            return []

        top_k = min(top_k, self.index.ntotal)

        query_embedding = np.asarray(query_embedding, dtype="float32")

        distances, indices = self.index.search(query_embedding, top_k)

        results = []

        for idx, dist in zip(indices[0], distances[0]):
            idx = int(idx)
            dist = float(dist)

            meta = self.metadata[idx] if 0 <= idx < len(self.metadata) else None

            results.append({"index": idx, "distance": dist, "metadata": meta})

        return results

    # ---------------------------------------------------------
    # Query
    # ---------------------------------------------------------

    def query(self, query_text: str, top_k: int = 5):
        print(f"[INFO] Querying vector store for: '{query_text}'")

        # Queries get "search_query" input type -- this asymmetric
        # embedding is part of why Cohere v3 retrieves better than
        # a single-purpose model like MiniLM
        query_embedding = self._embed([query_text], input_type="search_query")

        return self.search(query_embedding, top_k=top_k)


if __name__ == "__main__":
    from src.data_loader import load_all_documents

    docs = load_all_documents("../data/")

    store = FaissVectorStore("faiss_store")
    store.build_from_documents(docs)
    store.load()

    results = store.query("What is Reinforcement learning?", top_k=3)

    for result in results:
        print("\n-------------------------")
        print("Index:", result["index"])
        print("Distance:", result["distance"])
        print("Metadata:", result["metadata"])