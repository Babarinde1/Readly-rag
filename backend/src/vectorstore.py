import os
import faiss
import numpy as np
import pickle
from typing import List, Any

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


class FaissVectorStore:

    def __init__(
        self,
        persist_dir: str = "../data/faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.persist_dir = persist_dir

        os.makedirs(
            self.persist_dir,
            exist_ok=True
        )

        self.embedding_model = embedding_model

        # Load the embedding model ONCE
        self.model = SentenceTransformer(
            embedding_model
        )

        self.index = None
        self.metadata = []

        self.faiss_path = os.path.join(
            self.persist_dir,
            "faiss.index"
        )

        self.meta_path = os.path.join(
            self.persist_dir,
            "metadata.pkl"
        )

        # Text splitter does NOT load another model
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=[
                "\n\n",
                "\n",
                " ",
                ""
            ]
        )

    # ---------------------------------------------------------
    # Build vector store
    # ---------------------------------------------------------

    def build_from_documents(
        self,
        documents: List[Any]
    ):

        print(
            f"[INFO] Building vector store from "
            f"{len(documents)} raw documents..."
        )

        # Split documents into chunks
        chunks = self.text_splitter.split_documents(
            documents
        )

        print(
            f"[DEBUG] {len(documents)} documents "
            f"split into {len(chunks)} chunks."
        )

        if not chunks:
            raise ValueError(
                "No text chunks were generated from the documents."
            )

        # Extract text
        texts = [
            chunk.page_content
            for chunk in chunks
        ]

        # Generate embeddings using the SINGLE model
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        print(
            f"[DEBUG] Generated embeddings "
            f"of shape: {embeddings.shape}"
        )

        # Metadata
        metadatas = []

        for chunk in chunks:

            metadata = getattr(
                chunk,
                "metadata",
                {}
            ) or {}

            metadatas.append(
                {
                    "text": chunk.page_content,
                    "source": metadata.get("source"),
                    "page": metadata.get("page")
                }
            )

        # Create index
        self.add_embeddings(
            embeddings,
            metadatas
        )

        # Save
        self.save()

        print(
            f"[INFO] Vector store built and saved "
            f"to {self.persist_dir}"
        )

    # ---------------------------------------------------------
    # Add embeddings
    # ---------------------------------------------------------

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadatas: List[Any] = None
    ):

        if embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2D NumPy array."
            )

        dim = embeddings.shape[1]

        # Create FAISS index if necessary
        if self.index is None:

            self.index = faiss.IndexFlatL2(
                dim
            )

        # Check dimension consistency
        if self.index.d != dim:

            raise ValueError(
                f"Embedding dimension mismatch. "
                f"FAISS expects {self.index.d}, "
                f"but received {dim}."
            )

        # Add vectors
        self.index.add(
            embeddings
        )

        # Add metadata
        if metadatas:

            if len(metadatas) != embeddings.shape[0]:

                raise ValueError(
                    "Number of metadata entries must "
                    "match number of embeddings."
                )

            self.metadata.extend(
                metadatas
            )

        print(
            f"[INFO] Added "
            f"{embeddings.shape[0]} vectors "
            f"to Faiss index."
        )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    def save(self):

        if self.index is None:

            raise ValueError(
                "Cannot save an empty FAISS index."
            )

        faiss.write_index(
            self.index,
            self.faiss_path
        )

        with open(
            self.meta_path,
            "wb"
        ) as f:

            pickle.dump(
                self.metadata,
                f
            )

        print(
            f"[INFO] Saved Faiss index and "
            f"metadata to {self.persist_dir}"
        )

    # ---------------------------------------------------------
    # Load
    # ---------------------------------------------------------

    def load(self):

        if not os.path.exists(
            self.faiss_path
        ):

            raise FileNotFoundError(
                f"FAISS index not found: "
                f"{self.faiss_path}"
            )

        if not os.path.exists(
            self.meta_path
        ):

            raise FileNotFoundError(
                f"Metadata file not found: "
                f"{self.meta_path}"
            )

        self.index = faiss.read_index(
            self.faiss_path
        )

        with open(
            self.meta_path,
            "rb"
        ) as f:

            self.metadata = pickle.load(
                f
            )

        print(
            f"[INFO] Loaded Faiss index and "
            f"metadata from {self.persist_dir}"
        )

    # ---------------------------------------------------------
    # Search
    # ---------------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ):

        if self.index is None:

            raise ValueError(
                "FAISS index has not been initialized."
            )

        if self.index.ntotal == 0:

            return []

        # Don't ask FAISS for more vectors
        # than actually exist
        top_k = min(
            top_k,
            self.index.ntotal
        )

        # Make sure query is float32
        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        # Search
        distances, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        # IMPORTANT:
        # results.append MUST be inside the loop
        for idx, dist in zip(
            indices[0],
            distances[0]
        ):

            idx = int(idx)
            dist = float(dist)

            if 0 <= idx < len(
                self.metadata
            ):

                meta = self.metadata[idx]

            else:

                meta = None

            results.append(
                {
                    "index": idx,
                    "distance": dist,
                    "metadata": meta
                }
            )

        return results

    # ---------------------------------------------------------
    # Query
    # ---------------------------------------------------------

    def query(
        self,
        query_text: str,
        top_k: int = 5
    ):

        print(
            f"[INFO] Querying vector store for: "
            f"'{query_text}'"
        )

        # Encode query using the SAME model
        query_embedding = self.model.encode(
            [query_text],
            convert_to_numpy=True
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        return self.search(
            query_embedding,
            top_k=top_k
        )


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

if __name__ == "__main__":

    from src.data_loader import load_all_documents

    docs = load_all_documents(
        "../data/"
    )

    store = FaissVectorStore(
        "faiss_store"
    )

    store.build_from_documents(
        docs
    )

    store.load()

    results = store.query(
        "What is Reinforcement learning?",
        top_k=3
    )

    for result in results:

        print("\n-------------------------")

        print(
            "Index:",
            result["index"]
        )

        print(
            "Distance:",
            result["distance"]
        )

        print(
            "Metadata:",
            result["metadata"]
        )