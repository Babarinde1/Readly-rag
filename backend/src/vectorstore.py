import os
import faiss
import numpy as np
import pickle
from typing import List, Any
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingPipeline

class FaissVectorStore:
    def __init__(self, persist_dir: str = "../data/faiss_store",embedding_model: str = "all-MiniLM-L6-v2"):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.index = None
        self.metadata = []
        self.faiss_path = os.path.join(self.persist_dir, "faiss.index")
        self.meta_path = os.path.join(self.persist_dir, "metadata.pkl")
   ### Extracting text from the documents and generating embeddings
    def build_from_documents(self, documents: List[Any]):
        print(f"[INFO] Building vector store from {len(documents)} raw documents...")
        emb_pipe = EmbeddingPipeline()
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_documents(chunks)
        metadatas = [{
            "text": chunk.page_content,
            'source':chunk.metadata.get("source"),
            'page':chunk.metadata.get("page")} 
            for chunk in chunks]
        
        self.add_embeddings(np.array(embeddings).astype('float32'),metadatas)
        self.save()
        print(f"[INFO] Vector store built and saved to {self.persist_dir}")
  ### Adding embeddings to the Faiss index and saving the index and metadata
    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        print(f"[INFO] Added {embeddings.shape[0]} vectors to Faiss index.")
        # print(f"[INFO] Added {metadatas} metadatas to Faiss index.")
    ### Saving the Faiss index and metadata to disk
    def save(self):
        faiss_path = self.faiss_path
        meta_path = self.meta_path
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[INFO] Saved Faiss index and metadata to {self.persist_dir}")
    #loading the Faiss index and metadata from disk
    def load(self):
        faiss_path = self.faiss_path
        meta_path = self.meta_path
        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"[INFO] Loaded Faiss index and metadata from {self.persist_dir}")
    ## Searching the Faiss index for the top_k nearest neighbors of the query embedding
    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        D, I = self.index.search(query_embedding, top_k)

        results = []
        for idx, dist in zip(I[0], D[0]):
            idx = int(idx)
            dist = float(dist)

            meta = self.metadata[idx] if 0 <= idx < len(self.metadata) else None

        results.append({
            "index": idx,
            "distance": dist,
            "metadata": meta
        })
        return results

    def query(self, query_text: str, top_k: int = 5):
        print(f"[INFO] Querying vector store for: '{query_text}'")
        query_emb = self.model.encode([query_text]).astype('float32')
        return self.search(query_emb, top_k=top_k)

# Example usage
if __name__ == "__main__":
    from data_loader import load_all_documents
    docs = load_all_documents("../data/")
    store = FaissVectorStore("faiss_store")
    store.build_from_documents(docs)
    store.load()
    print(store.query("What is Reinforcement learning?", top_k=3))