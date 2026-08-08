from typing import List, Any
import langchain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from src.data_loader import load_all_documents

class EmbeddingPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            length_function=len, 
            separators=["\n\n", "\n", " ", ""]
            )
        

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        """ Split the Texts into chunks using the text splitter. """
        chunks = self.text_splitter.split_documents(documents)
        print(f"[DEBUG]{len(documents)} Split into {len(chunks)} chunks.")
        return chunks
    
    def embed_documents(self, documents: List[Any]) -> np.ndarray:
        """ Generate embeddings for the given documents. """
        texts = [doc.page_content for doc in documents]
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        print(f"[DEBUG] Generated embeddings of shape: {embeddings.shape}")
        return embeddings

if __name__ == "__main__":
    try:
        # Load documents
        documents = load_all_documents("../data/")
        # Chunk documents
        chunked_docs = EmbeddingPipeline().chunk_documents(documents)
        # Generate embeddings
        embeddings = EmbeddingPipeline().embed_documents(chunked_docs)
        print("[INFO] Example embedding:", embeddings[0] if len(embeddings) > 0 else None)
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
