import os
import re

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.vectorstore import FaissVectorStore
from src.data_loader import load_all_documents


# =========================================================
# Environment
# =========================================================

load_dotenv()


# =========================================================
# Markdown cleaning
# =========================================================

def clean_answer(text: str) -> str:


    if not text:
        return ""

    # Remove bold Markdown
    # **important** -> important
    text = re.sub(
        r"\*\*(.*?)\*\*",
        r"\1",
        text,
        flags=re.DOTALL
    )

    # Remove underscore bold
    # __important__ -> important
    text = re.sub(
        r"__(.*?)__",
        r"\1",
        text,
        flags=re.DOTALL
    )

    # Remove italic Markdown
    # *important* -> important
    text = re.sub(
        r"(?<!\*)\*([^*\n]+)\*(?!\*)",
        r"\1",
        text
    )

    # Remove Markdown headings
    # ### Heading -> Heading
    text = re.sub(
        r"^#{1,6}\s*",
        "",
        text,
        flags=re.MULTILINE
    )

    # Normalize excessive blank lines
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# =========================================================
# RAG Search
# =========================================================

class RAGSearch:

    def __init__(
        self,
        persist_dir: str = "../data/faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "openai/gpt-oss-120b",
        documents=None
    ):
        """
        Create a RAG system.

        If documents are provided:
            Build a document-specific FAISS index.

        If documents are not provided:
            Load the existing persisted FAISS index.
        """

        # -------------------------------------------------
        # Create vector store
        # -------------------------------------------------

        self.vectorstore = FaissVectorStore(
            persist_dir=persist_dir,
            embedding_model=embedding_model
        )

        # -------------------------------------------------
        # Uploaded document
        # -------------------------------------------------

        if documents is not None:

            print(
                "[INFO] Building vector store "
                "for uploaded document..."
            )

            self.vectorstore.build_from_documents(
                documents
            )

        # -------------------------------------------------
        # Existing persisted store
        # -------------------------------------------------

        else:

            faiss_path = os.path.join(
                persist_dir,
                "faiss.index"
            )

            meta_path = os.path.join(
                persist_dir,
                "metadata.pkl"
            )

            if (
                not os.path.exists(faiss_path)
                or not os.path.exists(meta_path)
            ):

                print(
                    "[INFO] Persistent FAISS store "
                    "not found. Building one..."
                )

                documents = load_all_documents(
                    "../data"
                )

                self.vectorstore.build_from_documents(
                    documents
                )

            else:

                print(
                    "[INFO] Loading existing "
                    "FAISS store..."
                )

                self.vectorstore.load()

        # -------------------------------------------------
        # Groq
        # -------------------------------------------------

        groq_api_key = os.getenv(
            "GROQ_API_KEY"
        )

        if not groq_api_key:

            raise ValueError(
                "GROQ_API_KEY is not configured."
            )

        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=llm_model
        )

        print(
            f"[INFO] Groq LLM initialized: {llm_model}"
        )


    # =====================================================
    # Question answering
    # =====================================================

    def search_and_answer(
        self,
        query: str,
        top_k: int = 5
    ):

        try:

            query = query.strip()

            if not query:

                return {
                    "answer": "Please enter a question.",
                    "sources": []
                }

            # -------------------------------------------------
            # Retrieve relevant chunks
            # -------------------------------------------------

            results = self.vectorstore.query(
                query,
                top_k=top_k
            )

            if not results:

                return {
                    "answer": (
                        "I couldn't find relevant "
                        "information in the document."
                    ),
                    "sources": []
                }

            # -------------------------------------------------
            # Build context
            # -------------------------------------------------

            context_parts = []
            sources = []

            for i, result in enumerate(
                results,
                start=1
            ):

                metadata = (
                    result.get("metadata")
                    or {}
                )

                text = metadata.get(
                    "text",
                    ""
                )

                source = metadata.get(
                    "source"
                )

                page = metadata.get(
                    "page"
                )

                # Ignore empty chunks
                if not text.strip():
                    continue

                context_parts.append(
                    f"""
[Source {i}]
Document: {source or "Unknown"}
Page: {page if page is not None else "N/A"}

{text}
"""
                )

                # -------------------------------------------------
                # Convert NumPy values to JSON-safe Python values
                # -------------------------------------------------

                distance = result.get(
                    "distance"
                )

                if distance is not None:
                    distance = float(distance)

                if page is not None:

                    try:
                        page = int(page)

                    except (
                        ValueError,
                        TypeError
                    ):
                        pass

                sources.append(
                    {
                        "source": source,
                        "page": page,
                        "distance": distance
                    }
                )

            # -------------------------------------------------
            # Make context
            # -------------------------------------------------

            context = "\n\n".join(
                context_parts
            )

            if not context:

                return {
                    "answer": (
                        "I couldn't find relevant "
                        "information in the document."
                    ),
                    "sources": []
                }

            # -------------------------------------------------
            # Prompt
            # -------------------------------------------------

            prompt = f"""
You are READLY, a document question-answering assistant.

Answer the user's question using ONLY the information
contained in the provided document context.

RULES:

1. Do not use outside knowledge.
2. Do not invent information.
3. If the answer cannot be found in the context,
   say that the document does not contain enough
   information to answer the question.
4. Give a clear and concise answer.
5. Use numbered lists when explaining multiple steps
   or items.
6. Use bullet points when appropriate.
7. Do not use Markdown headings.
8. Do not use bold or italic Markdown.
9. Do not create fake sources.
10. Do not create fake page numbers.
11. Only use page numbers supplied in the context.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{query}

ANSWER:
"""

            # -------------------------------------------------
            # Generate answer
            # -------------------------------------------------

            response = self.llm.invoke(
                prompt
            )

            answer = clean_answer(
                response.content
            )

            # -------------------------------------------------
            # Return
            # -------------------------------------------------

            return {
                "answer": answer,
                "sources": sources
            }

        except Exception as e:

            print(
                "[ERROR] Error during "
                f"search and answer: {e}"
            )

            return {
                "answer": (
                    "Something went wrong while "
                    "processing your question."
                ),
                "sources": []
            }


# =========================================================
# Test
# =========================================================

if __name__ == "__main__":

    rag_search = RAGSearch()

    query = "What is attention mechanism?"

    result = rag_search.search_and_answer(
        query,
        top_k=3
    )

    print("\nANSWER:")
    print(result["answer"])

    print("\nSOURCES:")

    for source in result["sources"]:

        print(
            f"Source: {source['source']}"
        )

        print(
            f"Page: {source['page']}"
        )

        print(
            f"Distance: {source['distance']}"
        )

        print("-" * 50)