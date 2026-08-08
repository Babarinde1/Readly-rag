import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.vectorstore import FaissVectorStore
from src.data_loader import load_all_documents


# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

load_dotenv(Path(__file__).resolve().parent / ".env")

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


# ---------------------------------------------------------
# Markdown cleaning
# ---------------------------------------------------------

def clean_answer(text: str) -> str:
    """
    Clean unnecessary Markdown formatting while preserving
    numbered and bulleted lists.
    """

    if not text:
        return ""

    # Remove bold
    # **important** -> important
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)

    # Remove underscore bold
    # __important__ -> important
    text = re.sub(r"__(.*?)__", r"\1", text)

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
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ---------------------------------------------------------
# RAG Search
# ---------------------------------------------------------

class RAGSearch:

    def __init__(
        self,
        persist_dir: str = "../data/faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "openai/gpt-oss-120b",
        documents=None
    ):

        """
        If documents are provided, build a fresh vector store
        for the uploaded document.

        Otherwise, use the existing persisted vector store.
        """

        self.vectorstore = FaissVectorStore(
            persist_dir,
            embedding_model
        )

        # -------------------------------------------------
        # Uploaded document
        # -------------------------------------------------

        if documents is not None:

            self.vectorstore.build_from_documents(
                documents
            )

        # -------------------------------------------------
        # Existing document store
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

            if not (
                os.path.exists(faiss_path)
                and os.path.exists(meta_path)
            ):

                docs = load_all_documents("../data")

                self.vectorstore.build_from_documents(
                    docs
                )

            else:

                self.vectorstore.load()

        # -------------------------------------------------
        # Groq LLM
        # -------------------------------------------------

        groq_api_key = os.getenv(
            "GROQ_API_KEY"
        )

        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=llm_model
        )

        print(
            f"[INFO] Groq LLM initialized: {llm_model}"
        )


    # -----------------------------------------------------
    # Question answering
    # -----------------------------------------------------

    def search_and_answer(
        self,
        query: str,
        top_k: int = 5
    ):

        try:

            # ---------------------------------------------
            # Retrieve relevant chunks
            # ---------------------------------------------

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


            # ---------------------------------------------
            # Build context
            # ---------------------------------------------

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

                # -----------------------------------------
                # Context for LLM
                # -----------------------------------------

                context_parts.append(
                    f"""
[Source {i}]
Document: {source or "Unknown"}
Page: {page if page is not None else "N/A"}

{text}
"""
                )

                # -----------------------------------------
                # Convert NumPy values to Python values
                # -----------------------------------------

                distance = result.get(
                    "distance"
                )

                if distance is not None:
                    distance = float(distance)

                if page is not None:
                    try:
                        page = int(page)
                    except (ValueError, TypeError):
                        pass

                sources.append(
                    {
                        "source": source,
                        "page": page,
                        "distance": distance
                    }
                )


            context = "\n\n".join(
                context_parts
            )


            # ---------------------------------------------
            # Prompt
            # ---------------------------------------------

            prompt = f"""
You are READLY, a document question-answering assistant.

Your job is to answer the user's question using ONLY
the information contained in the provided document context.

IMPORTANT RULES:

1. Do not use outside knowledge.
2. Do not invent information.
3. If the answer cannot be found in the document,
   clearly say that the document does not contain
   enough information.
4. Give a clear and concise answer.
5. Preserve useful numbered and bulleted lists.
6. Do not create fake sources.
7. Do not create fake page numbers.
8. Only information present in the context may be used.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{query}

ANSWER:
"""


            # ---------------------------------------------
            # Generate answer
            # ---------------------------------------------

            response = self.llm.invoke(
                prompt
            )

            answer = clean_answer(
                response.content
            )


            # ---------------------------------------------
            # Return result
            # ---------------------------------------------

            return {
                "answer": answer,
                "sources": sources
            }


        except Exception as e:

            print(
                f"[ERROR] Error during "
                f"search and answer: {e}"
            )

            return {
                "answer": (
                    "Something went wrong while "
                    "processing your question."
                ),
                "sources": []
            }


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

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