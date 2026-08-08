import os
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.data_loader import load_document
from src.search import RAGSearch


# =========================================================
# Application
# =========================================================

app = FastAPI(
    title="READLY API",
    description="Document Question Answering System",
    version="1.0.0"
)


# =========================================================
# CORS
# =========================================================

frontend_url = os.getenv(
    "FRONTEND_URL",
    "http://localhost:5173"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        frontend_url,
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# Storage
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# Document sessions
# =========================================================

rag_sessions = {}


# =========================================================
# Request models
# =========================================================

class QuestionRequest(BaseModel):
    document_id: str
    question: str


# =========================================================
# Health check
# =========================================================

@app.get("/")
def root():
    return {
        "message": "READLY API is running",
        "status": "healthy"
    }


# =========================================================
# Upload document
# =========================================================

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...)
):

    allowed_extensions = {
        ".pdf",
        ".docx",
        ".txt"
    }

    # -----------------------------------------------------
    # Validate filename
    # -----------------------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No filename provided."
        )

    # -----------------------------------------------------
    # Validate extension
    # -----------------------------------------------------

    extension = Path(
        file.filename
    ).suffix.lower()

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. "
                "Upload PDF, DOCX or TXT."
            )
        )

    # -----------------------------------------------------
    # Generate document ID
    # -----------------------------------------------------

    document_id = str(
        uuid.uuid4()
    )

    filename = Path(
        file.filename
    ).name

    saved_filename = (
        f"{document_id}{extension}"
    )

    file_path = (
        UPLOAD_DIR / saved_filename
    )

    try:

        # -------------------------------------------------
        # Save uploaded file
        # -------------------------------------------------

        with open(
            file_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        print(
            f"[INFO] Saved document: {file_path}"
        )

        # -------------------------------------------------
        # Load document
        # -------------------------------------------------

        documents = load_document(
            str(file_path)
        )

        if not documents:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not extract text "
                    "from document."
                )
            )

        print(
            f"[INFO] Extracted "
            f"{len(documents)} document sections."
        )

        # -------------------------------------------------
        # Create document-specific RAG
        # -------------------------------------------------

        rag = RAGSearch(
            documents=documents
        )

        # -------------------------------------------------
        # Store session
        # -------------------------------------------------

        rag_sessions[document_id] = {
            "rag": rag,
            "filename": filename,
            "path": str(file_path)
        }

        print(
            f"[INFO] Created session: "
            f"{document_id}"
        )

        return {
            "document_id": document_id,
            "filename": filename,
            "message": "Document uploaded successfully."
        }

    except HTTPException:
        raise

    except Exception as e:

        if file_path.exists():
            file_path.unlink()

        print(
            f"[ERROR] Upload failed: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to process document."
        )


# =========================================================
# Ask question
# =========================================================

@app.post("/ask")
async def ask_question(
    request: QuestionRequest
):

    # -----------------------------------------------------
    # Find document session
    # -----------------------------------------------------

    session = rag_sessions.get(
        request.document_id
    )

    if session is None:

        raise HTTPException(
            status_code=404,
            detail=(
                "Document session not found. "
                "Please upload the document again."
            )
        )

    # -----------------------------------------------------
    # Validate question
    # -----------------------------------------------------

    question = request.question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:

        rag = session["rag"]

        # -------------------------------------------------
        # Ask RAG system
        # -------------------------------------------------

        result = rag.search_and_answer(
            question,
            top_k=5
        )

        return {
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"]
        }

    except Exception as e:

        print(
            f"[ERROR] Question failed: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to answer question."
        )


# =========================================================
# Optional: document information
# =========================================================

@app.get("/document/{document_id}")
async def document_info(
    document_id: str
):

    session = rag_sessions.get(
        document_id
    )

    if session is None:

        raise HTTPException(
            status_code=404,
            detail="Document session not found."
        )

    return {
        "document_id": document_id,
        "filename": session["filename"]
    }