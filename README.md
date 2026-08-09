# 📖 Readly

**Upload a document. Ask it anything. Get answers grounded only in what you gave it.**

Readly is a full-stack Retrieval-Augmented Generation (RAG) web app that lets users upload a PDF, Word document, or text file and have a real conversation with its contents — no hallucinated answers, no outside knowledge, just accurate, source-cited responses pulled directly from the document.
<img width="1260" height="560" alt="Screenshot 2026-08-08 223704" src="https://github.com/user-attachments/assets/109bd263-5656-48a6-be0f-96ce5118434b" />

🔗 **Live app:** [readly-doc.vercel.app](https://readly-doc.vercel.app)
🔗 **API:** [readly-rag.onrender.com](https://readly-rag.onrender.com)

> ⚠️ The backend runs on Render's free tier, which spins down after periods of inactivity. The first request after idle time may take 30–60 seconds while the server wakes up — this is expected, not a bug.

---

## ✨ Features

- 📄 **Multi-format upload** — PDF, DOCX, and TXT files supported
- 💬 **Grounded Q&A** — answers are generated strictly from the uploaded document, with a clear fallback when the document doesn't contain the answer
- 📌 **Source citations** — every answer links back to the document and page it came from
- 🗂️ **Session-scoped indexing** — each upload builds its own isolated vector index, so documents and conversations never mix between users
- 🌓 **Dark / light mode** — theme toggle with system-preference detection and persistence
- 🎨 **Animated, responsive UI** — soft drifting color background, glassmorphism hover states, and a hamburger-drawer navigation that adapts across desktop, tablet, and mobile
- 🗑️ **Full session control** — delete individual chat messages or remove the active document at any time

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React (Vite), custom CSS (no framework) |
| Backend | FastAPI (Python) |
| Vector Store | FAISS |
| Embeddings | Cohere (`embed-english-v3.0`) |
| LLM | Groq (`openai/gpt-oss-120b`) |
| Frontend Hosting | Vercel |
| Backend Hosting | Render |

---

## 🧠 How It Works

1. **Upload** — a document is parsed and split into overlapping text chunks
2. **Embed** — each chunk is converted into a vector using Cohere's embedding API
3. **Index** — vectors are stored in an in-memory FAISS index, scoped to that upload session
4. **Ask** — a question is embedded the same way, and FAISS retrieves the most relevant chunks
5. **Answer** — the retrieved context is passed to a Groq-hosted LLM with strict instructions to answer only from that context, and the response is returned with source citations

```
Upload → Parse → Chunk → Embed (Cohere) → FAISS Index
                                                │
Question → Embed (Cohere) → FAISS Search ──────┘
                                │
                      Retrieved Context
                                │
                      Groq LLM → Answer + Sources
```

---

## 🚀 Getting Started Locally

### Prerequisites
- Python 3.10+
- Node.js 18+
- A [Cohere API key](https://dashboard.cohere.com/welcome/register)
- A [Groq API key](https://console.groq.com)

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

Create a `.env` file in `backend/`:
```
GROQ_API_KEY=your_groq_key_here
COHERE_API_KEY=your_cohere_key_here
FRONTEND_URL=http://localhost:5173
```

Run the API:
```bash
uvicorn main:app --reload
```
The backend will be available at `http://127.0.0.1:8000`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```
The app will be available at `http://localhost:5173`.

---

## 🌍 Deployment

- **Backend** is deployed on [Render](https://render.com) as a web service, with `GROQ_API_KEY`, `COHERE_API_KEY`, and `FRONTEND_URL` set as environment variables.
- **Frontend** is deployed on [Vercel](https://vercel.com), with `API_URL` in the frontend pointing to the live Render backend URL.

---

## 📂 Project Structure

```
readly/
├── backend/
│   ├── main.py              # FastAPI app & routes
│   ├── src/
│   │   ├── data_loader.py   # Document parsing (PDF/DOCX/TXT)
│   │   ├── vectorstore.py   # FAISS + Cohere embedding logic
│   │   └── search.py        # RAG query + answer generation
│   └── requirements.txt
│
└── frontend/
    ├── src/
    │   ├── App.jsx
    │   ├── App.css
    │   └── ThreeBackground.jsx
    └── package.json
```

---

## 🗺️ Roadmap

- [ ] Persistent chat history across sessions
- [ ] Multi-document support per session
- [ ] Streaming responses
- [ ] Support for additional file types (e.g. Markdown, CSV)

---

## 👤 Author

Built by **Babarinde Johnson Omotayo**
Self-taught AI/ML developer | Computer educator

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
