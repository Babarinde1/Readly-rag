import { useState, useEffect } from "react";
import "./App.css";
import ThreeBackground from "./ThreeBackground";
const API_URL = "http://127.0.0.1:8000";

function App() {
  // -----------------------------
  // Theme
  // -----------------------------
  const [theme, setTheme] = useState(() => {
    const saved = localStorage.getItem("readly-theme");

    if (saved) {
      return saved;
    }

    return window.matchMedia?.("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";
  });

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("readly-theme", theme);
  }, [theme]);

  function toggleTheme() {
    setTheme((current) =>
      current === "dark" ? "light" : "dark"
    );
  }

  // -----------------------------
  // Mobile sidebar (hamburger drawer)
  // -----------------------------
  const [sidebarOpen, setSidebarOpen] = useState(false);

  useEffect(() => {
    document.body.style.overflow = sidebarOpen ? "hidden" : "";
    return () => {
      document.body.style.overflow = "";
    };
  }, [sidebarOpen]);

  function closeSidebar() {
    setSidebarOpen(false);
  }

  // -----------------------------
  // Document state
  // -----------------------------
  const [documentId, setDocumentId] = useState(null);
  const [filename, setFilename] = useState("");
  const [uploading, setUploading] = useState(false);

  // -----------------------------
  // Chat state
  // -----------------------------
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [asking, setAsking] = useState(false);

  // -----------------------------
  // Upload document
  // -----------------------------
  async function handleUpload(event) {
    const file = event.target.files?.[0];

    if (!file) {
      return;
    }

    setUploading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_URL}/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Upload failed");
      }

      setDocumentId(data.document_id);
      setFilename(data.filename || file.name);
      setMessages([]);
      setQuestion("");
      closeSidebar();
    } catch (error) {
      console.error("Upload error:", error);
      alert(error.message || "Failed to upload document.");
    } finally {
      setUploading(false);

      // Allows uploading the same file again
      event.target.value = "";
    }
  }

  // -----------------------------
  // Ask question
  // -----------------------------
  async function askQuestion() {
    if (!question.trim()) {
      return;
    }

    if (!documentId) {
      alert("Please upload a document first.");
      return;
    }

    const currentQuestion = question.trim();

    setQuestion("");

    // Add user message immediately
    setMessages((previous) => [
      ...previous,
      {
        role: "user",
        content: currentQuestion,
      },
    ]);

    setAsking(true);

    try {
      const response = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          document_id: documentId,
          question: currentQuestion,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Question failed");
      }

      setMessages((previous) => [
        ...previous,
        {
          role: "assistant",
          content: data.answer || "No answer was returned.",
          sources: Array.isArray(data.sources)
            ? data.sources
            : [],
        },
      ]);
    } catch (error) {
      console.error("Question error:", error);

      setMessages((previous) => [
        ...previous,
        {
          role: "assistant",
          content:
            "Sorry, something went wrong while answering your question.",
          sources: [],
        },
      ]);
    } finally {
      setAsking(false);
    }
  }

  // -----------------------------
  // Delete document
  // -----------------------------
  async function handleDeleteDocument() {
    if (!documentId) {
      return;
    }

    const confirmed = window.confirm(
      `Remove "${filename}"? This clears the current session.`
    );

    if (!confirmed) {
      return;
    }

    try {
      const response = await fetch(
        `${API_URL}/document/${documentId}`,
        {
          method: "DELETE",
        }
      );

      if (!response.ok) {
        console.warn("Server could not delete the document.");
      }
    } catch (error) {
      console.error(
        "Failed to delete document on server:",
        error
      );
    }

    setDocumentId(null);
    setFilename("");
    setMessages([]);
    setQuestion("");
  }

  // -----------------------------
  // Delete chat message
  // -----------------------------
  function handleDeleteMessage(indexToRemove) {
    setMessages((previous) =>
      previous.filter(
        (_, index) => index !== indexToRemove
      )
    );
  }

  // -----------------------------
  // Enter key
  // -----------------------------
  function handleKeyDown(event) {
    if (
      event.key === "Enter" &&
      !event.shiftKey
    ) {
      event.preventDefault();

      if (!asking) {
        askQuestion();
      }
    }
  }

  // -----------------------------
  // Get filename from source path
  // -----------------------------
  function getSourceName(source) {
    if (!source) {
      return "Document";
    }

    return source.split(/[\\/]/).pop() || "Document";
  }

  // -----------------------------
  // Render
  // -----------------------------
  return (
    <div className="app">
      <ThreeBackground theme={theme} />

      {/* Mobile overlay (click to close drawer) */}
      {sidebarOpen && (
        <div
          className="sidebar-overlay"
          onClick={closeSidebar}
        />
      )}

      {/* Sidebar */}
      <aside className={`sidebar ${sidebarOpen ? "sidebar-open" : ""}`}>

        {/* Brand + mobile close button */}
        <div className="sidebar-top-row">
          <div className="brand">
            <div className="brand-mark">
              R
            </div>

            <span>Readly</span>
          </div>

          <button
            className="sidebar-close"
            onClick={closeSidebar}
            aria-label="Close menu"
          >
            ×
          </button>
        </div>

        {/* Upload button */}
        <button
          className="upload-button"
          onClick={() =>
            document
              .getElementById("file-input")
              ?.click()
          }
          disabled={uploading}
        >
          <span>+</span>

          {uploading
            ? "Processing..."
            : "New document"}
        </button>

        <input
          id="file-input"
          type="file"
          accept=".pdf,.docx,.txt"
          hidden
          onChange={handleUpload}
        />

        {/* Current document */}
        <div className="sidebar-section">

          <div className="section-label">
            DOCUMENT
          </div>

          {filename && (
            <div className="document-item active">

              <div className="document-icon">
                {filename
                  .toLowerCase()
                  .endsWith(".pdf")
                  ? "PDF"
                  : filename
                      .toLowerCase()
                      .endsWith(".docx")
                  ? "DOCX"
                  : "TXT"}
              </div>

              <div className="document-info">

                <div className="document-name">
                  {filename}
                </div>

                <div className="document-status">
                  Ready to explore
                </div>

              </div>

              <button
                className="document-delete"
                onClick={handleDeleteDocument}
                title="Remove document"
              >
                ×
              </button>

            </div>
          )}

        </div>

        {/* Sidebar bottom */}
        <div className="sidebar-bottom">

          <button
            className="sidebar-link"
            onClick={toggleTheme}
          >
            <span className="sidebar-link-icon">
              {theme === "dark" ? "☀" : "🌙"}
            </span>
            <span>
              {theme === "dark"
                ? "Light mode"
                : "Dark mode"}
            </span>
          </button>

          <button className="sidebar-link">
            <span className="sidebar-link-icon">⚙</span>
            <span>Settings</span>
          </button>

          <button className="sidebar-link">
            <span className="sidebar-link-icon">?</span>
            <span>Help</span>
          </button>

        </div>

      </aside>

      {/* Main */}
      <main className="main">

        {/* Top bar */}
        <header className="topbar">

          <div className="topbar-left">

            <button
              className="hamburger-button"
              onClick={() => setSidebarOpen(true)}
              aria-label="Open menu"
            >
              <span />
              <span />
              <span />
            </button>

            <div>

              <div className="eyebrow">
                DOCUMENT READER
              </div>

              <h1>
                {filename ||
                  "Your document, understood."}
              </h1>

            </div>

          </div>

          {filename && (
            <div className="status-pill">
              <span className="status-dot" />
              <span>Ready</span>
            </div>
          )}

        </header>

        {/* Empty state */}
        {!filename &&
          messages.length === 0 && (

            <div className="empty-state">

              <div className="empty-icon">
                ↑
              </div>

              <h2>
                Start with a document
              </h2>

              <p>
                Upload a PDF, Word document or
                text file and ask questions
                about its contents.
              </p>

              <label className="large-upload">

                <input
                  type="file"
                  accept=".pdf,.docx,.txt"
                  hidden
                  onChange={handleUpload}
                />

                <span>
                  {uploading
                    ? "Processing..."
                    : "Choose a document"}
                </span>

                <small>
                  PDF · DOCX · TXT
                </small>

              </label>

            </div>
          )}

        {/* Chat */}
        {filename && (
          <div className="chat-area">

            {/* Welcome */}
            {messages.length === 0 && (

              <div className="welcome">

                <span className="welcome-label">
                  READY WHEN YOU ARE
                </span>

                <h2>
                  What would you like
                  to know?
                </h2>

                <p>
                  Ask a question about{" "}
                  <strong>{filename}</strong>.
                </p>

                <div className="suggestions">

                  <button
                    onClick={() =>
                      setQuestion(
                        "Summarize this document."
                      )
                    }
                  >
                    Summarize this document
                  </button>

                  <button
                    onClick={() =>
                      setQuestion(
                        "What are the main ideas?"
                      )
                    }
                  >
                    What are the main ideas?
                  </button>

                  <button
                    onClick={() =>
                      setQuestion(
                        "What are the key conclusions?"
                      )
                    }
                  >
                    Key conclusions
                  </button>

                </div>

              </div>
            )}

            {/* Messages */}
            <div className="messages">

              {messages.map(
                (message, index) => (

                  <div
                    className={`message ${message.role}`}
                    key={index}
                  >

                    <div className="message-header">

                      <div className="message-label">
                        {message.role === "user"
                          ? "YOU"
                          : "READLY"}
                      </div>

                      <button
                        className="message-delete"
                        onClick={() =>
                          handleDeleteMessage(index)
                        }
                        title="Delete message"
                      >
                        ×
                      </button>

                    </div>

                    <div className="message-content">
                      {message.content}
                    </div>

                    {/* Sources */}
                    {message.role === "assistant" &&
                      message.sources?.length > 0 && (

                        <div className="sources">

                          <div className="sources-title">
                            Sources
                          </div>

                          {message.sources.map(
                            (source, sourceIndex) => (

                              <div
                                className="source-card"
                                key={sourceIndex}
                              >

                                <div className="source-icon">
                                  ↗
                                </div>

                                <div>

                                  <div className="source-name">
                                    {getSourceName(
                                      source.source
                                    )}
                                  </div>

                                  <div className="source-page">
                                    {source.page !== null &&
                                    source.page !== undefined
                                      ? `Page ${source.page}`
                                      : "Document source"}
                                  </div>

                                </div>

                              </div>
                            )
                          )}

                        </div>
                      )}

                  </div>
                )
              )}

              {/* Thinking indicator */}
              {asking && (

                <div className="message assistant">

                  <div className="message-header">

                    <div className="message-label">
                      READLY
                    </div>

                  </div>

                  <div className="thinking">
                    <span />
                    <span />
                    <span />
                  </div>

                </div>
              )}

            </div>

            {/* Composer */}
            <div className="composer">

              <textarea
                value={question}
                onChange={(event) =>
                  setQuestion(event.target.value)
                }
                onKeyDown={handleKeyDown}
                placeholder="Ask anything about this document..."
                rows={1}
                disabled={asking}
              />

              <button
                className="send-button"
                onClick={askQuestion}
                disabled={
                  asking ||
                  !question.trim()
                }
              >
                ↑
              </button>

            </div>

            <div className="composer-note">
              Answers are generated only from
              your uploaded document.
            </div>

          </div>
        )}

      </main>

    </div>
  );
}

export default App;