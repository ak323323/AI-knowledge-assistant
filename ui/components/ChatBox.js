"use client";
import { useState, useRef, useEffect } from "react";
import axios from "axios";
import toast from "react-hot-toast"
import ReactMarkdown from "react-markdown";

export default function ChatBox() {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadInfo, setUploadInfo] = useState(null);
  const [documents, setDocuments] = useState([]);

const ask = async () => {
    if (!query.trim()) {
      toast.error("Please enter a question");
      return;
    }

    const userMessage = { role: "user", content: query };
    setMessages((prev) => [...prev, userMessage]);
    setQuery("");
    setLoading(true);

    try {
      const res = await axios.post("http://localhost:3001/ask", {
        question: query,
      });

      const aiMessage = {
        role: "assistant",
        content: res.data.answer,
        sources: res.data.sources || res.data.context || [],
      };

      setMessages((prev) => [...prev, aiMessage]);
    } catch (err) {
      console.error(err);
      toast.error("Failed to fetch answer");
    }

    setLoading(false);
};
  
  const uploadFile = async () => {
    if (!file) {
      toast.error("Please select a file");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setUploading(true);

    try {
      const res = await axios.post("http://localhost:8000/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setUploadInfo(res.data); //  store response
      toast.success("File uploaded & indexed!");
      setFile(null);
    } catch (err) {
      console.error(err);
      toast.error("Upload failed");
    }
    loadDocuments();

    setUploading(false);
};

  const download = async (format, msg) => {

  try {

    // =====================================================
    // SEND EXPORT REQUEST
    // =====================================================

    const res = await fetch(
      "http://127.0.0.1:8000/export",
      {
        method: "POST",

        headers: {
          "Content-Type": "application/json"
        },

        body: JSON.stringify({

          answer: msg.content,

          sources: msg.sources || [],

          format: format
        })
      }
    );

    // =====================================================
    // CONVERT RESPONSE TO BLOB
    // =====================================================

    const blob = await res.blob();

    const url = window.URL.createObjectURL(blob);

    // =====================================================
    // DETERMINE FILE EXTENSION
    // =====================================================

    let filename = "answer";

    if (format === "pdf") {

      filename += ".pdf";

    } else if (format === "docx") {

      filename += ".docx";

    } else if (format === "md") {

      filename += ".md";

    } else if (format === "xlsx") {

      filename += ".xlsx";

    } else if (format === "csv") {

      filename += ".csv";

    } else {

      filename += ".txt";
    }

    // =====================================================
    // CREATE DOWNLOAD LINK
    // =====================================================

    const a = document.createElement("a");

    a.href = url;

    a.download = filename;

    document.body.appendChild(a);

    a.click();

    // =====================================================
    // CLEANUP
    // =====================================================

    a.remove();

    window.URL.revokeObjectURL(url);

  } catch (err) {

    console.error(
      "[DOWNLOAD ERROR]",
      err
    );
  }
};

  const loadDocuments = async () => {
    try {

      const controller = new AbortController();

      const timeout = setTimeout(() => {
        controller.abort();
      }, 5000);

      const res = await fetch(
        "http://127.0.0.1:8000/documents",
        {
          signal: controller.signal,
        }
      );

      clearTimeout(timeout);

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }

      const data = await res.json();

      console.log("API RESPONSE:", data);

      setDocuments(Array.isArray(data) ? data : []);

    } catch (err) {
      console.error("LOAD DOCS ERROR:", err);
    }
  };


const deleteDoc = async (docId) => {
  try {

    const res = await fetch(
      `http://localhost:8000/documents/${docId}`,
      {
        method: "DELETE",
      }
    );

    const data = await res.json();

    console.log("DELETE RESPONSE:", data);

    loadDocuments();

  } catch (err) {
    console.error("Delete failed", err);
  }
};

  const bottomRef = useRef();
  useEffect(() => {
  bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
  loadDocuments();
}, []);

console.log("IS ARRAY:", Array.isArray(documents));
console.log("DOCUMENTS:", documents);

    return (
    <div className="flex h-screen bg-[#0f1117] text-gray-200">

      {/* Sidebar */}
      <div className="w-64 bg-[#161a23] border-r border-gray-800 p-4 hidden md:block">

        <h3 className="text-sm text-gray-400 mb-2">📂 Indexed Files</h3>

        <div className="space-y-2 text-sm">
          {Array.isArray(documents) && documents.map((doc) => (
            <div
              key={doc.doc_id}
              className="flex justify-between items-center bg-[#111827] p-2 rounded-lg border border-gray-700"
            >
              <span className="truncate">
                {doc.source.split("\\").pop()}
              </span>

              <button
                onClick={() => deleteDoc(doc.doc_id)}
                className="text-red-400 hover:text-red-500 text-xs"
              >
                ❌
              </button>
            </div>
          ))}
        </div>

      </div>

      {/* Main Chat */}
      <div className="flex flex-col flex-1">

        {/* Header */}
        <div className="p-4 border-b border-gray-800 text-lg font-semibold">
          Knowledge Assistant
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.map((msg, index) => (
            <div
              key={index}
              className={`max-w-2xl px-4 py-3 rounded-xl ${
                msg.role === "user"
                  ? "ml-auto bg-[#2a2f3a]"
                  : "mr-auto bg-[#1b2130]"
              }`}
            >
              <ReactMarkdown
                components={{
                  p: ({ children }) => <p className="mb-2">{children}</p>,
                  h1: ({ children }) => <h1 className="text-xl font-bold">{children}</h1>,
                  code: ({ children }) => (
                    <code className="bg-gray-800 px-1 rounded">{children}</code>
                  ),
                }}
              >
                {msg.content}
              </ReactMarkdown>

              {msg.role === "assistant" && (
            <div className="flex gap-2 mt-3">
              <button
                onClick={() => download("pdf", msg)}
                className="px-2 py-1 bg-blue-600 rounded-lg text-xs"
              >
                ⬇ PDF
              </button>

              <button
                onClick={() => download("docx", msg)}
                className="px-2 py-1 bg-green-600 rounded-lg text-xs"
              >
                ⬇ Word
              </button>
              <button
                onClick={() => download("md", msg)}
                className="px-2 py-1 bg-gray-700 rounded-lg text-xs"
              >
                ⬇ MD
              </button>
              <button
                onClick={() => download("xlsx", msg)}
                className="px-2 py-1 bg-emerald-600 rounded-lg text-xs"
              >
                ⬇ Excel
              </button>
              <button
                onClick={() => download("csv", msg)}
                className="px-2 py-1 bg-yellow-600 rounded-lg text-xs"
              >
                ⬇ CSV
              </button>
            </div>
            )}

              {/* Context for RAG */}
              {msg.sources && msg.sources.length > 0 && (
              <details className="mt-3 text-sm text-gray-400">
                <summary className="cursor-pointer">📚 Sources</summary>

                <div className="mt-2 space-y-2">
                  {msg.sources.map((src, i) => (
                    <div
                      key={i}
                      className="p-2 rounded-lg bg-[#111827] border border-gray-700"
                    >
                      {/* 🔹 Metadata row */}
                      <div className="flex justify-between text-xs text-gray-400 mb-1">
                        <span>
                          📄 {src.source?.split("\\").pop() || "Unknown file"}
                        </span>

                        <span>Chunk #{src.chunk_id ?? "-"}</span>

                        {/*  SAFE score rendering */}
                        <span>
                          Score: {((src.score ?? 0) * 100).toFixed(1)}%
                        </span>
                      </div>

                      {/* 🔹 Preview text */}
                      <p className="text-gray-300 text-sm">
                        {src.content.slice(0, 200)}...
                      </p>
                    </div>
                  ))}
                </div>
              </details>
            )}
            {msg.sources?.[0] && (
            <div className="text-xs text-green-400 mt-2">
              ⭐ Top source: {msg.sources[0].source?.split("\\").pop()}
            </div>
            )}
            </div>
          ))}

          {loading && (
            <div className="text-gray-400 text-sm">AI is thinking...</div>
          )}
          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div className="p-4 border-t border-gray-800 bg-[#0f1117]">
          <div className="flex gap-3 max-w-2xl mx-auto">

            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask anything..."
              className="flex-1 p-3 rounded-xl bg-[#161a23] border border-gray-700 focus:outline-none focus:ring-2 focus:ring-purple-500"
            />

            <button
              onClick={ask}
              className="px-5 py-2 rounded-xl bg-purple-600 hover:bg-purple-700"
            >
              {loading ? "..." : "Send"}
            </button>
          </div>

          {/* Upload inside input area (cleaner UX) */}
          <div className="flex gap-3 max-w-2xl mx-auto mt-3">
            <input
              type="file"
              accept=".pdf,.docx,.md"
              onChange={(e) => setFile(e.target.files[0])}
              className="text-sm"
            />
            <button
              onClick={uploadFile}
              className="px-3 py-1 bg-green-600 rounded-lg"
            >
              {uploading ? "Uploading..." : "Upload"}
            </button>  
          </div>
        </div>
        {uploadInfo && (
            <div className="max-w-2xl mx-auto mt-4 p-4 rounded-xl bg-[#161a23] border border-gray-700">

              <h3 className="text-purple-400 font-semibold mb-2">
                📂 {uploadInfo.file_name}
              </h3>

              <div className="text-sm text-gray-300 space-y-1">
                <p>📄 Chunks created: {uploadInfo.chunks_created ?? 0}</p>
                <p>🧠 Vectors added: {uploadInfo.vectors_added ?? 0}</p>
                <p>📦 Total index size: {uploadInfo.total_vectors ?? 0}</p>
              </div>

              {/* Preview */}
              {uploadInfo.preview?.length > 0 && (
                <details className="mt-3 text-sm text-gray-400">
                  <summary className="cursor-pointer">🔍 Preview chunks</summary>
                  <ul className="list-disc ml-5 mt-2 space-y-1">
                    {uploadInfo.preview.map((chunk, i) => (
                      <li key={i}>{chunk.slice(0, 150)}...</li>
                    ))}
                  </ul>
                </details>
              )}

            </div>
          )}
      </div>
    </div>
  );
}