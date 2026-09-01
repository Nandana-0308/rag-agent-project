

An AI-powered **Retrieval-Augmented Generation (RAG) Research Assistant** built with Python, Streamlit, Google Gemini, ChromaDB, and Sentence Transformers.

The application allows users to upload PDF or TXT documents, index their content, and ask questions through a chat interface. It retrieves relevant information from the uploaded documents and uses Gemini to generate an answer. If the available document context is not sufficient, the system can fall back to web search.

## ✨ Features

- 📄 Upload PDF and TXT documents
- ✂️ Split documents into smaller overlapping text chunks
- 🧠 Generate semantic embeddings using Sentence Transformers
- 🗄️ Store and retrieve embeddings using ChromaDB
- 🔎 Perform semantic document retrieval
- 🤖 Generate answers using Google Gemini
- 🌐 Use web search when the uploaded documents do not provide enough information
- 💬 Streamlit-based conversational interface
- 🔁 Retry/fallback handling for unavailable or overloaded Gemini models
- 🧭 Agentic workflow that decides when document retrieval or web search is needed
- 📌 Display a trace showing how the answer was obtained

## 🧠 How It Works

The project follows an agentic RAG workflow:

```text
User Question
      │
      ▼
   Router
      │
      ├── Simple question ──────────────► Gemini
      │
      └── Requires information
                    │
                    ▼
            Document Retrieval
                    │
                    ▼
              Context Grader
                    │
              ┌─────┴─────┐
              │           │
          Sufficient   Insufficient
              │           │
              ▼           ▼
           Gemini      Web Search
                          │
                          ▼
                       Gemini
                          │
                          ▼
                    Final Answer
