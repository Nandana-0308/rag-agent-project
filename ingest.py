import chromadb
from sentence_transformers import SentenceTransformer
from chunker import chunk_text
from pypdf import PdfReader

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="articles")


def extract_text(file_path, filename):
    """Pull plain text out of a PDF, or read a .txt file directly."""

    if filename.lower().endswith(".pdf"):
        reader = PdfReader(file_path)

        text = "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

        return text

    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()


def add_document(file_path, filename, chunk_size=300, overlap=50):
    """Extract, chunk, embed, and store an uploaded file."""

    # 1. Extract text
    text = extract_text(file_path, filename)

    if not text or not text.strip():
        raise ValueError(
            f"No text could be extracted from '{filename}'. "
            "If this is a scanned/image-only PDF, OCR is required."
        )

    print(f"Extracted characters: {len(text)}")

    # 2. Create chunks
    chunks = chunk_text(
        text,
        chunk_size=chunk_size,
        overlap=overlap
    )

    if not chunks:
        raise ValueError(
            f"No chunks were created from '{filename}'. "
            "Check your chunk_text() function."
        )

    # Remove empty chunks just in case
    chunks = [
        chunk.strip()
        for chunk in chunks
        if chunk and chunk.strip()
    ]

    if not chunks:
        raise ValueError(
            f"All generated chunks are empty for '{filename}'."
        )

    print(f"Created chunks: {len(chunks)}")

    # 3. Generate embeddings
    embeddings = model.encode(
        chunks,
        show_progress_bar=False
    ).tolist()

    if not embeddings:
        raise ValueError(
            f"No embeddings were generated for '{filename}'."
        )

    if len(embeddings) != len(chunks):
        raise ValueError(
            f"Embedding/chunk mismatch: "
            f"{len(chunks)} chunks, {len(embeddings)} embeddings."
        )

    print(f"Created embeddings: {len(embeddings)}")

    # 4. Create IDs
    start_id = collection.count()

    ids = [
        f"chunk_{start_id + i}"
        for i in range(len(chunks))
    ]

    # 5. Metadata
    metadatas = [
        {"source": filename}
        for _ in chunks
    ]

    # 6. Store in ChromaDB
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas
    )

    return len(chunks)


def clear_all_documents():
    """Delete all stored chunks and recreate an empty collection."""

    global collection

    client.delete_collection(name="articles")

    collection = client.get_or_create_collection(
        name="articles"
    )