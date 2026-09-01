import os
import chromadb
from sentence_transformers import SentenceTransformer
from chunker import chunk_text

# Load embedding model (runs locally, downloads once on first run)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Set up ChromaDB (persists to disk in ./chroma_db)
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="articles")

# Process each .txt file in the data folder
data_dir = "data"
chunk_id = 0

for filename in os.listdir(data_dir):
    if not filename.endswith(".txt"):
        continue  # skip anything that isn't a .txt file

    filepath = os.path.join(data_dir, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text, chunk_size=300, overlap=50)

    for chunk in chunks:
        embedding = model.encode(chunk).tolist()
        collection.add(
            ids=[f"chunk_{chunk_id}"],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"source": filename}]  # remember which file this came from
        )
        chunk_id += 1

print(f"Indexed {chunk_id} chunks from {len([f for f in os.listdir(data_dir) if f.endswith('.txt')])} articles.")