import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="articles")


def retrieve(query, top_k=4):
    query_embedding = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    return list(zip(chunks, sources))


if __name__ == "__main__":
    query = input("Ask a question about your topic: ")
    results = retrieve(query)
    for chunk, source in results:
        print(f"\n--- from {source} ---")
        print(chunk[:300], "...")