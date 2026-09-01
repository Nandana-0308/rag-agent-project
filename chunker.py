def chunk_text(text, chunk_size=300, overlap=50):
    """
    Splits text into overlapping chunks.
    chunk_size: max words per chunk
    overlap: words repeated between consecutive chunks (keeps context continuous)
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap   # move forward, but overlap a bit
    return chunks


# ---- Test block: only runs when you execute "python chunker.py" directly ----

if __name__ == "__main__":
    import os

    # Pick the first .txt file in your data folder automatically
    data_dir = "data"
    files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]

    if not files:
        print("No .txt files found in the data folder!")
    else:
        test_file = files[0]
        print(f"Testing chunker on: {test_file}\n")

        with open(os.path.join(data_dir, test_file), "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text, chunk_size=100, overlap=20)
        print(f"Total chunks: {len(chunks)}")

        for i, chunk in enumerate(chunks):
            print(f"\n--- Chunk {i+1} ---")
            print(chunk)