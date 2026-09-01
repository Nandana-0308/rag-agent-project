import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import errors
from retriever import retrieve

# Load variables from a .env file in the same folder (e.g. GEMINI_API_KEY=...)
load_dotenv()

# Configure Gemini with your free API key (from aistudio.google.com)
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Try the newest model first; fall back to an older, less busy one if it's overloaded
PRIMARY_MODEL = "gemini-3.6-flash"
FALLBACK_MODEL = "gemini-2.5-flash"


def call_gemini_with_retry(prompt, max_retries=3, wait_seconds=5):
    """
    Calls Gemini with automatic retry on server overload (503) errors.
    Tries PRIMARY_MODEL first; after repeated failures, switches to FALLBACK_MODEL.
    """
    models_to_try = [PRIMARY_MODEL, FALLBACK_MODEL]

    for model_name in models_to_try:
        for attempt in range(1, max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                return response.text
            except errors.ServerError as e:
                print(f"[{model_name}] attempt {attempt}/{max_retries} failed: {e}")
                if attempt < max_retries:
                    time.sleep(wait_seconds)  # brief pause before retrying
        print(f"Giving up on {model_name}, trying next model if available...\n")

    # If every model + every retry failed, fail clearly instead of silently
    raise RuntimeError("All Gemini models are currently unavailable. Please try again in a few minutes.")


def answer_question(query, top_k=4):
    results = retrieve(query, top_k=top_k)

    # Build context block with numbered source labels
    context = ""
    for i, (chunk, source) in enumerate(results):
        context += f"[{i+1}] (source: {source})\n{chunk}\n\n"

    prompt = f"""Answer the question using ONLY the context below.
Cite sources using [1], [2] etc. matching the context numbers.
If the context doesn't contain the answer, say so honestly.

Context:
{context}

Question: {query}

Answer:"""

    return call_gemini_with_retry(prompt)


if __name__ == "__main__":
    query = input("Ask a question about your topic: ")
    print("\n" + answer_question(query))