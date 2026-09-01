import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import errors
from retriever import retrieve

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# gemini-2.5-flash is restricted to older accounts and isn't available to
# new API keys, so we use two models from the currently available lineup —
# each has its own separate daily quota.
PRIMARY_MODEL = "gemini-3.6-flash"
FALLBACK_MODEL = "gemini-3.5-flash-lite"


def call_gemini_with_retry(prompt, max_retries=3, wait_seconds=5):
    """Tries the primary model a few times, then falls back to a different
    model if it's overloaded, out of quota, or unavailable."""
    for model_name in [PRIMARY_MODEL, FALLBACK_MODEL]:
        for attempt in range(1, max_retries + 1):
            try:
                response = client.models.generate_content(model=model_name, contents=prompt)
                return response.text
            except errors.ClientError as e:
                # 404 = model doesn't exist for this account -> no point retrying, skip to next model
                if "NOT_FOUND" in str(e):
                    print(f"[{model_name}] not available for this account, trying next model...")
                    break
                # 429 = quota used up -> retrying won't help until quota resets, skip to next model
                elif "RESOURCE_EXHAUSTED" in str(e):
                    print(f"[{model_name}] quota exceeded, trying next model...")
                    break
                else:
                    print(f"[{model_name}] attempt {attempt}/{max_retries} failed: {e}")
                    if attempt < max_retries:
                        time.sleep(wait_seconds)
            except errors.ServerError as e:
                # 503 = temporarily overloaded -> worth retrying the same model
                print(f"[{model_name}] attempt {attempt}/{max_retries} failed: {e}")
                if attempt < max_retries:
                    time.sleep(wait_seconds)
    raise RuntimeError("All Gemini models are currently unavailable or out of quota. Please try again later.")


# ---- Agent 1: Router — does this even need document lookup? ----
def route_question(question):
    prompt = f"""Question: "{question}"
Reply with exactly one word: "yes" if answering this needs looking up specific
facts/documents, or "no" if it's a simple question you could answer directly.
Answer with only: yes or no."""
    result = call_gemini_with_retry(prompt).strip().lower()
    return result.startswith("yes")


# ---- Agent 2: Grader — is this context actually enough to answer fully? ----
def grade_context(question, context):
    if not context.strip():
        return False
    prompt = f"""Question: {question}

Context:
{context}

Does this context FULLY and CONFIDENTLY answer the question, with no
important part missing? Reply with only "yes" or "no"."""
    result = call_gemini_with_retry(prompt).strip().lower()
    return result.startswith("yes")


# ---- Helper: rewrite the query for a retry ----
def reformulate_query(question, missing_reason="missing some details"):
    prompt = f"""Original question: {question}
The search so far did not fully answer it ({missing_reason}).
Rewrite this as a more specific search query that would find the missing piece.
Reply with ONLY the rewritten query, nothing else."""
    return call_gemini_with_retry(prompt).strip()


# ---- Agent 3: Web search fallback — free, no API key ----
def web_search(query, max_results=3):
    from duckduckgo_search import DDGS
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(f"{r['title']}: {r['body']} (source: {r['href']})")
    except Exception:
        return ""
    return "\n\n".join(results)


# ---- Agent 4: Generator — writes the final answer with clear source labels ----
def generate_final_answer(question, local_context, web_context):
    prompt = f"""Answer the question using the context below.

RULES FOR YOUR ANSWER:
- If a fact comes from "UPLOADED DOCUMENTS", say it normally and cite like [1], [2].
- If a fact comes from "WEB SEARCH", clearly say: "This wasn't in your uploaded
  documents — from the web (source: <link>): ..."
- If something truly cannot be found in EITHER source, say so honestly instead
  of guessing.

UPLOADED DOCUMENTS:
{local_context if local_context else "(nothing relevant found here)"}

WEB SEARCH:
{web_context if web_context else "(no web search was needed or nothing found)"}

Question: {question}

Answer:"""
    return call_gemini_with_retry(prompt)


# ---- The full agentic loop ----
def answer_question_agentic(question, top_k=4, max_retries=1):
    trace = []

    needs_lookup = route_question(question)
    trace.append(f"Router: needs lookup = {needs_lookup}")

    local_context = ""
    local_sufficient = False
    if needs_lookup:
        current_query = question
        for attempt in range(max_retries + 1):  # +1 so max_retries=1 means "try, then retry once"
            results = retrieve(current_query, top_k=top_k)
            local_context = "\n\n".join(f"[{i+1}] (source: {src}) {chunk}"
                                         for i, (chunk, src) in enumerate(results))
            trace.append(f"Local retrieval attempt {attempt + 1}: {len(results)} chunks")
            local_sufficient = grade_context(question, local_context)
            if local_sufficient:
                trace.append("Grader: local context is sufficient")
                break
            if attempt < max_retries:
                current_query = reformulate_query(question)
                trace.append(f"Grader: insufficient -> retry query '{current_query}'")

    # Reuse the grading result we already have instead of calling the grader again
    web_context = ""
    if not local_sufficient:
        trace.append("Local context insufficient -> trying web search")
        current_query = question
        for attempt in range(max_retries + 1):
            web_context = web_search(current_query)
            trace.append(f"Web search attempt {attempt + 1}")
            if grade_context(question, web_context):
                trace.append("Grader: web context is sufficient")
                break
            if attempt < max_retries:
                current_query = reformulate_query(question, missing_reason="web results too vague")

    answer = generate_final_answer(question, local_context, web_context)
    return answer, trace


if __name__ == "__main__":
    query = input("Ask a question about your topic: ")
    answer, trace = answer_question_agentic(query)
    print("\n" + answer)
    print("\n--- How I found this ---")
    for step in trace:
        print("- " + step)