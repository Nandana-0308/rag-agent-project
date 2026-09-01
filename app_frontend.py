import streamlit as st
import tempfile
import os
from ingest import add_document
from agent import answer_question_agentic

st.set_page_config(page_title="Research Assistant", page_icon="🔍")
st.title("🔍 Research Assistant")

# ---- Sidebar: document upload ----
st.sidebar.header("Upload a document")
uploaded_file = st.sidebar.file_uploader("Choose a PDF or .txt file", type=["pdf", "txt"])

if uploaded_file is not None:
    if st.sidebar.button("Process document"):
        with st.spinner("Reading and indexing your document... this can take a moment for large PDFs"):
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            num_chunks = add_document(tmp_path, uploaded_file.name)
            os.remove(tmp_path)
        st.sidebar.success(f"Indexed '{uploaded_file.name}' — {num_chunks} chunks stored")

st.sidebar.divider()
st.sidebar.caption("Upload a document above, then ask questions about it in the chat. "
                    "If it's not fully covered in your document, I'll search the web too.")

# ---- Chat history lives in session_state so it persists while you chat ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# Redraw all previous messages every time the page reruns (Streamlit's normal pattern)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "trace" in message:
            with st.expander("How I found this answer"):
                for step in message["trace"]:
                    st.write("- " + step)

# ---- Chat input box, pinned to the bottom of the page ----
user_question = st.chat_input("Ask a question about your document...")

if user_question:
    # Show and store the user's message immediately
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Generate and show the assistant's response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, trace = answer_question_agentic(user_question)
            st.markdown(answer)
            with st.expander("How I found this answer"):
                for step in trace:
                    st.write("- " + step)

    st.session_state.messages.append({"role": "assistant", "content": answer, "trace": trace})