import os
import openai
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from typing import List

# === Load API Keys ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
openai.api_key = OPENAI_API_KEY

# === Pinecone Setup ===
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("spanishpoc")

# === Embedding Function ===
def embed_text(text: str) -> List[float]:
    response = openai.Embedding.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return response["data"][0]["embedding"]

# === Retrieval Function ===
def retrieve_top_k_chunks(query: str, k: int = 10):
    embedded_query = embed_text(query)
    search_result = index.query(
        vector=embedded_query,
        top_k=k,
        include_metadata=True
    )
    return search_result["matches"]

# === Streaming Answer Generation ===
def stream_answer(query: str, documents: List[dict], chunk_dir: str):
    context_parts = []
    for match in documents:
        meta = match.get("metadata", {})
        source_file = meta.get("source_file", "unknown")
        chunk_number = meta.get("chunk_number", "unknown")
        filename = f"{source_file}_chunk_{chunk_number}.txt"
        file_path = os.path.join(chunk_dir, filename)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                chunk_text = f.read().strip()
        except FileNotFoundError:
            chunk_text = "[Text not found on disk]"

        context_parts.append(f"{source_file} (Chunk {chunk_number}):\n{chunk_text}")

    context = "\n\n".join(context_parts) or "[No relevant document text found for this query.]"

    prompt = f"""You are a helpful assistant. Use the following retrieved documents to give me the list of Sanctions from this.

Context:
{context}

Question: {query}

Answer:"""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        stream=True
    )

    for chunk in response:
        if "choices" in chunk:
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                yield delta["content"]

# === Streamlit App ===
st.set_page_config(page_title="Secure RAG App", page_icon="🔐")

# --- Session State ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# --- Page 1: Access Code Gate ---
if not st.session_state.authenticated:
    st.title("🔐 Access Required")
    access_code = st.text_input("Enter Access Code:", type="password")
    if st.button("Submit"):
        if access_code == "2468":
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect code. Please try again.")
    st.stop()  # prevent rest of app from rendering

# --- Page 2: Main RAG Interface ---
st.title("🧠 Sanctions RAG Assistant")

query = st.text_input("Enter your question about Sanctions:")
submit = st.button("Get Answer")

if submit and query.strip():
    with st.spinner("🔍 Retrieving relevant documents..."):
        docs = retrieve_top_k_chunks(query)

    st.subheader("📄 Retrieved Chunks")
    for d in docs:
        meta = d.get("metadata", {})
        st.markdown(f"**{meta.get('source_file', 'unknown')} (Chunk {meta.get('chunk_number', '-')})**")

    st.subheader("🧠 GPT-4 Streaming Answer")
    placeholder = st.empty()
    full_response = ""

    for part in stream_answer(query, docs, chunk_dir="chunked_output"):
        full_response += part
        placeholder.markdown(full_response)

    with open("response.txt", "w", encoding="utf-8") as f:
        f.write(full_response)

# --- Optional: Logout Button ---
if st.button("Logout"):
    st.session_state.authenticated = False
    try:
     st.rerun()
    except AttributeError:
     st.experimental_rerun()

