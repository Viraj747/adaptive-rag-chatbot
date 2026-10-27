import os
import uuid
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(page_title="Adaptive RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Adaptive RAG Chatbot")
st.caption("Ask about uploaded documents, general topics, or current information.")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Upload a document")
    uploaded = st.file_uploader("Choose PDF or TXT", type=["pdf", "txt"])
    description = st.text_input("Document description", placeholder="e.g. Machine learning notes")
    if st.button("Upload and index", disabled=not (uploaded and description.strip())):
        try:
            response = requests.post(
                f"{API_URL}/rag/documents/upload",
                files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                data={"description": description},
                timeout=180,
            )
            response.raise_for_status()
            payload = response.json()
            st.success(f"Indexed {payload['chunks_indexed']} chunks from {payload['filename']}.")
        except requests.RequestException as exc:
            st.error(f"Upload failed: {exc}")
    st.divider()
    st.caption(f"Session: {st.session_state.session_id[:8]}…")
    if st.button("New conversation"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("route"):
            st.caption(f"Route: {message['route']}")
        if message.get("sources"):
            st.caption("Sources: " + " · ".join(message["sources"]))

question = st.chat_input("Ask a question…")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                response = requests.post(
                    f"{API_URL}/rag/query",
                    json={"query": question, "session_id": st.session_state.session_id},
                    timeout=180,
                )
                response.raise_for_status()
                result = response.json()["result"]
                content = result.get("content", "No response returned.")
                st.markdown(content)
                st.caption(f"Route: {result.get('route', 'unknown')}")
                if result.get("sources"):
                    st.caption("Sources: " + " · ".join(result["sources"]))
                st.session_state.messages.append({
                    "role": "assistant", "content": content,
                    "route": result.get("route"), "sources": result.get("sources", [])
                })
            except requests.RequestException as exc:
                content = f"Request failed: {exc}"
                st.error(content)
                st.session_state.messages.append({"role": "assistant", "content": content})
