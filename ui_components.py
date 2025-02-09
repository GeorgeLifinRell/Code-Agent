import json
import os
import streamlit as st
from constants import REPOSITORY_PATH
from anytree.exporter import DictExporter
from repository_handler import build_file_tree
from repository_handler import (
    fetch_repository,
    split_repository_documents,
    parse_and_print_repository
)
from vectorization_handler import (
    generate_and_store_embeddings,
    load_faiss_vector_store
)
from llm_handler import (
    get_llm,
    query_llm
)
from constants import (
    EMBEDDING_MODEL_NAME,
    VECTOR_STORE_DIR,
    AVAILABLE_LLMS
)
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores.faiss import FAISS

def sidebar():
    st.header("Written in sidebar")
    st.selectbox(
        label="Select the embedding model:",
        options=["sentence-transformers/all-MiniLM-L6-v2"]
    )

def print_file_tree(repo_path=REPOSITORY_PATH):
    st.write(f"Repository path: {repo_path}")
    file_tree = build_file_tree(repo_path)
    exporter = DictExporter()
    st.json(json.dumps(exporter.export(file_tree), indent=2))

def similarity_search_tab():
    repo_url = st.text_input("Enter the repository URL")
    if repo_url:
        st.write(f"Repository URL: {repo_url}")
        st.button("Get Repository Tree", on_click=print_file_tree())
        repo_documents = fetch_repository("./repository", repo_url)
        parsed_repo = split_repository_documents(repo_documents)
        if parsed_repo:
            st.write(f"Parsed repository contains {len(parsed_repo)} documents.")
            st.write(f"First document: {parsed_repo[0]}")

            # Generate embeddings and create FAISS index
            st.write(f"Generating embeddings using {EMBEDDING_MODEL_NAME}.")
            vector_store = generate_and_store_embeddings(parsed_repo, EMBEDDING_MODEL_NAME)
            vector_store.save_local(VECTOR_STORE_DIR)

            query = st.text_input("Enter your query")
            if query:
                st.write(f"Query: {query}")
                st.write("Processing query...")

                embeddings = HuggingFaceInferenceAPIEmbeddings(
                    api_key=os.getenv("HF_TOKEN"),
                    model_name=EMBEDDING_MODEL_NAME
                )
                if not embeddings:
                    st.error("Error generating embeddings...")
                else:
                    vector_store = FAISS.load_local(VECTOR_STORE_DIR, embeddings, allow_dangerous_deserialization=True)
                similar_documents = vector_store.similarity_search(query=query)
                for i, document in enumerate(similar_documents, 1):
                    st.write(f"Results by relevance : {i} {document}")
        else:
            st.error("Failed to parse the repository or repository is empty.")


def llm_inference_tab():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    selected_llm = st.selectbox(
        label="💬 Select an LLM to chat with.",
        options=AVAILABLE_LLMS
    )
    llm = get_llm(selected_llm)
    if llm:
        user_input = st.text_input("You:", "", key="user_input")
        if st.button("Send") and user_input:
            response_text = query_llm(llm, user_input)
            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("LLM", response_text))
    for speaker, text in st.session_state.chat_history:
        st.markdown(f"**{speaker}:** {text}")