import dotenv
import os
import shutil
import streamlit as st
from utils import (
    load_and_parse_repo,
    generate_and_store_embeddings
)
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores.faiss import FAISS

dotenv.load_dotenv()
if os.path.exists("./repository"):
    shutil.rmtree("./repository")
os.makedirs("./repository")

st.title("Repository Query Interface")

repo_url = st.text_input("Enter the repository URL")
EMBEDDING_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
VECTOR_STORE_DIR="./vector_store"

if repo_url:
    st.write(f"Repository URL: {repo_url}")
    parsed_repo = load_and_parse_repo("./repository", repo_url)
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
            result = vector_store.similarity_search(query=query)
            st.write(f"Query result: {result}")
    else:
        st.error("Failed to parse the repository or repository is empty.")