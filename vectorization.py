import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores.faiss import FAISS

def generate_and_store_embeddings(processed_repo, embedding_model):
    """
    Get the processed documents of the repository, 
    create embeddings and return the vector store.
    """

    hf_embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=os.getenv("HF_TOKEN"),
        model_name=embedding_model
    )
    vector_store=FAISS.from_documents(processed_repo, embedding=hf_embeddings)
    return vector_store

def load_faiss_vector_store(index_path, embedding_model_name):
    """
    Load FAISS index and perform query
    Args:
        index_path (str): Path to FAISS index
        query (str): Query string
        embedding_model_name (str): Name of the embedding model
    Returns:
        List: Query results
    """
    try:
        embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=os.getenv("HF_TOKEN"),
            model_name=embedding_model_name
        )
        vector_store = FAISS.load_local(index_path, embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error in query processing: {str(e)}")
        return []
