import os
import dotenv
import requests
import streamlit as st
from langchain_core.documents.base import Document
from langchain_core.vectorstores.base import VectorStore
from langchain_community.embeddings.huggingface import (
    HuggingFaceInferenceAPIEmbeddings
)
from langchain_community.vectorstores.faiss import FAISS
import numpy as np
from typing import List

dotenv.load_dotenv()

from typing import List
from langchain.embeddings.base import Embeddings

class PrecomputedEmbeddings(Embeddings):
    def __init__(self, document_embeddings: List[List[float]]):
        self.document_embeddings = document_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return precomputed embeddings for documents."""
        return self.document_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Handle query embeddings. Adjust as needed."""
        raise NotImplementedError("Query embedding requires a different method or precomputed values.")

def get_jina_embeddings(documents: List[str]):
    if not documents:
        raise ValueError("documents are empty!")
    data = {
        "model": "jina-embeddings-v2-base-code",
        "input": documents
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv("JINA_API_KEY")}"
    }
    response = requests.post("https://api.jina.ai/v1/embeddings", json=data, headers=headers)
    print(response.text)


def load_embeddings_from_np_file(embeddings_path: str='embeddings/embeddings.npz'):
    """
    Load the document embeddings from the given path
    """
    try:
        # Load the embeddings
        embeddings = np.load(embeddings_path)
        return np.array(embeddings)
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return None

def get_query_embedding(query, hf_embeddings: HuggingFaceInferenceAPIEmbeddings):
    """
    Get the query embedding
    """
    try:
        # Generate the query embedding
        query_embedding = hf_embeddings.embed_query(query)
        print(f"Query embedding: {len(query_embedding)}")
        if isinstance(query_embedding, dict):
            query_embedding = query_embedding.get('vector', None)
        if query_embedding is None:
            print("Failed to generate query embedding")
            return None
        return query_embedding
    except Exception as e:
        print(f"Error initializing embedding model: {e}")
        return None

def get_vector_store(
        documents: List[Document], 
        hf_embeddings: HuggingFaceInferenceAPIEmbeddings
        ) -> FAISS:
    if not documents:
        print("No documents provided for vector store creation")
        return None
    if not all(isinstance(doc, Document) for doc in documents):
        print("Invalid document format. Ensure all items are LangChain Document objects.")
        return None

    try:
        print(f"Number of documents: {len(documents)}")
        # Create vector store
        vector_store = FAISS.from_documents(
            documents=documents,
            embedding=hf_embeddings
        )
        assert isinstance(vector_store, VectorStore)
        return vector_store
    except Exception as e:
        # Log detailed error information
        print(f"Error in get_vector_store: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"API Response: {e.response.text}")
        return None

def save_faiss_vector_store(vector_store: FAISS, folder_path: str="vector_store"):
    vector_store.save_local(folder_path=folder_path)

def load_faiss_vector_store(index_path, hf_embeddings:HuggingFaceInferenceAPIEmbeddings) -> FAISS:
    """
    Load FAISS index from disk
    Args:
        index_path (str): Path to FAISS index
        query (str): Query string
        embedding_model_name (str): Name of the embedding model
    Returns:
        List: Query results
    """
    try:
        vector_store = FAISS.load_local(
            folder_path=index_path, 
            embeddings=hf_embeddings,
            allow_dangerous_deserialization=True
        )
        return vector_store
    except Exception as e:
        st.error(f"Error in query processing: {str(e)}")
        return None

def similarity_search_on(vector_store: FAISS, query: str, k: int = 5):
    """
    Perform similarity search using FAISS
    
    Args:
        vector_store (FAISS): FAISS index
        query (str): Query string
        k (int): Number of results to return
    
    Returns:
        documents: List of documents
    """
    if vector_store is None:
        print("Vector store not found")
        return None
    
    if not query or not query.strip():
        print("Query not found")
        return None
    
    try:
        documents = vector_store.similarity_search(
            query=query,
            k=k
        )
        return documents
    except Exception as e:
        print(f"Error in similarity_search: {e.args}")
        return None

if __name__ == '__main__':
    from repository_handler import load_repository_documents

    documents = load_repository_documents(
        "https://github.com/harshsingh-io/Java.DSA.git", 
        branch="master"
        )
    print(f"Number of documents: {len(documents)}")
    import os
    import dotenv
    dotenv.load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token is None:
        print("Hugging Face API token not found")
        exit(1)
    hf_embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=hf_token,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    if hf_embeddings is None:
        print("Failed to initialize Hugging Face embeddings")
        exit(1)

    embeddings = generate_and_save_document_embeddings(documents, hf_embeddings)
    print(f"Number of embeddings: {len(embeddings)}")

    vector_store = create_and_save_faiss_vectorstore(
        hf_embeddings=hf_embeddings,
        documents=documents,
        index_path="vector_store"
        )
    print("Index created and saved")

    _vector_store = load_faiss_vector_store("vector_store", hf_embeddings)
    if _vector_store is None:
        print("Failed to load FAISS index")
        exit(1)
    print("Index loaded successfully")
    results = vector_store.similarity_search("binary search", 5)
    print(f"Number of results: {len(results)}")
    for result in results:
        print(result)
    # results = similarity_search_on(vector_store, "binary search", 5)
    # print(f"Number of results: {len(results)}")
    # for result in results:
    #     print(result)

