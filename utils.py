import os
import dotenv
import logging
import streamlit as st
from langchain_community.document_loaders import GitLoader
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
dotenv.load_dotenv()

def load_and_parse_repo(repo_path, repo_url, branch="main", file_filter=None):
    """
    Load and parse a GitHub repository.
    """
    # Load the GitHub repository
    loader = GitLoader(
        repo_path=repo_path,
        clone_url=repo_url,
        file_filter=file_filter,
        branch=branch
    )
    
    try:
        documents = loader.load()
        
        # Create Document objects with metadata
        parsed_documents = []
        for doc in documents:
            parsed_documents.append(
                Document(
                    page_content=doc.page_content,
                    metadata={
                        "source": doc.metadata.get("source", ""),
                        "file_path": doc.metadata.get("file_path", ""),
                    }
                )
            )

        # Split documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        split_documents = text_splitter.split_documents(parsed_documents)
        
        return split_documents
        
    except Exception as e:
        print(f"Error processing repository: {e}")
        return []

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

def load_faiss_index_and_query(index_path, query, embedding_model_name):
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
        return vector_store.similarity_search(query)
    except Exception as e:
        st.error(f"Error in query processing: {str(e)}")
        return []
