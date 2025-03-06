import os
import dotenv
import streamlit as st
from langchain_core.documents.base import Document
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.vectorstores.base import VectorStore
import faiss
import numpy as np
from typing import List

def generate_and_save_document_embeddings(
        processed_documents: List[Document],
        hf_embeddings: HuggingFaceInferenceAPIEmbeddings
    ):
    """
    Get the processed documents of the repository, 
    create and return the embeddings.
    """
    try:
        # Extract text content from documents
        texts = [doc.page_content for doc in processed_documents]
        
        if not texts:
            print("No texts found in processed repository")
            return None
            
        # Generate embeddings
        try:    
            # Generate embeddings for all texts
            embeddings = []
            for text in texts:
                embedding = hf_embeddings.embed_documents(text)
                if isinstance(embedding, dict):
                    embedding = embedding.get('vector', None)
                if embedding is not None:
                    embeddings.append(embedding)
                    
            if not embeddings:
                print("No valid embeddings generated")
                return None
            np.savez_compressed("./embeddings/embeddings.npz", embeddings)
            return embeddings
        except FileNotFoundError as e:
            print(f"Error finding directory: {e}")
            return None
        except Exception as e:
            print(f"Error during embedding generation: {e}")
            return None
            
    except Exception as e:
        print(f"Error in generate_and_store_embeddings: {e}")
        return None

def load_document_embeddings(embeddings_path: str='embeddings/embeddings.npz'):
    """
    Load the document embeddings from the given path
    """
    try:
        # Load the embeddings
        embeddings = np.load(embeddings_path)
        return embeddings
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

def create_and_save_faiss_index(embeddings, index_path, hf_embeddings: HuggingFaceInferenceAPIEmbeddings):
    """
    Create a FAISS index from the embeddings and save it to disk
    """
    try:
        # Get dimension from first embedding
        dimension = len(embeddings[0])
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Create FAISS index
        index = faiss.IndexFlatL2(dimension)
        
        # Add vectors to the index
        index.add(embeddings_array)
        
        # Create IDs for the embeddings
        ids = [str(i) for i in range(len(embeddings))]
        
        # Create mapping from IDs to document store
        index_to_docstore_id = {i: id for i, id in enumerate(ids)}
        
        # Initialize FAISS vector store
        vector_store = FAISS(
            embedding_function=hf_embeddings,
            index=index,
            docstore=InMemoryDocstore({}),  # Start with empty docstore
            index_to_docstore_id=index_to_docstore_id
        )
        
        # Save the index to disk
        vector_store.save_local(index_path)
        return True
    except Exception as e:
        print(f"Error in create_and_save_faiss_index: {e}")
        return False

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
    
def create_and_save_faiss_vectorstore(
        documents: List[Document], 
        hf_embeddings: HuggingFaceInferenceAPIEmbeddings,
        index_path: str
        ) -> FAISS:
    """
    Load FAISS index from disk
    Args:
        index_path (str): Path to FAISS index
        documents (List[Document]): List of documents
        hf_embeddings (HuggingFaceInferenceAPIEmbeddings): Hugging Face embeddings
    Returns:
        FAISS: FAISS index
    """
    try:
        vector_store = FAISS.from_documents(
            documents=documents,
            embedding=hf_embeddings
        )
        vector_store.save_local(index_path)
        return vector_store
    except Exception as e:
        print(f"Error in load_faiss_vectorstore: {e}")
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

